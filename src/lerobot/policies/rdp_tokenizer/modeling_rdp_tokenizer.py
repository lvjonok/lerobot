"""Asymmetric Tokenizer (AT) policy for LeRobot.

Ported from ``reactive_diffusion_policy/model/vae/model.py``.
"""

from collections import deque

import einops
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rdp_tokenizer.configuration_rdp_tokenizer import RDPTokenizerConfig
from lerobot.policies.utils import get_output_shape, populate_queues
from lerobot.utils.constants import ACTION, OBS_STATE


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

def _weights_init_encoder(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.orthogonal_(m.weight.data.view(m.weight.size(0), -1))
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class _MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 16, hidden_dim: int = 128, layer_num: int = 1):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(layer_num):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.apply(_weights_init_encoder)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(self.encoder(x))


class _EncoderCNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 16, hidden_dim: int = 128, layer_num: int = 1):
        super().__init__()
        self.action_dim = input_dim
        layers: list[nn.Module] = []
        for i in range(layer_num):
            in_ch = input_dim if i == 0 else hidden_dim
            layers += [nn.Conv1d(in_ch, hidden_dim, kernel_size=5, stride=2, padding=2), nn.ReLU()]
        layers.append(nn.Conv1d(hidden_dim, output_dim, kernel_size=5, stride=2, padding=2))
        self.encoder = nn.Sequential(*layers)
        self.apply(_weights_init_encoder)

    def forward(self, x: Tensor, flatten: bool = False) -> Tensor:
        x = einops.rearrange(x, "N (T A) -> N A T", A=self.action_dim)
        h = self.encoder(x)
        h = einops.rearrange(h, "N C T -> N T C")
        if flatten:
            h = einops.rearrange(h, "N T C -> N (T C)")
        return h


class _DecoderRNN(nn.Module):
    def __init__(
        self,
        global_cond_dim: int,
        temporal_cond_dim: int,
        output_dim: int,
        hidden_dim: int,
        layer_num: int = 1,
    ):
        super().__init__()
        self.rnn = nn.GRU(global_cond_dim + temporal_cond_dim, hidden_dim, layer_num, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.apply(_weights_init_encoder)

    def forward(self, global_cond: Tensor, temporal_cond: Tensor) -> Tensor:
        global_cond = global_cond.unsqueeze(1).expand(-1, temporal_cond.size(1), -1)
        x = torch.cat([global_cond, temporal_cond], dim=-1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return einops.rearrange(x, "N T A -> N (T A)")


# ---------------------------------------------------------------------------
# Diagonal Gaussian (for VAE mode)
# ---------------------------------------------------------------------------

class _DiagonalGaussianDistribution:
    def __init__(self, parameters: Tensor):
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self) -> Tensor:
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self) -> Tensor:
        dims = list(range(1, self.mean.dim()))
        return 0.5 * torch.sum(self.mean.pow(2) + self.var - 1.0 - self.logvar, dim=dims)


# ---------------------------------------------------------------------------
# Residual VQ (thin wrapper around vector-quantize-pytorch)
# ---------------------------------------------------------------------------

class _ResidualVQ(nn.Module):
    """Minimal Residual Vector Quantization layer.

    Uses ``vector_quantize_pytorch`` if available, otherwise a simple
    straight-through codebook.
    """

    def __init__(self, dim: int, num_quantizers: int = 4, codebook_size: int = 32):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebooks = nn.ParameterList(
            [nn.Parameter(torch.randn(codebook_size, dim)) for _ in range(num_quantizers)]
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """x: (B, T, D) → quantized, codes, commitment_loss."""
        residual = x
        quantized = torch.zeros_like(x)
        all_codes = []
        total_loss = torch.tensor(0.0, device=x.device)
        for cb in self.codebooks:
            # (B, T, D) vs (K, D)
            flat = residual.reshape(-1, residual.size(-1))
            dist = torch.cdist(flat, cb)
            codes = dist.argmin(dim=-1)
            all_codes.append(codes)
            q = cb[codes].reshape_as(residual)
            # straight-through
            quantized = quantized + residual + (q - residual).detach()
            total_loss = total_loss + ((q.detach() - residual).pow(2).mean() + (q - residual.detach()).pow(2).mean())
            residual = residual - q.detach()
        codes = torch.stack(all_codes, dim=-1)
        return quantized, codes, total_loss


# ---------------------------------------------------------------------------
# Main policy
# ---------------------------------------------------------------------------

class RDPTokenizerPolicy(PreTrainedPolicy):
    """Asymmetric Tokenizer policy for training only.

    ``forward()`` computes reconstruction + regularization loss.
    ``select_action()`` and ``predict_action_chunk()`` are not meaningful for
    standalone AT usage — the trained checkpoint is loaded by the Latent
    Diffusion Policy for inference.
    """

    config_class = RDPTokenizerConfig
    name = "rdp_tokenizer"

    def __init__(self, config: RDPTokenizerConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        action_dim = config.action_feature.shape[0]
        self.action_dim = action_dim
        self.horizon = config.horizon

        # --- Encoder ---
        if config.encoder_type == "conv1d":
            self.encoder = _EncoderCNN(
                input_dim=action_dim,
                output_dim=config.n_latent_dims,
                hidden_dim=config.encoder_hidden_dim,
                layer_num=config.encoder_n_layers,
            )
        else:
            self.encoder = _MLP(
                input_dim=action_dim * config.horizon,
                output_dim=config.n_latent_dims,
                hidden_dim=config.encoder_hidden_dim,
                layer_num=config.encoder_n_layers,
            )

        # Compute the latent shape after encoding
        dummy = torch.zeros(1, action_dim * config.horizon)
        with torch.no_grad():
            enc_out = self.encoder(dummy)
        if enc_out.dim() == 2:
            decoder_latent_dim = enc_out.shape[-1]
            self.downsampled_input_h = 1
        else:
            decoder_latent_dim = int(np.prod(enc_out.shape[1:]))
            self.downsampled_input_h = enc_out.shape[1]

        # --- Decoder ---
        temporal_cond_dim = 0
        if config.decoder_type == "rnn":
            for key in config.temporal_cond_keys:
                feat_key = f"observation.{key}" if not key.startswith("observation.") else key
                if feat_key in config.input_features:
                    temporal_cond_dim += config.input_features[feat_key].shape[0]
                else:
                    raise ValueError(
                        f"temporal_cond_key '{key}' not found in input_features. "
                        f"Available: {list(config.input_features.keys())}"
                    )
            self.temporal_cond_dim = temporal_cond_dim
            self.decoder = _DecoderRNN(
                global_cond_dim=decoder_latent_dim,
                temporal_cond_dim=temporal_cond_dim,
                output_dim=action_dim,
                hidden_dim=config.decoder_hidden_dim,
                layer_num=config.decoder_n_layers,
            )
        else:
            self.temporal_cond_dim = 0
            self.decoder = _MLP(
                input_dim=decoder_latent_dim,
                output_dim=action_dim * config.horizon,
                hidden_dim=config.decoder_hidden_dim,
                layer_num=config.decoder_n_layers,
            )

        # --- Quantization ---
        if config.use_vq:
            self.vq_layer = _ResidualVQ(
                dim=config.n_latent_dims,
                num_quantizers=config.vqvae_groups,
                codebook_size=config.n_embed,
            )
        else:
            self.quant = nn.Conv1d(config.n_latent_dims, 2 * config.n_embed, 1)
            self.post_quant = nn.Conv1d(config.n_embed, config.n_latent_dims, 1)

        self._queues = None
        self.reset()

    def state_dict(self, *args, **kwargs):
        # nn.GRU flattens weights into a shared cuDNN buffer on CUDA.
        # Cloning breaks the shared storage so safetensors can save them.
        sd = super().state_dict(*args, **kwargs)
        return {k: v.clone() for k, v in sd.items()}

    def get_optim_params(self):
        return self.parameters()

    def reset(self):
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.robot_state_feature:
            self._queues[OBS_STATE] = deque(maxlen=self.config.n_obs_steps)

    # ---- Quantization helpers (mirror original VAE API) ----

    def _preprocess_actions(self, actions: Tensor) -> Tensor:
        """Flatten action chunk for encoder input."""
        return einops.rearrange(actions, "N T A -> N (T A)")

    def _quant_with_vq(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if state.dim() == 2:
            state = einops.rearrange(state, "N (T A) -> N T A", T=self.downsampled_input_h)
        state_vq, codes, loss = self.vq_layer(state)
        return state_vq.reshape(state.size(0), -1), codes.reshape(state.size(0), -1), loss

    def _quant_without_vq(self, state: Tensor) -> tuple[Tensor, _DiagonalGaussianDistribution]:
        if state.dim() == 2:
            state = einops.rearrange(state, "N (T A) -> N A T", T=self.downsampled_input_h)
        else:
            state = einops.rearrange(state, "N T A -> N A T")
        moments = self.quant(state)
        posterior = _DiagonalGaussianDistribution(moments)
        state_vq = posterior.sample()
        state_vq = einops.rearrange(state_vq, "N A T -> N (T A)")
        return state_vq, posterior

    def _postprocess_quant_without_vq(self, state_vq: Tensor) -> Tensor:
        state_vq = einops.rearrange(state_vq, "N (T A) -> N A T", T=self.downsampled_input_h)
        state_vq = self.post_quant(state_vq)
        return einops.rearrange(state_vq, "N A T -> N (T A)")

    # ---- Core API ----

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """Compute reconstruction + regularisation loss."""
        actions = batch[ACTION]  # (B, horizon, action_dim)
        state = self._preprocess_actions(actions / self.config.act_scale)

        # Encode
        state_rep = self.encoder(state)

        # Quantize
        if self.config.use_vq:
            state_vq, vq_code, vq_loss = self._quant_with_vq(state_rep)
        else:
            state_vq, posterior = self._quant_without_vq(state_rep)
            state_vq = self._postprocess_quant_without_vq(state_vq)

        # Decode
        if self.config.decoder_type == "rnn":
            temporal_cond = self._gather_temporal_cond(batch)
            dec_out = self.decoder(state_vq, temporal_cond)
        else:
            dec_out = self.decoder(state_vq)

        # Losses
        recon_l1 = (state - dec_out).abs().mean()
        recon_mse = torch.nn.functional.mse_loss(state, dec_out)

        info = {
            "recon_l1": recon_l1.item(),
            "recon_mse": recon_mse.item(),
        }

        if self.config.use_vq:
            loss = recon_l1 * self.config.encoder_loss_multiplier + vq_loss * self.config.vq_loss_multiplier
            info["vq_loss"] = vq_loss.item()
        else:
            kl_loss = posterior.kl().mean()
            loss = recon_l1 * self.config.encoder_loss_multiplier + kl_loss * self.config.kl_multiplier
            info["kl_loss"] = kl_loss.item()

        info["total_loss"] = loss.item()
        return loss, info

    def _gather_temporal_cond(self, batch: dict[str, Tensor]) -> Tensor:
        """Collect and concatenate temporal conditioning tensors for the RNN decoder."""
        parts = []
        for key in self.config.temporal_cond_keys:
            feat_key = f"observation.{key}" if not key.startswith("observation.") else key
            parts.append(batch[feat_key])
        return torch.cat(parts, dim=-1)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Not meaningful for standalone AT — encodes then decodes the *given* actions."""
        raise NotImplementedError(
            "RDPTokenizerPolicy is a training-only policy.  Use RDPLatentDiffusionPolicy for inference."
        )

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError(
            "RDPTokenizerPolicy is a training-only policy.  Use RDPLatentDiffusionPolicy for inference."
        )
