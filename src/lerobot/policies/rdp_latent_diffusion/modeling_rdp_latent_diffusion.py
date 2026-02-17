"""Latent Diffusion Policy (LDP) for LeRobot.

Ported from ``reactive_diffusion_policy/policy/latent_diffusion_unet_image_policy.py``.
The policy reuses the vision encoder and 1-D UNet from the standard LeRobot
Diffusion Policy, but operates in the latent action space of a frozen
Asymmetric Tokenizer (AT).
"""

import logging
from collections import deque
from pathlib import Path

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn

from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    DiffusionRgbEncoder,
    _make_noise_scheduler,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rdp_latent_diffusion.configuration_rdp_latent_diffusion import (
    RDPLatentDiffusionConfig,
)
from lerobot.policies.rdp_tokenizer.configuration_rdp_tokenizer import RDPTokenizerConfig
from lerobot.policies.rdp_tokenizer.modeling_rdp_tokenizer import RDPTokenizerPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

logger = logging.getLogger(__name__)


class RDPLatentDiffusionPolicy(PreTrainedPolicy):
    """Latent Diffusion Policy from Reactive Diffusion Policy (Xue et al., RSS 2025).

    Stage 2 of RDP training: a conditional 1-D UNet denoises latent actions
    produced by a frozen Asymmetric Tokenizer.
    """

    config_class = RDPLatentDiffusionConfig
    name = "rdp_latent_diffusion"

    def __init__(self, config: RDPLatentDiffusionConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # ---- Build / load the frozen AT ----
        self.at = self._build_tokenizer(config)
        for p in self.at.parameters():
            p.requires_grad = False
        self.at.eval()

        # Derive the latent action dimension & latent horizon
        if config.at_use_vq:
            latent_action_dim = config.at_n_latent_dims
        else:
            latent_action_dim = config.at_n_embed
        self.latent_action_dim = latent_action_dim
        self.latent_horizon = self.at.downsampled_input_h

        # Adjust UNet kernel size based on encoder type
        if config.at_encoder_type == "conv1d":
            effective_kernel = min(config.kernel_size, 3)
        else:
            effective_kernel = 1

        # ---- Build DiffusionModel (obs encoder + UNet + scheduler) ----
        self.diffusion = _LatentDiffusionModel(
            config,
            latent_action_dim=latent_action_dim,
            latent_horizon=self.latent_horizon,
            effective_kernel=effective_kernel,
        )

        self._queues = None
        self.reset()

    @staticmethod
    def _build_tokenizer(config: RDPLatentDiffusionConfig) -> RDPTokenizerPolicy:
        """Instantiate an RDPTokenizerPolicy from AT-related config fields.

        If ``pretrained_tokenizer_path`` is set, load weights from that checkpoint.
        """
        at_cfg = RDPTokenizerConfig(
            horizon=config.horizon,
            n_obs_steps=config.n_obs_steps,
            n_action_steps=config.n_action_steps,
            encoder_type=config.at_encoder_type,
            n_latent_dims=config.at_n_latent_dims,
            encoder_hidden_dim=config.at_encoder_hidden_dim,
            encoder_n_layers=config.at_encoder_n_layers,
            decoder_type=config.at_decoder_type,
            decoder_hidden_dim=config.at_decoder_hidden_dim,
            decoder_n_layers=config.at_decoder_n_layers,
            use_vq=config.at_use_vq,
            n_embed=config.at_n_embed,
            vqvae_groups=config.at_vqvae_groups,
            act_scale=config.at_act_scale,
            temporal_cond_keys=config.at_temporal_cond_keys,
            # Copy feature definitions so the AT can validate
            input_features=config.input_features,
            output_features=config.output_features,
        )

        at_policy = RDPTokenizerPolicy(at_cfg)

        if config.pretrained_tokenizer_path is not None:
            path = Path(config.pretrained_tokenizer_path)
            if path.is_dir():
                # HuggingFace-style directory
                loaded = RDPTokenizerPolicy.from_pretrained(str(path))
                at_policy.load_state_dict(loaded.state_dict())
            else:
                # Single file checkpoint
                state = torch.load(str(path), map_location="cpu", weights_only=False)
                if "state_dicts" in state:
                    state = state["state_dicts"]["model"]
                at_policy.load_state_dict(state, strict=False)
            logger.info("Loaded AT checkpoint from %s", config.pretrained_tokenizer_path)

        return at_policy

    def state_dict(self, *args, **kwargs):
        # The frozen AT contains an nn.GRU whose weights share a flattened
        # cuDNN buffer.  Clone all tensors so safetensors can save them.
        sd = super().state_dict(*args, **kwargs)
        return {k: v.clone() for k, v in sd.items()}

    def get_optim_params(self):
        # Only train the diffusion model, not the frozen AT
        return self.diffusion.parameters()

    def reset(self):
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    # ---- Latent statistics ----

    @torch.no_grad()
    def compute_latent_stats(self, dataset) -> None:
        """Compute min/max of latent actions across the dataset.

        This must be called before training so that latent actions are normalized
        to [-1, 1] for the diffusion model.  Results are stored as registered
        buffers and automatically persist in checkpoints.
        """
        if self.diffusion.latent_stats_computed:
            logger.info("Latent stats already computed (loaded from checkpoint), skipping.")
            return

        logger.info("Computing latent action statistics over the dataset...")
        device = next(self.parameters()).device
        self.at.to(device)

        # Fetch action stats for inline normalization
        stats = dataset.meta.stats
        action_min = torch.tensor(stats[ACTION]["min"], device=device, dtype=torch.float32)
        action_max = torch.tensor(stats[ACTION]["max"], device=device, dtype=torch.float32)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=64, num_workers=4, shuffle=False, drop_last=False,
        )

        running_min = None
        running_max = None
        n_batches = 0

        for batch in dataloader:
            actions = batch[ACTION].to(device, dtype=torch.float32)

            # Normalize actions to [-1, 1] using dataset stats (same as the
            # preprocessor does during training)
            denom = action_max - action_min
            eps = 1e-8
            denom = torch.where(denom == 0, torch.tensor(eps, device=device), denom)
            actions = 2 * (actions - action_min) / denom - 1

            latent = self.diffusion._encode_to_latent(actions, self.at, self.config)
            batch_min = latent.flatten(0, 1).min(dim=0).values
            batch_max = latent.flatten(0, 1).max(dim=0).values

            if running_min is None:
                running_min = batch_min
                running_max = batch_max
            else:
                running_min = torch.min(running_min, batch_min)
                running_max = torch.max(running_max, batch_max)
            n_batches += 1

        self.diffusion.latent_min.data.copy_(running_min)
        self.diffusion.latent_max.data.copy_(running_max)
        self.diffusion.latent_stats_computed = True

        logger.info(
            "Latent stats computed over %d batches: min=%s, max=%s",
            n_batches, running_min.tolist(), running_max.tolist(),
        )

    # ---- Inference ----

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        return self.diffusion.generate_actions(
            batch,
            at=self.at,
            config=self.config,
            noise=noise,
        )

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        if ACTION in batch:
            batch.pop(ACTION)
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()

    # ---- Training ----

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        loss = self.diffusion.compute_loss(batch, at=self.at, config=self.config)
        return loss, None


# ---------------------------------------------------------------------------
# Internal diffusion model
# ---------------------------------------------------------------------------

class _LatentDiffusionModel(nn.Module):
    """Wraps vision encoder + UNet + noise scheduler for latent diffusion."""

    def __init__(
        self,
        config: RDPLatentDiffusionConfig,
        latent_action_dim: int,
        latent_horizon: int,
        effective_kernel: int,
    ):
        super().__init__()
        self.config = config
        self.latent_action_dim = latent_action_dim
        self.latent_horizon = latent_horizon

        # --- Observation encoder ---
        global_cond_dim = config.robot_state_feature.shape[0]
        if config.image_features:
            num_images = len(config.image_features)
            if config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if config.env_state_feature:
            global_cond_dim += config.env_state_feature.shape[0]

        # --- UNet ---
        # Override config fields for the UNet that differ from standard DP
        # We create a lightweight shim config to reuse DiffusionConditionalUnet1d
        unet_config = _UnetShimConfig(
            action_dim=latent_action_dim,
            down_dims=config.down_dims,
            kernel_size=effective_kernel,
            n_groups=config.n_groups,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            use_film_scale_modulation=config.use_film_scale_modulation,
            horizon=latent_horizon,
        )
        self.unet = _build_unet(unet_config, global_cond_dim=global_cond_dim * config.n_obs_steps)

        # If the latent horizon is 1, remove downsampling/upsampling
        if latent_horizon == 1:
            for module_list in self.unet.down_modules:
                module_list[-1] = nn.Identity()
            for module_list in self.unet.up_modules:
                module_list[-1] = nn.Identity()

        # --- Latent action normalization buffers ---
        # These are populated by RDPLatentDiffusionPolicy.compute_latent_stats()
        # before training begins.  Once computed they are saved/loaded via state_dict.
        self.register_buffer("latent_min", torch.zeros(1))
        self.register_buffer("latent_max", torch.ones(1))
        self.register_buffer("_latent_stats_flag", torch.tensor(0, dtype=torch.bool))

        # --- Noise scheduler ---
        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )
        if config.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

    @property
    def latent_stats_computed(self) -> bool:
        return bool(self._latent_stats_flag.item())

    @latent_stats_computed.setter
    def latent_stats_computed(self, value: bool):
        self._latent_stats_flag.fill_(value)

    # --- Latent normalization (MIN_MAX to [-1, 1]) ---

    def normalize_latent(self, x: Tensor) -> Tensor:
        """Map latent actions from [latent_min, latent_max] to [-1, 1]."""
        denom = self.latent_max - self.latent_min
        eps = 1e-8
        denom = torch.where(denom == 0, torch.tensor(eps, device=x.device, dtype=x.dtype), denom)
        return 2 * (x - self.latent_min) / denom - 1

    def unnormalize_latent(self, x: Tensor) -> Tensor:
        """Map latent actions from [-1, 1] back to [latent_min, latent_max]."""
        denom = self.latent_max - self.latent_min
        return (x + 1) / 2 * denom + self.latent_min

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        feats = [batch[OBS_STATE]]
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_cam = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_feats = torch.cat(
                    [enc(imgs) for enc, imgs in zip(self.rgb_encoder, images_per_cam, strict=True)]
                )
                img_feats = einops.rearrange(img_feats, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps)
            else:
                img_feats = self.rgb_encoder(einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ..."))
                img_feats = einops.rearrange(img_feats, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps)
            feats.append(img_feats)
        if self.config.env_state_feature:
            feats.append(batch[OBS_ENV_STATE])
        return torch.cat(feats, dim=-1).flatten(start_dim=1)

    def _encode_to_latent(
        self,
        actions: Tensor,
        at: RDPTokenizerPolicy,
        config: RDPLatentDiffusionConfig,
    ) -> Tensor:
        """Encode raw actions to latent via frozen AT encoder + quantization."""
        state = at._preprocess_actions(actions / config.at_act_scale)
        state_rep = at.encoder(state)

        if config.at_use_vq:
            if not config.use_latent_action_before_vq:
                state_vq, _, _ = at._quant_with_vq(state_rep)
            else:
                if config.at_encoder_type == "conv1d":
                    state_vq = einops.rearrange(state_rep, "N T A -> N (T A)")
                else:
                    state_vq = state_rep
        else:
            state_vq, _ = at._quant_without_vq(state_rep)

        return einops.rearrange(state_vq, "N (T A) -> N T A", T=self.latent_horizon)

    def _decode_from_latent(
        self,
        latent: Tensor,
        batch: dict[str, Tensor],
        at: RDPTokenizerPolicy,
        config: RDPLatentDiffusionConfig,
    ) -> Tensor:
        """Decode latent actions back to original action space via frozen AT decoder."""
        latent_flat = einops.rearrange(latent, "N T A -> N (T A)")

        if config.at_use_vq:
            if config.use_latent_action_before_vq:
                state_vq, _, _ = at._quant_with_vq(latent_flat)
            else:
                state_vq = latent_flat
        else:
            state_vq = latent_flat
            state_vq = at._postprocess_quant_without_vq(state_vq)

        if config.at_decoder_type == "rnn":
            temporal_cond = at._gather_temporal_cond(batch)
            dec_out = at.decoder(state_vq, temporal_cond)
        else:
            dec_out = at.decoder(state_vq)

        return einops.rearrange(dec_out * config.at_act_scale, "N (T A) -> N T A", A=at.action_dim)

    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        sample = (
            noise
            if noise is not None
            else torch.randn(
                (batch_size, self.latent_horizon, self.latent_action_dim),
                dtype=dtype,
                device=device,
            )
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            sample = self.noise_scheduler.step(model_output, t, sample).prev_sample

        return sample

    def generate_actions(
        self,
        batch: dict[str, Tensor],
        at: RDPTokenizerPolicy,
        config: RDPLatentDiffusionConfig,
        noise: Tensor | None = None,
    ) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond = self._prepare_global_conditioning(batch)
        latent = self.conditional_sample(batch_size, global_cond=global_cond, noise=noise)
        if self.latent_stats_computed:
            latent = self.unnormalize_latent(latent)
        actions = self._decode_from_latent(latent, batch, at, config)

        # Extract the action window
        start = n_obs_steps - 1
        end = start + config.n_action_steps
        return actions[:, start:end]

    def compute_loss(
        self,
        batch: dict[str, Tensor],
        at: RDPTokenizerPolicy,
        config: RDPLatentDiffusionConfig,
    ) -> Tensor:
        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
        n_obs_steps = batch[OBS_STATE].shape[1]
        assert n_obs_steps == config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)

        # Encode actions to latent and normalize to [-1, 1]
        with torch.no_grad():
            latent_actions = self._encode_to_latent(batch[ACTION], at, config)
            if self.latent_stats_computed:
                latent_actions = self.normalize_latent(latent_actions)

        # Forward diffusion
        eps = torch.randn(latent_actions.shape, device=latent_actions.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (latent_actions.shape[0],), device=latent_actions.device,
        ).long()
        noisy = self.noise_scheduler.add_noise(latent_actions, eps, timesteps)

        pred = self.unet(noisy, timesteps, global_cond=global_cond)

        if config.prediction_type == "epsilon":
            target = eps
        elif config.prediction_type == "sample":
            target = latent_actions
        else:
            raise ValueError(f"Unsupported prediction_type: {config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        if config.do_mask_loss_for_padding and "action_is_pad" in batch:
            # Broadcast padding mask to latent shape
            # action_is_pad: (B, horizon) → need to downsample if latent_horizon < horizon
            pad_mask = batch["action_is_pad"]
            if pad_mask.shape[1] != self.latent_horizon:
                # Pool the padding mask to match latent horizon
                pad_mask = F.adaptive_max_pool1d(
                    pad_mask.float().unsqueeze(1), self.latent_horizon
                ).squeeze(1).bool()
            loss = loss * (~pad_mask).unsqueeze(-1)

        return loss.mean()


# ---------------------------------------------------------------------------
# Shim to reuse DiffusionConditionalUnet1d
# ---------------------------------------------------------------------------

class _UnetShimConfig:
    """Minimal config object that DiffusionConditionalUnet1d expects."""

    def __init__(
        self,
        action_dim: int,
        down_dims: tuple,
        kernel_size: int,
        n_groups: int,
        diffusion_step_embed_dim: int,
        use_film_scale_modulation: bool,
        horizon: int,
    ):
        from lerobot.configs.types import PolicyFeature, FeatureType
        self.action_feature = PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))
        self.down_dims = down_dims
        self.kernel_size = kernel_size
        self.n_groups = n_groups
        self.diffusion_step_embed_dim = diffusion_step_embed_dim
        self.use_film_scale_modulation = use_film_scale_modulation
        self.horizon = horizon


def _build_unet(shim_config: _UnetShimConfig, global_cond_dim: int) -> DiffusionConditionalUnet1d:
    return DiffusionConditionalUnet1d(shim_config, global_cond_dim=global_cond_dim)
