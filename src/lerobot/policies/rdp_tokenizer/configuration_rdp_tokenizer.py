"""Configuration for the Reactive Diffusion Policy Asymmetric Tokenizer."""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("rdp_tokenizer")
@dataclass
class RDPTokenizerConfig(PreTrainedConfig):
    """Configuration for the Asymmetric Tokenizer (AT) from Reactive Diffusion Policy.

    The AT is a VAE / VQ-VAE that learns a compressed latent action space.
    The encoder compresses an action chunk into a low-dimensional latent; the
    decoder reconstructs the full action trajectory, optionally conditioned on
    temporal tactile / force observations (via an RNN).

    Args:
        horizon: Length of the action chunk to encode / decode.
        n_obs_steps: Number of observation history steps (unused for AT training
            but required by the LeRobot interface).
        n_action_steps: Number of action steps to execute from a decoded chunk.
        n_latent_dims: Dimensionality of the latent code produced by the encoder.
        encoder_type: ``"mlp"`` or ``"conv1d"``.
        encoder_hidden_dim: Hidden dimension for encoder layers.
        encoder_n_layers: Number of hidden layers in the encoder.
        decoder_type: ``"mlp"`` or ``"rnn"``.  RNN enables temporal conditioning.
        decoder_hidden_dim: Hidden dimension for decoder layers.
        decoder_n_layers: Number of hidden layers / RNN layers.
        use_vq: If True use Residual VQ; otherwise Gaussian VAE.
        n_embed: For VQ: codebook size.  For VAE: half of the ``quant`` Conv1d
            output channels (the other half is the log-variance).
        vqvae_groups: Number of residual VQ quantizer layers.
        kl_multiplier: Weight of the KL term when ``use_vq=False``.
        encoder_loss_multiplier: Weight of the reconstruction L1 term.
        vq_loss_multiplier: Weight of the VQ commitment loss (when ``use_vq=True``).
        act_scale: Divisor applied to normalised actions before encoding.
        temporal_cond_keys: Observation keys whose full-horizon values are fed as
            temporal conditioning to the RNN decoder.  These must appear in the
            dataset as ``observation.<key>`` features.
    """

    # Temporal structure
    horizon: int = 32
    n_obs_steps: int = 2
    n_action_steps: int = 29

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Encoder
    encoder_type: str = "conv1d"  # "mlp" | "conv1d"
    n_latent_dims: int = 4
    encoder_hidden_dim: int = 32
    encoder_n_layers: int = 1

    # Decoder
    decoder_type: str = "mlp"  # "mlp" | "rnn"
    decoder_hidden_dim: int = 32
    decoder_n_layers: int = 1

    # Quantization
    use_vq: bool = False
    n_embed: int = 32
    vqvae_groups: int = 4

    # Loss weights
    kl_multiplier: float = 1e-6
    encoder_loss_multiplier: float = 1.0
    vq_loss_multiplier: float = 5.0
    act_scale: float = 1.0

    # Temporal conditioning keys for RNN decoder
    temporal_cond_keys: tuple[str, ...] = ()

    # Training presets
    optimizer_lr: float = 1e-3
    optimizer_weight_decay: float = 1e-4
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 100

    # No need to drop frames – AT does pure reconstruction
    drop_n_last_frames: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.encoder_type not in ("mlp", "conv1d"):
            raise ValueError(f"encoder_type must be 'mlp' or 'conv1d', got {self.encoder_type}")
        if self.decoder_type not in ("mlp", "rnn"):
            raise ValueError(f"decoder_type must be 'mlp' or 'rnn', got {self.decoder_type}")
        if self.decoder_type == "rnn" and len(self.temporal_cond_keys) == 0:
            raise ValueError("RNN decoder requires at least one temporal_cond_keys entry")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if self.action_feature is None:
            raise ValueError("RDP Tokenizer requires an 'action' output feature.")

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def observation_delta_indices_per_key(self) -> dict[str, list] | None:
        if self.decoder_type == "rnn" and len(self.temporal_cond_keys) > 0:
            full_horizon = list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))
            per_key = {}
            for key in self.temporal_cond_keys:
                feat_key = f"observation.{key}" if not key.startswith("observation.") else key
                per_key[feat_key] = full_horizon
            return per_key
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
