"""Configuration for the Reactive Diffusion Policy Latent Diffusion Policy."""

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("rdp_latent_diffusion")
@dataclass
class RDPLatentDiffusionConfig(PreTrainedConfig):
    """Configuration for the Latent Diffusion Policy from Reactive Diffusion Policy.

    This policy runs a 1-D conditional UNet diffusion model in the latent
    action space of a frozen Asymmetric Tokenizer (AT).  At inference time the
    predicted latent is decoded back to the original action space via the AT
    decoder (optionally using RNN temporal conditioning).

    The config duplicates several AT parameters so that the AT can be
    reconstructed when loading from a standalone checkpoint.

    Args:
        pretrained_tokenizer_path: Path to a pre-trained AT checkpoint
            (``pretrained_model/`` directory from an ``rdp_tokenizer`` training
            run, or a HuggingFace Hub repo ID).
        use_latent_action_before_vq: When the AT uses VQ, whether to run
            diffusion on the pre-quantisation latent (True) or the
            post-quantisation latent (False, default).
    """

    # Temporal structure
    n_obs_steps: int = 2
    horizon: int = 32
    n_action_steps: int = 29

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )
    drop_n_last_frames: int = 0

    # AT reference ---------------------------------------------------------
    pretrained_tokenizer_path: str | Path | None = None
    use_latent_action_before_vq: bool = False

    # AT architecture (needed to reconstruct the AT when it isn't loaded
    # from a full PreTrainedPolicy checkpoint)
    at_encoder_type: str = "conv1d"
    at_n_latent_dims: int = 4
    at_encoder_hidden_dim: int = 32
    at_encoder_n_layers: int = 1
    at_decoder_type: str = "rnn"
    at_decoder_hidden_dim: int = 32
    at_decoder_n_layers: int = 1
    at_use_vq: bool = False
    at_n_embed: int = 32
    at_vqvae_groups: int = 4
    at_act_scale: float = 1.0
    at_temporal_cond_keys: tuple[str, ...] = ()

    # Vision backbone ------------------------------------------------------
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    resize_shape: tuple[int, int] | None = None
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False

    # UNet -----------------------------------------------------------------
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True

    # Noise scheduler ------------------------------------------------------
    noise_scheduler_type: str = "DDIM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    num_inference_steps: int | None = None

    # Loss
    do_mask_loss_for_padding: bool = False

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(f"vision_backbone must be a ResNet variant, got {self.vision_backbone}")

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("LDP requires at least one image or environment_state input feature.")
        if self.crop_shape is not None:
            for key, ft in self.image_features.items():
                if self.crop_shape[0] > ft.shape[1] or self.crop_shape[1] > ft.shape[2]:
                    raise ValueError(
                        f"crop_shape {self.crop_shape} does not fit image shape {ft.shape} for '{key}'"
                    )
        if len(self.image_features) > 0:
            first_shape = next(iter(self.image_features.values())).shape
            for key, ft in self.image_features.items():
                if ft.shape != first_shape:
                    raise ValueError(f"All images must have the same shape; '{key}' differs.")

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
