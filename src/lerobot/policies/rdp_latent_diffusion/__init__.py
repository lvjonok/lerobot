"""Reactive Diffusion Policy — Latent Diffusion Policy (LDP) for LeRobot.

This module implements the second stage of the Reactive Diffusion Policy
(Xue et al., RSS 2025).  It runs a conditional 1-D diffusion UNet in the
latent action space of a *frozen* Asymmetric Tokenizer (AT).

Training (requires a pre-trained AT checkpoint):
    python -m lerobot.scripts.lerobot_train \
        --policy.type=rdp_latent_diffusion \
        --dataset.repo_id=<user>/<dataset> \
        --policy.pretrained_tokenizer_path=<path_to_at_checkpoint>
"""

from lerobot.policies.rdp_latent_diffusion.configuration_rdp_latent_diffusion import (
    RDPLatentDiffusionConfig,
)
from lerobot.policies.rdp_latent_diffusion.modeling_rdp_latent_diffusion import (
    RDPLatentDiffusionPolicy,
)
from lerobot.policies.rdp_latent_diffusion.processor_rdp_latent_diffusion import (
    make_rdp_latent_diffusion_pre_post_processors,
)

__all__ = [
    "RDPLatentDiffusionConfig",
    "RDPLatentDiffusionPolicy",
    "make_rdp_latent_diffusion_pre_post_processors",
]
