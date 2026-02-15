"""Reactive Diffusion Policy — Asymmetric Tokenizer (AT) for LeRobot.

This module implements the Asymmetric Tokenizer (VAE/VQ-VAE) from the
Reactive Diffusion Policy paper (Xue et al., RSS 2025).  The AT learns a
compressed latent action space that can be decoded with optional temporal
conditioning from tactile / force observations.

Training:
    python -m lerobot.scripts.lerobot_train \
        --policy.type=rdp_tokenizer \
        --dataset.repo_id=<user>/<dataset> \
        --policy.horizon=32 \
        --policy.n_latent_dims=4
"""

from lerobot.policies.rdp_tokenizer.configuration_rdp_tokenizer import RDPTokenizerConfig
from lerobot.policies.rdp_tokenizer.modeling_rdp_tokenizer import RDPTokenizerPolicy
from lerobot.policies.rdp_tokenizer.processor_rdp_tokenizer import (
    make_rdp_tokenizer_pre_post_processors,
)

__all__ = [
    "RDPTokenizerConfig",
    "RDPTokenizerPolicy",
    "make_rdp_tokenizer_pre_post_processors",
]
