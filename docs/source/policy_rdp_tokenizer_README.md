## Paper

Reactive Diffusion Policy (Xue et al., RSS 2025)

## Overview

The Asymmetric Tokenizer (AT) is the first stage of the Reactive Diffusion Policy. It compresses action trajectories into a low-dimensional latent space using a VAE or VQ-VAE architecture. The encoder is simple (MLP or Conv1D), while the decoder can be an RNN conditioned on per-step temporal observations (e.g. tactile/force data), making the architecture "asymmetric".

This is a training-only policy. For inference, use `rdp_latent_diffusion` which loads a frozen AT and runs diffusion in the latent space.

See [TRAINING.md](../../TRAINING.md) for usage.

## Citation

```bibtex
@inproceedings{xue2025reactive,
  title={Reactive Diffusion Policy},
  author={Xue, Wenbo and Bing, Zhenshan and Li, Haodong and Knoll, Alois},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2025},
}
```
