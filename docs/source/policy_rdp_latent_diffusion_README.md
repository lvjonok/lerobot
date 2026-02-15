## Paper

Reactive Diffusion Policy (Xue et al., RSS 2025)

## Overview

The Latent Diffusion Policy (LDP) is the second stage of the Reactive Diffusion Policy. It runs a conditional 1-D UNet diffusion model in the latent action space of a frozen Asymmetric Tokenizer (AT). At inference time, the predicted latent is decoded back to the original action space via the AT decoder.

Requires a pre-trained `rdp_tokenizer` checkpoint. See [TRAINING.md](../../TRAINING.md) for the two-stage training workflow.

## Citation

```bibtex
@inproceedings{xue2025reactive,
  title={Reactive Diffusion Policy},
  author={Xue, Wenbo and Bing, Zhenshan and Li, Haodong and Knoll, Alois},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2025},
}
```
