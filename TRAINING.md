# Training

This document covers training policies on real-world datasets collected with crisp_fastapi robots (Franka, Flexiv), including standard Diffusion Policy and the two-stage Reactive Diffusion Policy (RDP).

## Quick Start

```bash
# Train Diffusion Policy on 10Hz Franka dataset
python -m lerobot.scripts.lerobot_train \
    --policy.type=diffusion \
    --dataset.repo_id=domrachev03/franka_timing_belt_haply_static_v3_10hz \
    --policy.push_to_hub=false \
    --policy.crop_shape='[480,640]' \
    --num_epochs=600 \
    --wandb.enable=true \
    --wandb.project=rdp_timing_belt

# Resume training from checkpoint
python -m lerobot.scripts.lerobot_train \
    --config_path=outputs/train/.../checkpoints/last/pretrained_model/train_config.json \
    --resume=true
```

### Epoch-based training

LeRobot's training loop is step-based. If you prefer to specify the number of epochs instead, use `--num_epochs`:

```bash
--num_epochs=600
```

This overrides `steps` with `num_epochs * ceil(dataset_size / effective_batch_size)`, where `effective_batch_size = batch_size * num_gpu_processes`. The computed value is logged at startup.

## Configuration

LeRobot uses **draccus** (dataclass-based CLI parsing). All training configuration is defined in `TrainPipelineConfig`.

### TrainPipelineConfig

```python
@dataclass
class TrainPipelineConfig(HubMixin):
    # Core
    dataset: DatasetConfig                    # Dataset configuration
    policy: PreTrainedConfig | None = None    # Policy configuration

    # Training
    output_dir: Path | None = None            # Auto-generated if None
    job_name: str | None = None               # Run identifier
    resume: bool = False                      # Resume from checkpoint
    seed: int | None = 1000

    # Hyperparameters
    batch_size: int = 8
    num_workers: int = 4
    steps: int = 100_000
    num_epochs: int | None = None             # If set, overrides steps

    # Frequencies
    log_freq: int = 200                       # Logging frequency
    save_freq: int = 20_000                   # Checkpoint frequency

    # Optimizer / Scheduler
    use_policy_training_preset: bool = True    # Use policy's default optimizer/scheduler
    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None

    # Integrations
    wandb: WandBConfig = WandBConfig()

    # Observation renaming
    rename_map: dict[str, str] = {}           # Override observation keys
```

### DatasetConfig

```python
@dataclass
class DatasetConfig:
    repo_id: str                              # HuggingFace dataset repo ID
    root: str | None = None                   # Local cache directory
    episodes: list[int] | None = None         # Subset of episodes
    revision: str | None = None               # Git revision/tag
    use_imagenet_stats: bool = True            # Use ImageNet normalization for images
    video_backend: str = "torchcodec"         # Video decoder backend
    image_transforms: ImageTransformsConfig = ImageTransformsConfig()
```

### WandBConfig

```python
@dataclass
class WandBConfig:
    enable: bool = False
    project: str = "lerobot"
    entity: str | None = None
    notes: str | None = None
    run_id: str | None = None                 # Resume existing run
    mode: str | None = None                   # "online", "offline", "disabled"
    disable_artifact: bool = False
```

### CLI syntax for list/tuple fields

Draccus does not support Python tuple syntax in CLI arguments. Use JSON array syntax:

```bash
# Correct:
--policy.crop_shape='[480,640]'
--policy.resize_shape='[120,160]'
--policy.down_dims='[256,512,1024]'
--policy.temporal_cond_keys='[observation.effort]'

# WRONG (will fail):
--policy.crop_shape='(480, 640)'
```

## Dataset Observation Structure

Real-world datasets use grouped observation columns instead of a single flat `observation.state` vector. Each group is a separate dataset column:

| Column | Dims | Contents |
|---|---|---|
| `observation.state` | 8 | `tcp.pos`(3) + `tcp.quat`(4) + `gripper.pos`(1) |
| `observation.effort` | 6 | `ft_sensor.force`(3) + `ft_sensor.torque`(3) |
| `observation.joints` | 7 | `joint.pos`(7) |
| `observation.joint_vel` | 7 | `joint.vel`(7) |

All columns are mapped to `FeatureType.STATE` and loaded into the batch. However, each policy type uses different subsets:

- **Diffusion Policy** — conditions the UNet on `observation.state` only (`robot_state_feature` matches exactly `observation.state`). Other columns are loaded but unused by the model.
- **RDP Tokenizer** — the RNN decoder is conditioned per-step on columns listed in `temporal_cond_keys`. Typically `observation.effort` for force/torque reactivity.
- **RDP Latent Diffusion** — the UNet conditions on `observation.state` (same as DP). The frozen AT decoder conditions on `at_temporal_cond_keys`.

This grouping is supported by `hw_to_dataset_features()` in `datasets/utils.py`, which treats dict-valued entries in `observation_features` as named groups, each becoming a separate `observation.<group_name>` column.

## Dataset FPS Guidelines

Different policies work best at different frame rates. Datasets recorded at 30Hz (the default for `crisp_fastapi` recording configs) should be **downsampled** for standard Diffusion Policy:

| Policy | Recommended FPS | Reason |
|---|---|---|
| **Diffusion Policy** | 10 Hz | Lower FPS = longer planning horizon in fewer steps |
| **RDP (AT + LDP)** | 30 Hz | Needs high-frequency force/torque conditioning for reactivity |

Use the downsample script to create a 10Hz variant:

```bash
python -m lerobot.scripts.lerobot_downsample_dataset \
    --repo-id domrachev03/franka_timing_belt_haply_static_v3 \
    --target-fps 10 \
    --action-absolute-subfeatures gripper.pos
```

This creates a `*_10hz` dataset. Velocity action dims (`linear_vel`, `angular_vel`) are summed over each window to preserve total displacement. Absolute dims (`gripper.pos`) take the last value in each window.

**Important:** The target FPS must evenly divide the source FPS (e.g. 30 / 10 = 3).

When running a 10Hz policy on a 30Hz robot, use **action interpolation** to smooth the trajectory (see [REPLAY.md](REPLAY.md) and [INFERENCE.md](INFERENCE.md)).

## Training Diffusion Policy

```bash
python -m lerobot.scripts.lerobot_train \
    --policy.type=diffusion \
    --dataset.repo_id=domrachev03/franka_timing_belt_haply_static_v3_10hz \
    --policy.push_to_hub=false \
    --policy.down_dims='[256,512,1024]' \
    --policy.crop_shape='[480,640]' \
    --num_epochs=600 \
    --wandb.enable=true \
    --wandb.project=rdp_timing_belt
```

### Important notes

- **`push_to_hub=false`**: Required when `repo_id` is not set. Otherwise validation fails.
- **`down_dims`**: The UNet downsampling factor is `2^len(down_dims)`. The `horizon` must be divisible by this factor. Default `(512,1024,2048)` requires `horizon % 8 == 0`. Use `(256,512,1024)` for a smaller model.
- **`crop_shape`**: Set to match camera resolution from the recording config. For 640x480 cameras (see `configs/record/franka_haply.yaml`), use `'[480,640]'` (H, W) to train on full-size images. Set to `null` to disable cropping.
- **`resize_shape`**: Optional (H, W) target size using bilinear interpolation. Applied after cropping (if any). Use this to reduce image dimensions without cropping, e.g. `--policy.crop_shape=null --policy.resize_shape='[120,160]'`. Defaults to `None` (no resizing).
- **Dataset FPS**: Use a 10Hz downsampled dataset for DP (see above). The 30Hz dataset is for RDP.
- **Feature mapping**: The dataset's `observation.state` is mapped to `STATE` features and camera images to `VISUAL` features automatically via `dataset_to_policy_features()`.
- **State conditioning**: The UNet's `robot_state_feature` property matches only the column named exactly `observation.state` (8-dim: tcp + gripper). Other observation columns (`observation.effort`, `observation.joints`, `observation.joint_vel`) are present in `input_features` but are not read by the Diffusion model.

---

# Training Reactive Diffusion Policy (RDP)

This section covers training the two-stage Reactive Diffusion Policy (Xue et al., RSS 2025) using the LeRobot framework.

## Overview

RDP consists of two stages trained sequentially:

1. **Asymmetric Tokenizer (AT)** (`rdp_tokenizer`) — a VAE/VQ-VAE that compresses action chunks into a low-dimensional latent space. The encoder is simple (MLP or Conv1D); the decoder is optionally an RNN conditioned on temporal observations (e.g. force/torque data).
2. **Latent Diffusion Policy (LDP)** (`rdp_latent_diffusion`) — a conditional 1-D UNet diffusion model that operates in the latent action space of the frozen AT. It uses the same vision encoder architecture as the standard Diffusion Policy.

### Plugin discovery

RDP policies are not built into the LeRobot CLI's default policy choices. You **must** pass `--policy.discover_packages_path` to register them:

```bash
# For AT (Stage 1):
--policy.discover_packages_path=lerobot.policies.rdp_tokenizer

# For LDP (Stage 2):
--policy.discover_packages_path=lerobot.policies.rdp_latent_diffusion
```

## Stage 1: Train the Asymmetric Tokenizer

The AT learns to encode and decode action trajectories. It does not use image observations.

```bash
python -m lerobot.scripts.lerobot_train \
    --policy.discover_packages_path=lerobot.policies.rdp_tokenizer \
    --policy.type=rdp_tokenizer \
    --policy.push_to_hub=false \
    --dataset.repo_id=domrachev03/franka_timing_belt_haply_static_v3 \
    --policy.decoder_type=rnn \
    --policy.temporal_cond_keys='[observation.effort]' \
    --num_epochs=601 \
    --wandb.enable=true \
    --wandb.project=rdp_timing_belt
```

The RNN decoder is conditioned per-step on `observation.effort` (force/torque), making the decoded actions reactive to contact forces. The encoder compresses action chunks without any observation input.

### Key configuration options

| Parameter | Default | Description |
|---|---|---|
| `horizon` | 32 | Length of the action chunk to encode/decode |
| `encoder_type` | `"conv1d"` | Encoder architecture: `"mlp"` or `"conv1d"` |
| `n_latent_dims` | 4 | Dimensionality of the latent code |
| `encoder_hidden_dim` | 32 | Hidden dimension for encoder layers |
| `decoder_type` | `"mlp"` | Decoder architecture: `"mlp"` or `"rnn"` |
| `decoder_hidden_dim` | 32 | Hidden dimension for decoder layers |
| `use_vq` | `False` | Use Residual VQ instead of Gaussian VAE |
| `n_embed` | 32 | Codebook size (VQ) or quant channel dim (VAE) |
| `temporal_cond_keys` | `()` | Observation columns for RNN temporal conditioning (required when `decoder_type="rnn"`). Each entry must match a dataset column name (e.g. `observation.effort`). Multiple columns can be listed: `'[observation.effort,observation.state]'`. |
| `kl_multiplier` | 1e-6 | KL divergence loss weight (Gaussian VAE mode) |
| `vq_loss_multiplier` | 5.0 | VQ commitment loss weight (VQ mode) |
| `act_scale` | 1.0 | Divisor applied to normalised actions before encoding |

### Encoder types

- **`mlp`**: Flattens the action chunk and processes with an MLP. Latent is a single vector.
- **`conv1d`**: Uses 1-D convolutions with stride-2 downsampling. Latent is a sequence shorter than the input, which the LDP diffuses over as a 1-D trajectory.

### Decoder types

- **`mlp`**: Standard MLP decoder. No temporal conditioning.
- **`rnn`**: GRU-based decoder with temporal conditioning. Requires `temporal_cond_keys` to specify which observation features to use as step-by-step conditioning (e.g. force/torque readings, tactile data). This is the "asymmetric" part — the decoder has access to per-step observations that the encoder does not. When using RNN, the `observation_delta_indices` is automatically extended to cover the full horizon so that per-step conditioning data is available.

### Training metrics

The AT logs the following metrics to wandb. The table below shows the mapping to the original RDP codebase metric names:

| LeRobot metric | Original RDP metric | Formula | Description |
|---|---|---|---|
| `loss` | `train_loss` | `recon_l1 * encoder_loss_multiplier + regularization` | Total weighted loss (used for backprop) |
| `recon_l1` | `train_encoder_loss` | `\|state - dec_out\|_1` | L1 reconstruction loss |
| `recon_mse` | `train_vae_recon_loss` | `MSE(state, dec_out)` | MSE reconstruction (logged only, not backpropped) |
| `kl_loss` | `train_kl_loss` | `posterior.kl().mean()` | KL divergence (Gaussian VAE mode) |
| `vq_loss` | `train_vq_loss_state` | VQ commitment loss | VQ commitment loss (VQ mode) |

The total loss formula:
- **Gaussian VAE**: `loss = recon_l1 * encoder_loss_multiplier + kl_loss * kl_multiplier`
- **VQ-VAE**: `loss = recon_l1 * encoder_loss_multiplier + vq_loss * vq_loss_multiplier`

### Training output

The trained AT checkpoint will be saved to the standard LeRobot output directory:

```
outputs/train/<run_name>/checkpoints/<step>/pretrained_model/
```

Note this path — you will need it for Stage 2.

## Stage 2: Train the Latent Diffusion Policy

The LDP requires a pre-trained AT checkpoint from Stage 1.

```bash
python -m lerobot.scripts.lerobot_train \
    --policy.discover_packages_path=lerobot.policies.rdp_latent_diffusion \
    --policy.type=rdp_latent_diffusion \
    --policy.push_to_hub=false \
    --dataset.repo_id=domrachev03/franka_timing_belt_haply_static_v3 \
    --policy.pretrained_tokenizer_path=outputs/train/<stage1_run>/checkpoints/last/pretrained_model \
    --policy.at_decoder_type=rnn \
    --policy.at_temporal_cond_keys='[observation.effort]' \
    --policy.crop_shape='[480,640]' \
    --num_epochs=401 \
    --wandb.enable=true \
    --wandb.project=rdp_timing_belt
```

The UNet conditions on `observation.state` (tcp + gripper) via the same `robot_state_feature` mechanism as standard DP. The frozen AT decoder uses `at_temporal_cond_keys` for per-step force/torque conditioning during action decoding.

### Latent action normalization

Before training begins, the LDP automatically computes min/max statistics over the latent action space by encoding the full dataset through the frozen AT. Latent actions are then normalized to [-1, 1] (MIN_MAX) before being passed to the diffusion model, matching the original RDP implementation.

This happens automatically on fresh training. On resume, stats are loaded from the checkpoint. The computed statistics are logged at startup:

```
Latent stats computed over N batches: min=[...], max=[...]
```

### Key configuration options

| Parameter | Default | Description |
|---|---|---|
| `pretrained_tokenizer_path` | `None` | Path to trained AT checkpoint (required) |
| `use_latent_action_before_vq` | `False` | Diffuse on pre-quantisation latent (only relevant when AT uses VQ) |
| `vision_backbone` | `"resnet18"` | Vision encoder backbone |
| `crop_shape` | `(84, 84)` | Crop size for image augmentation. Set to camera resolution `(480, 640)` for full-size images. `None` to disable |
| `resize_shape` | `None` | Resize images to (H, W) via bilinear interpolation (applied after crop). Use instead of or in addition to cropping |
| `down_dims` | `(512, 1024, 2048)` | UNet channel dimensions |
| `noise_scheduler_type` | `"DDIM"` | Noise scheduler: `"DDIM"` or `"DDPM"` |
| `num_train_timesteps` | 100 | Number of diffusion timesteps |
| `num_inference_steps` | `None` | Inference steps (defaults to `num_train_timesteps`) |
| `prediction_type` | `"epsilon"` | What the UNet predicts: `"epsilon"` or `"sample"` |

### AT architecture parameters

The LDP config duplicates the AT architecture parameters with an `at_` prefix so that the AT can be reconstructed at load time without the original config:

| Parameter | Default | Mirrors AT |
|---|---|---|
| `at_encoder_type` | `"conv1d"` | `encoder_type` |
| `at_n_latent_dims` | 4 | `n_latent_dims` |
| `at_encoder_hidden_dim` | 32 | `encoder_hidden_dim` |
| `at_encoder_n_layers` | 1 | `encoder_n_layers` |
| `at_decoder_type` | `"rnn"` | `decoder_type` |
| `at_decoder_hidden_dim` | 32 | `decoder_hidden_dim` |
| `at_decoder_n_layers` | 1 | `decoder_n_layers` |
| `at_use_vq` | `False` | `use_vq` |
| `at_n_embed` | 32 | `n_embed` |
| `at_vqvae_groups` | 4 | `vqvae_groups` |
| `at_act_scale` | 1.0 | `act_scale` |
| `at_temporal_cond_keys` | `()` | `temporal_cond_keys` |

These **must match** the values used when training the AT in Stage 1.

### What happens during training

1. (Once, before training) Latent action min/max statistics are computed over the full dataset.
2. Actions from the dataset are encoded to latent space using the frozen AT encoder + quantization.
3. Latent actions are normalized to [-1, 1] using the precomputed statistics.
4. The UNet learns to denoise these normalized latent actions, conditioned on image and state observations.
5. The AT is never updated — only the vision encoder and UNet are trained.

### What happens during inference

1. The vision encoder + UNet produce a denoised latent action (in [-1, 1] normalized space).
2. The latent is unnormalized back to the original latent scale.
3. The frozen AT decoder converts the latent back to the original action space.
4. If the AT uses an RNN decoder, temporal conditioning from the current observations is used during decoding.

## Training Duration

The original RDP codebase uses epoch-based training with these defaults:

| Stage | Epochs | Batch size |
|---|---|---|
| AT (tokenizer) | 601 | 64 |
| Latent Diffusion | 401 | 64 |
| Standard Diffusion | 600 | 64 |

LeRobot uses step-based training. Convert epochs to steps:

```
steps = num_epochs * ceil(dataset_size / batch_size)
```

### Horizon and action steps

The original RDP maintains a constant ~1.33s wall-clock horizon. The `horizon` must be divisible by `2^len(down_dims)` (8 for the default 3-stage UNet).

| FPS | `horizon` | `n_obs_steps` | `n_action_steps` | Wall-clock |
|-----|-----------|---------------|-------------------|------------|
| 12  | 16        | 2             | 15                | 1.33s      |
| 10  | 16        | 2             | 15                | 1.60s      |
| 24  | 32        | 2             | 29*               | 1.33s      |

*AT/LDP at 24fps use `dataset_obs_temporal_downsample_ratio=2`, giving `n_action_steps = horizon - n_obs_steps * 2 + 1 = 29`.

### Pre-computed steps for `franka_timing_belt_haply_static_v2`

| Dataset | FPS | Frames | batch_size=64 | | |
|---|---|---|---|---|---|
| | | | **DP (600 ep)** | **AT (601 ep)** | **LDP (401 ep)** |
| `*_v2` | 30 | 32,811 | 307,800 | 308,313 | 205,713 |
| `*_v2_10hz` | 10 | 10,917 | 102,600 | 102,771 | 68,571 |

## Complete Examples

### Diffusion Policy on 10Hz dataset

Train DP on a 10Hz downsampled dataset with full-size 640x480 images:

```bash
python -m lerobot.scripts.lerobot_train \
    --policy.type=diffusion \
    --dataset.repo_id=domrachev03/franka_timing_belt_haply_static_v3_10hz \
    --policy.push_to_hub=false \
    --policy.down_dims='[512,1024,2048]' \
    --policy.crop_shape='[480,640]' \
    --policy.horizon=16 \
    --policy.n_obs_steps=2 \
    --policy.n_action_steps=15 \
    --batch_size=64 \
    --steps=102600 \
    --wandb.enable=true \
    --wandb.project=rdp_dp_timing_belt_static \
    --wandb.run_id=diffusion_10hz
```

When deploying this 10Hz policy on a 30Hz robot, use `--action_interpolation_steps=3` to interpolate actions (see [REPLAY.md](REPLAY.md) and [INFERENCE.md](INFERENCE.md)).

### Two-stage RDP on 30Hz dataset

RDP trains on the original 30Hz dataset (no downsampling needed):

```bash
# Stage 1: Train the Asymmetric Tokenizer
python -m lerobot.scripts.lerobot_train \
    --policy.discover_packages_path=lerobot.policies.rdp_tokenizer \
    --policy.type=rdp_tokenizer \
    --policy.push_to_hub=false \
    --dataset.repo_id=domrachev03/franka_timing_belt_haply_static_v3 \
    --policy.encoder_type=conv1d \
    --policy.decoder_type=rnn \
    --policy.temporal_cond_keys='[observation.effort]' \
    --policy.n_latent_dims=4 \
    --batch_size=64 \
    --steps=308313 \
    --wandb.enable=true \
    --wandb.project=rdp_timing_belt

# Stage 2: Train the Latent Diffusion Policy (point to Stage 1 output)
python -m lerobot.scripts.lerobot_train \
    --policy.discover_packages_path=lerobot.policies.rdp_latent_diffusion \
    --policy.type=rdp_latent_diffusion \
    --policy.push_to_hub=false \
    --dataset.repo_id=domrachev03/franka_timing_belt_haply_static_v3 \
    --policy.pretrained_tokenizer_path=outputs/train/<stage1_run>/checkpoints/last/pretrained_model \
    --policy.at_encoder_type=conv1d \
    --policy.at_decoder_type=rnn \
    --policy.at_temporal_cond_keys='[observation.effort]' \
    --policy.at_n_latent_dims=4 \
    --policy.vision_backbone=resnet18 \
    --policy.crop_shape='[480,640]' \
    --policy.num_train_timesteps=100 \
    --batch_size=64 \
    --steps=205713 \
    --wandb.enable=true \
    --wandb.project=rdp_timing_belt
```

## Checkpoints

### Directory Structure

```
output_dir/
  checkpoints/
    000000/
      pretrained_model/
        config.json             # Policy config
        model.safetensors       # Policy weights
        train_config.json       # Full training config
        preprocessor.json       # Preprocessor pipeline config
        postprocessor.json      # Postprocessor pipeline config
      training_state/
        optimizer_param_groups.json
        optimizer_state.safetensors
        scheduler_state.json
        rng_state.safetensors   # CPU/CUDA/NumPy/Python RNG states
        training_step.json
    020000/
    last -> 020000/             # Symlink to latest
```

### Resuming

Resume from the latest checkpoint:

```bash
python -m lerobot.scripts.lerobot_train \
    --config_path=outputs/train/.../checkpoints/last/pretrained_model/train_config.json \
    --resume=true
```

This restores the full training state: model weights, optimizer, scheduler, RNG states, and step counter.

## Distributed Training

LeRobot uses HuggingFace Accelerate for distributed training:

```bash
# Configure distributed setup
accelerate config

# Launch distributed training
accelerate launch -m lerobot.scripts.lerobot_train \
    --policy.type=diffusion \
    --dataset.repo_id=domrachev03/franka_timing_belt_haply_static_v3_10hz \
    --policy.crop_shape='[480,640]' \
    --batch_size=64
```

Mixed precision is handled automatically via `accelerator.autocast()`.

## Next Steps

- **Replay**: Verify the trained policy by replaying recorded trajectories — see [REPLAY.md](REPLAY.md)
- **Inference**: Deploy the trained policy on the real robot — see [INFERENCE.md](INFERENCE.md)

## Reference

- **Paper**: Xue et al., "Reactive Diffusion Policy", RSS 2025
- **Original implementation**: `reactive_diffusion_policy/` (Hydra-based training pipeline)
- **LeRobot port**: `lerobot/policies/rdp_tokenizer/` and `lerobot/policies/rdp_latent_diffusion/`
