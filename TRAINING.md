# LeRobot Training

This document describes how training works in LeRobot, including configuration, dataset format, training loop, and checkpointing.

## Quick Start

```bash
# Train diffusion policy on pusht dataset
python -m lerobot.scripts.lerobot_train \
    --policy.path=lerobot/diffusion_pusht \
    --dataset.repo_id=lerobot/pusht \
    --batch_size=64 \
    --steps=100000

# Resume training from checkpoint
python -m lerobot.scripts.lerobot_train \
    --config_path=outputs/train/.../checkpoints/last/pretrained_model/train_config.json \
    --resume=true
```

## Configuration

LeRobot uses **draccus** (dataclass-based CLI parsing). All training configuration is defined in `TrainPipelineConfig`.

### TrainPipelineConfig

```python
@dataclass
class TrainPipelineConfig(HubMixin):
    # Core
    dataset: DatasetConfig                    # Dataset configuration
    policy: PreTrainedConfig | None = None    # Policy configuration
    env: EnvConfig | None = None              # Optional sim environment (for evaluation)

    # Training
    output_dir: Path | None = None            # Auto-generated if None
    job_name: str | None = None               # Run identifier
    resume: bool = False                      # Resume from checkpoint
    seed: int | None = 1000

    # Hyperparameters
    batch_size: int = 8
    num_workers: int = 4
    steps: int = 100_000

    # Frequencies
    eval_freq: int = 20_000                   # Evaluation frequency
    log_freq: int = 200                       # Logging frequency
    save_freq: int = 20_000                   # Checkpoint frequency

    # Optimizer / Scheduler
    use_policy_training_preset: bool = True    # Use policy's default optimizer/scheduler
    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None

    # Integrations
    eval: EvalConfig = EvalConfig()
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
    streaming: bool = False                   # Stream from Hub
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

## CLI Usage

### Override any config field via CLI

```bash
python -m lerobot.scripts.lerobot_train \
    --policy.path=lerobot/diffusion_pusht \
    --dataset.repo_id=lerobot/pusht \
    --policy.n_obs_steps=3 \
    --policy.horizon=32 \
    --optimizer.lr=5e-4 \
    --scheduler.num_warmup_steps=1000 \
    --wandb.enable=true \
    --wandb.project=my_project
```

### Load pretrained policy and fine-tune

```bash
python -m lerobot.scripts.lerobot_train \
    --policy.path=lerobot/diffusion_pusht \
    --dataset.repo_id=my_user/my_dataset \
    --steps=50000
```

### Create policy from scratch

```bash
python -m lerobot.scripts.lerobot_train \
    --policy.type=diffusion \
    --policy.n_obs_steps=2 \
    --policy.horizon=16 \
    --dataset.repo_id=lerobot/pusht
```

## Training Loop

The training script (`lerobot_train.py`) follows this flow:

### Initialization

1. Create `Accelerator` (handles distributed training, mixed precision)
2. Validate config (auto-set output_dir, load optimizer presets, etc.)
3. Initialize WandB logger (main process only)
4. Set random seed
5. Load dataset via `LeRobotDataset` (main process downloads first to avoid race conditions)
6. Create sim environment for evaluation (optional)
7. Instantiate policy via `make_policy(cfg.policy, ds_meta=dataset.meta)`
8. Create preprocessor/postprocessor pipelines
9. Build optimizer and LR scheduler (from policy presets or explicit config)
10. Create DataLoader with `EpisodeAwareSampler`

### Main Loop

```python
for step in range(steps):
    batch = next(dataloader)
    batch = preprocessor(batch)

    loss, info = update_policy(policy, batch, optimizer, scheduler)

    if step % log_freq == 0:
        log_metrics(loss, grad_norm, lr)

    if step % save_freq == 0:
        save_checkpoint(step)

    if step % eval_freq == 0:
        eval_policy(env, policy)
```

### update_policy()

1. **Forward pass** with accelerator autocast (mixed precision)
2. **Backward pass** via `accelerator.backward(loss)`
3. **Gradient clipping** via `accelerator.clip_grad_norm_()`
4. **Optimizer step**
5. **LR scheduler step**
6. **Policy update hook** (e.g., EMA update for diffusion policies)

## Policy System

### Available Policies

| Type | Description |
|---|---|
| `diffusion` | Diffusion Policy (1D conditional UNet) |
| `act` | Action Chunking Transformers |
| `vqbet` | VQ-BeT (Residual VQ-VAE + GPT) |
| `tdmpc` | Temporal Difference MPC |
| `pi0` | Physical Intelligence Pi0 (VLA) |
| `pi05` | Physical Intelligence Pi0.5 |
| `sac` | Soft Actor-Critic (RL) |
| `smolvla` | SmolVLA (lightweight VLA) |

### Policy Config (PreTrainedConfig)

All policies extend `PreTrainedConfig`:

```python
@dataclass
class PreTrainedConfig(HubMixin, draccus.ChoiceRegistry, abc.ABC):
    n_obs_steps: int = 1                      # Observation history length
    input_features: dict[str, PolicyFeature]  # Auto-populated from dataset
    output_features: dict[str, PolicyFeature]
    device: str | None = None
    use_amp: bool = False                     # Automatic Mixed Precision

    # HuggingFace Hub
    push_to_hub: bool = True
    repo_id: str | None = None
    pretrained_path: Path | None = None
```

Each policy defines its own default optimizer and scheduler presets:

```python
@PreTrainedConfig.register_subclass("diffusion")
@dataclass
class DiffusionConfig(PreTrainedConfig):
    horizon: int = 16
    n_action_steps: int = 8
    vision_backbone: str = "resnet18"
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    # ...

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(lr=1e-4, betas=(0.95, 0.999))

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(name="cosine", num_warmup_steps=500)
```

### Feature Types

```python
class FeatureType(str, Enum):
    STATE = "STATE"       # Proprioceptive state
    VISUAL = "VISUAL"     # Camera images
    ACTION = "ACTION"     # Robot actions
    ENV = "ENV"           # Environment state
    REWARD = "REWARD"     # Task rewards
    LANGUAGE = "LANGUAGE" # Language instructions

@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple[int, ...]    # Without batch/time dimensions
```

## Optimizer System

### Available Optimizers

| Type | Description |
|---|---|
| `adam` | Adam with configurable betas, eps, weight_decay |
| `adamw` | AdamW (decoupled weight decay) |
| `sgd` | SGD with momentum, nesterov support |

All optimizers include a `grad_clip_norm` field (default: 10.0).

### Available LR Schedulers

| Type | Description |
|---|---|
| `diffuser` | HuggingFace Diffusers scheduler (cosine, linear, etc.) |
| `vqbet` | Custom two-stage scheduler for VQ-BeT |
| `cosine_decay_with_warmup` | Linear warmup + cosine decay |

## Dataset System

### LeRobotDataset

Datasets are stored as chunked parquet files with optional video:

```
dataset_root/
  data/
    chunk-000/
      file-000.parquet          # Observation/action columns
  meta/
    info.json                   # fps, features, shapes
    stats.json                  # Normalization statistics
    tasks.parquet               # Task descriptions
    episodes/
      chunk-000/
        file-000.parquet        # Episode index
  videos/
    observation.images.front/
      chunk-000/
        file-000.mp4            # Video-encoded frames
```

Key properties:
- `fps` — frames per second
- `features` — dict of feature definitions (shape, dtype)
- `meta.stats` — normalization statistics (mean, std, min, max per feature)
- `camera_keys` — list of visual modality keys

### Dataset Factory

`make_dataset(cfg)` handles:
1. Creating `LeRobotDatasetMetadata` from `repo_id`
2. Resolving `delta_timestamps` from policy config and dataset FPS
3. Creating image transforms if enabled
4. Instantiating `LeRobotDataset` (or `StreamingLeRobotDataset`)
5. Optionally replacing camera stats with ImageNet statistics

## Preprocessor/Postprocessor Pipelines

Data flows through composable processor pipelines:

**Training**:
```
Dataset Batch → Preprocessor → Policy Forward → Loss
```

**Inference**:
```
Observation → Preprocessor → Policy → Postprocessor → Action
```

Common processor steps:
- `DeviceProcessorStep` — move tensors to device
- `NormalizerProcessorStep` — normalize with dataset statistics
- `UnnormalizerProcessorStep` — unnormalize actions
- `RenameObservationsProcessorStep` — rename observation keys
- `AddBatchDimensionProcessorStep` — add batch dim for single samples

Each policy defines its own processor factory (e.g., `make_diffusion_pre_post_processors()`).

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

## HuggingFace Hub Integration

### Push trained model to Hub

Set in config or CLI:

```bash
--policy.push_to_hub=true \
--policy.repo_id=my_user/my_policy
```

After training completes, the model is uploaded with:
- `config.json` + `model.safetensors`
- Auto-generated model card (README.md)
- Tags: `["robotics", "lerobot", "<policy_type>"]`

### Load pretrained model

```python
from lerobot.policies.pretrained import PreTrainedPolicy
policy = PreTrainedPolicy.from_pretrained("lerobot/diffusion_pusht")
```

Or via CLI: `--policy.path=lerobot/diffusion_pusht`

## Distributed Training

LeRobot uses HuggingFace Accelerate for distributed training:

```bash
# Configure distributed setup
accelerate config

# Launch distributed training
accelerate launch -m lerobot.scripts.lerobot_train \
    --policy.path=lerobot/diffusion_pusht \
    --dataset.repo_id=lerobot/pusht \
    --batch_size=64
```

Mixed precision is handled automatically via `accelerator.autocast()`.

---

# Training Reactive Diffusion Policy (RDP) with LeRobot

This guide covers training the two-stage Reactive Diffusion Policy (Xue et al., RSS 2025) using the LeRobot framework.

## Overview

RDP consists of two stages trained sequentially:

1. **Asymmetric Tokenizer (AT)** (`rdp_tokenizer`) — a VAE/VQ-VAE that compresses action chunks into a low-dimensional latent space. The encoder is simple (MLP or Conv1D); the decoder is optionally an RNN conditioned on temporal observations (e.g. tactile/force data).
2. **Latent Diffusion Policy (LDP)** (`rdp_latent_diffusion`) — a conditional 1-D UNet diffusion model that operates in the latent action space of the frozen AT. It uses the same vision encoder architecture as the standard Diffusion Policy.

## Stage 1: Train the Asymmetric Tokenizer

The AT learns to encode and decode action trajectories. It does not use image observations.

```bash
lerobot-train \
  --policy.type=rdp_tokenizer \
  --dataset.repo_id=<user>/<dataset> \
  --policy.horizon=32 \
  --policy.encoder_type=conv1d \
  --policy.decoder_type=rnn \
  --policy.temporal_cond_keys='("observation.wrench",)'
```

### Key configuration options

| Parameter | Default | Description |
|---|---|---|
| `horizon` | 32 | Length of the action chunk to encode/decode |
| `encoder_type` | `"mlp"` | Encoder architecture: `"mlp"` or `"conv1d"` |
| `n_latent_dims` | 4 | Dimensionality of the latent code |
| `encoder_hidden_dim` | 32 | Hidden dimension for encoder layers |
| `decoder_type` | `"mlp"` | Decoder architecture: `"mlp"` or `"rnn"` |
| `decoder_hidden_dim` | 32 | Hidden dimension for decoder layers |
| `use_vq` | `False` | Use Residual VQ instead of Gaussian VAE |
| `n_embed` | 32 | Codebook size (VQ) or quant channel dim (VAE) |
| `temporal_cond_keys` | `()` | Observation keys for RNN temporal conditioning (required when `decoder_type="rnn"`) |
| `kl_multiplier` | 1e-6 | KL divergence loss weight (Gaussian VAE mode) |
| `vq_loss_multiplier` | 5.0 | VQ commitment loss weight (VQ mode) |
| `act_scale` | 1.0 | Divisor applied to normalised actions before encoding |

### Encoder types

- **`mlp`**: Flattens the action chunk and processes with an MLP. Latent is a single vector.
- **`conv1d`**: Uses 1-D convolutions with stride-2 downsampling. Latent is a sequence shorter than the input, which the LDP diffuses over as a 1-D trajectory.

### Decoder types

- **`mlp`**: Standard MLP decoder. No temporal conditioning.
- **`rnn`**: GRU-based decoder with temporal conditioning. Requires `temporal_cond_keys` to specify which observation features to use as step-by-step conditioning (e.g. force/torque readings, tactile data). This is the "asymmetric" part — the decoder has access to per-step observations that the encoder does not.

### Training output

The trained AT checkpoint will be saved to the standard LeRobot output directory. Note the path — you will need it for Stage 2.

## Stage 2: Train the Latent Diffusion Policy

The LDP requires a pre-trained AT checkpoint from Stage 1.

```bash
lerobot-train \
  --policy.type=rdp_latent_diffusion \
  --dataset.repo_id=<user>/<dataset> \
  --policy.pretrained_tokenizer_path=<path_to_at_checkpoint> \
  --policy.horizon=32 \
  --policy.at_encoder_type=conv1d \
  --policy.at_decoder_type=rnn \
  --policy.at_temporal_cond_keys='("observation.wrench",)'
```

### Key configuration options

| Parameter | Default | Description |
|---|---|---|
| `pretrained_tokenizer_path` | `None` | Path to trained AT checkpoint (required) |
| `use_latent_action_before_vq` | `False` | Diffuse on pre-quantisation latent (only relevant when AT uses VQ) |
| `vision_backbone` | `"resnet18"` | Vision encoder backbone |
| `crop_shape` | `(84, 84)` | Random crop size for image augmentation |
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

1. Actions from the dataset are encoded to latent space using the frozen AT encoder + quantization.
2. The UNet learns to denoise these latent actions, conditioned on image and state observations.
3. The AT is never updated — only the vision encoder and UNet are trained.

### What happens during inference

1. The vision encoder + UNet produce a denoised latent action.
2. The frozen AT decoder converts the latent back to the original action space.
3. If the AT uses an RNN decoder, temporal conditioning from the current observations is used during decoding.

## Complete two-stage example

```bash
# Stage 1: Train the Asymmetric Tokenizer
lerobot-train \
  --policy.type=rdp_tokenizer \
  --dataset.repo_id=domrachev03/franka_peg_insertion \
  --policy.horizon=32 \
  --policy.encoder_type=conv1d \
  --policy.decoder_type=rnn \
  --policy.temporal_cond_keys='("observation.wrench",)' \
  --policy.n_latent_dims=4

# Stage 2: Train the Latent Diffusion Policy (point to Stage 1 output)
lerobot-train \
  --policy.type=rdp_latent_diffusion \
  --dataset.repo_id=domrachev03/franka_peg_insertion \
  --policy.pretrained_tokenizer_path=outputs/<stage1_run>/pretrained_model \
  --policy.at_encoder_type=conv1d \
  --policy.at_decoder_type=rnn \
  --policy.at_temporal_cond_keys='("observation.wrench",)' \
  --policy.at_n_latent_dims=4 \
  --policy.vision_backbone=resnet18 \
  --policy.num_train_timesteps=100
```

## Reference

- **Paper**: Xue et al., "Reactive Diffusion Policy", RSS 2025
- **Original implementation**: `reactive_diffusion_policy/` (Hydra-based training pipeline)
- **LeRobot port**: `lerobot/policies/rdp_tokenizer/` and `lerobot/policies/rdp_latent_diffusion/`
