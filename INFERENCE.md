# LeRobot Inference & Evaluation

This document describes how to run trained policies for evaluation in simulation and on real robots.

## Simulation Evaluation

### Quick Start

```bash
python -m lerobot.scripts.lerobot_eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.n_episodes=50 \
    --eval.batch_size=10 \
    --policy.device=cuda
```

### EvalPipelineConfig

```python
@dataclass
class EvalPipelineConfig:
    env: EnvConfig                            # Environment configuration
    eval: EvalConfig = EvalConfig()           # Evaluation settings
    policy: PreTrainedConfig | None = None    # Policy (loaded via --policy.path)
    output_dir: Path | None = None            # Auto-generated if None
    job_name: str | None = None
    seed: int | None = 1000
    rename_map: dict[str, str] = {}           # Observation key renaming
```

### EvalConfig

```python
@dataclass
class EvalConfig:
    n_episodes: int = 50                      # Total episodes to evaluate
    batch_size: int = 50                      # Parallel environments
    use_async_envs: bool = False              # AsyncVectorEnv (multiprocessing)
```

### Evaluation Flow

1. Create vectorized environments from config
2. Load policy via `PreTrainedPolicy.from_pretrained()`
3. Create preprocessor/postprocessor pipelines
4. Run batched rollouts via `eval_policy()`
5. Save results to `eval_info.json`

### rollout()

The core evaluation function runs a single batched rollout:

```python
def rollout(env, policy, preprocessor, postprocessor, seeds, ...):
    policy.reset()
    obs, info = env.reset()

    while not all_done:
        action = policy.select_action(preprocessor(obs))
        action = postprocessor(action)
        obs, reward, terminated, truncated, info = env.step(action)

    return {"action": ..., "reward": ..., "success": ..., "done": ...}
```

Returns tensors of shape `(batch, timesteps, ...)` for all tracked quantities.

### eval_policy()

Runs multiple batched rollouts, collecting metrics and optional videos:

```python
def eval_policy(env, policy, n_episodes, max_episodes_rendered, videos_dir, ...):
    # Run n_episodes // batch_size batches
    # Render videos for first max_episodes_rendered episodes
    # Aggregate metrics across all episodes
    return {
        "per_episode": [...],  # {episode_ix, sum_reward, max_reward, success, seed}
        "aggregated": {
            "avg_sum_reward": ...,
            "avg_max_reward": ...,
            "pc_success": ...,
            "eval_s": ...,
        },
        "video_paths": [...],
    }
```

### Available Simulation Environments

| Type | Description | FPS | Max Steps |
|---|---|---|---|
| `pusht` | 2D pushing task | 10 | 300 |
| `aloha` | Bimanual manipulation | 50 | 400 |
| `libero` | Multi-task benchmark (10 suites) | 30 | varies |
| `metaworld` | Multi-task benchmark (50 tasks) | 80 | 400 |

### Multi-Task Evaluation

For benchmarks like LIBERO with multiple suites and tasks:

```bash
python -m lerobot.scripts.lerobot_eval \
    --policy.path=my_user/my_policy \
    --env.type=libero \
    --env.task=libero_10 \
    --eval.n_episodes=50
```

`eval_policy_all()` evaluates across all tasks and aggregates results per suite and overall.

## Real Robot Evaluation

LeRobot provides three approaches for real robot evaluation:

### 1. Policy-Based Recording (lerobot-record)

Record policy rollouts as a dataset using `lerobot-record` with `--policy.path`. This reuses the recording infrastructure but uses a policy instead of a teleoperator.

```bash
lerobot-record \
    --robot.type=crisp_fastapi \
    --robot.server_url=http://192.168.50.67:8092 \
    --robot.cameras='{"external": {"type": "intelrealsense", "serial_number_or_name": "838212074376", "width": 640, "height": 480, "fps": 30}}' \
    --policy.path=my_user/my_policy \
    --dataset.repo_id=my_user/eval_dataset \
    --dataset.single_task="Policy evaluation" \
    --dataset.num_episodes=10 \
    --dataset.fps=30
```

The recording loop calls `policy.select_action()` instead of `teleop.get_action()` each step. The resulting dataset can be analyzed offline for success rate, trajectory quality, etc.

**Hybrid mode**: Provide both `--teleop` and `--policy` to use the policy for recording and the teleoperator for resetting the environment between episodes.

### 2. Async Inference (Client-Server)

For real-time action chunking with latency management, use the async inference system:

**Start the policy server** (can run on a GPU machine):

```bash
python -m lerobot.async_inference.policy_server \
    --host=127.0.0.1 \
    --port=8080 \
    --fps=30
```

**Start the robot client** (runs on the inference PC with robot access):

```bash
python -m lerobot.async_inference.robot_client \
    --robot.type=crisp_fastapi \
    --robot.server_url=http://192.168.50.67:8092 \
    --server_address=127.0.0.1:8080 \
    --policy_type=diffusion \
    --pretrained_name_or_path=my_user/my_policy \
    --actions_per_chunk=8 \
    --fps=30 \
    --task="Pick and place"
```

#### Action Chunking & Aggregation

The client maintains a queue of future actions. When a new action chunk arrives from the server and overlaps with queued actions, they are aggregated:

| Strategy | Formula | Use Case |
|---|---|---|
| `weighted_average` | 0.3 * old + 0.7 * new | Default, smooth transitions |
| `latest_only` | new | Responsive, no blending |
| `average` | 0.5 * old + 0.5 * new | Equal weighting |
| `conservative` | 0.7 * old + 0.3 * new | Stable, slow adaptation |

#### Client-Server Protocol (gRPC)

1. **`Ready()`** — client handshake, resets server state
2. **`SendPolicyInstructions()`** — client sends policy type, path, device; server loads policy
3. **`SendObservations()`** — client streams observations to server
4. **`GetActions()`** — server runs inference, returns timestamped action chunk

### 3. Custom Gym Environment

Wrap the robot as a Gym environment and use `lerobot-eval` directly:

```python
import gymnasium as gym

class FrankaRealEnv(gym.Env):
    def __init__(self, robot):
        self.robot = robot

    def reset(self, seed=None):
        self.robot.go_home()
        obs = self.robot.get_observation()
        return obs, {}

    def step(self, action):
        self.robot.send_action(action)
        obs = self.robot.get_observation()
        reward = self.compute_reward(obs)
        terminated = self.check_success(obs)
        return obs, reward, terminated, False, {}
```

## Policy Loading

### From HuggingFace Hub

```python
from lerobot.policies.pretrained import PreTrainedPolicy
policy = PreTrainedPolicy.from_pretrained("lerobot/diffusion_pusht", device="cuda")
```

Or via CLI: `--policy.path=lerobot/diffusion_pusht`

### From Local Checkpoint

```python
policy = PreTrainedPolicy.from_pretrained(
    "outputs/train/.../checkpoints/last/pretrained_model",
    device="cuda"
)
```

Or via CLI: `--policy.path=outputs/train/.../checkpoints/last/pretrained_model`

### Loading Process

1. Download/locate `config.json`
2. Parse config using draccus to determine policy subclass (e.g., `DiffusionConfig`)
3. Instantiate policy class
4. Download/locate `model.safetensors`
5. Load weights via `safetensors.torch.load_model()`
6. Move to device, set to eval mode

## Preprocessor/Postprocessor in Inference

The same processor pipelines used in training are loaded for inference:

```
Observation → Preprocessor → Policy → Postprocessor → Action
```

Processors are loaded from the pretrained model directory:
- `preprocessor.json` / `postprocessor.json` — pipeline configuration
- Step-specific state files (e.g., normalization statistics)

Key inference-specific processing:
- **`NormalizerProcessorStep`** — normalize observations using dataset statistics
- **`UnnormalizerProcessorStep`** — unnormalize predicted actions
- **`DeviceProcessorStep`** — move tensors to/from GPU

## Metrics & Output

### Per-Episode Metrics

| Metric | Description |
|---|---|
| `sum_reward` | Sum of rewards over the episode |
| `max_reward` | Maximum single-step reward |
| `success` | Whether the episode succeeded |
| `seed` | Random seed used |

### Aggregated Metrics

| Metric | Description |
|---|---|
| `avg_sum_reward` | Mean sum reward across all episodes |
| `avg_max_reward` | Mean max reward |
| `pc_success` | Success percentage |
| `eval_s` | Total evaluation wall time |
| `eval_ep_s` | Average time per episode |

### Output Files

```
output_dir/
  eval_info.json              # All metrics (per-episode + aggregated)
  videos/
    eval_episode_0.mp4        # Rendered evaluation videos
    eval_episode_1.mp4
```

## Sim vs Real Comparison

| Aspect | Simulation | Real Robot |
|---|---|---|
| Environment | Gym VectorEnv | Physical robot + sensors |
| Parallelism | Batched (batch_size envs) | Typically single |
| Termination | `terminated \| truncated` | Manual or success detector |
| Observation | Simulated sensors | Real cameras + proprioception |
| Action exec | Instant (`env.step()`) | Async (HTTP/gRPC to robot server) |
| Video | `env.render()` | Camera capture |
| Evaluation | `lerobot-eval` | `lerobot-record --policy` or async inference |
