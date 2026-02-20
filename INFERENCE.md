# Inference & Evaluation

This document describes how to deploy trained policies on the real robot for evaluation.

## Action Interpolation

When a policy is trained at lower FPS than the robot's control rate (e.g. 10Hz policy on a 30Hz robot), **action interpolation** smooths the trajectory by dividing each twist action into sub-steps:

```bash
--action_interpolation_steps=3   # 10Hz policy x 3 = 30Hz robot commands
```

How it works:
1. The policy outputs a twist action (linear_vel + angular_vel) at 10Hz
2. `divide_twist()` divides velocity components by N (keeping absolute targets like `gripper.pos` unchanged)
3. N sub-steps are executed at `1 / (fps x N)` intervals, each fetching a fresh robot observation
4. The `robot_action_processor` converts each divided twist to an absolute pose using the live observation

This is available in both `lerobot-record --policy` and `lerobot-replay`. See [REPLAY.md](REPLAY.md) for replay-specific details.

**When to use:**
- 10Hz Diffusion Policy on 30Hz robot: `--action_interpolation_steps=3`
- 30Hz RDP on 30Hz robot: not needed (steps = 0)

## Policy-Based Recording (lerobot-record)

Record policy rollouts as a dataset using `lerobot-record` with `--policy.path`. This reuses the recording infrastructure but uses a policy instead of a teleoperator.

```bash
lerobot-record \
    --robot.type=crisp_ws \
    --robot.ws_url=ws://localhost:8092/ws \
    --robot.max_gripper_width=0.078 \
    --robot.gripper_velocity=0.15 \
    --robot.gripper_force=20.0 \
    --robot.cameras='{"external_camera": {"type": "intelrealsense", "serial_number_or_name": "838212074376", "width": 640, "height": 480, "fps": 30}, "left_wrist_camera": {"type": "intelrealsense", "serial_number_or_name": "130322271369", "width": 640, "height": 480, "fps": 30}}' \
    --policy.path=outputs/train/.../checkpoints/last/pretrained_model \
    --dataset.repo_id=domrachev03/eval_dataset \
    --dataset.single_task="Policy evaluation" \
    --dataset.num_episodes=10 \
    --dataset.fps=30 \
    --action_interpolation_steps=3
```

The recording loop calls `policy.select_action()` instead of `teleop.get_action()` each step. The resulting dataset can be analyzed offline for success rate, trajectory quality, etc.

When using `--action_interpolation_steps`, the policy runs at its native FPS (e.g. 10Hz) while the robot receives interpolated commands at the higher rate. Dataset frames are saved at the policy FPS.

**Hybrid mode**: Provide both `--teleop` and `--policy` to use the policy for recording and the teleoperator for resetting the environment between episodes.

## Async Inference (Client-Server)

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
    --robot.type=crisp_ws \
    --robot.ws_url=ws://localhost:8092/ws \
    --server_address=127.0.0.1:8080 \
    --policy_type=diffusion \
    --pretrained_name_or_path=outputs/train/.../checkpoints/last/pretrained_model \
    --actions_per_chunk=8 \
    --fps=30 \
    --task="Pick and place"
```

### Action Chunking & Aggregation

The client maintains a queue of future actions. When a new action chunk arrives from the server and overlaps with queued actions, they are aggregated:

| Strategy | Formula | Use Case |
|---|---|---|
| `weighted_average` | 0.3 * old + 0.7 * new | Default, smooth transitions |
| `latest_only` | new | Responsive, no blending |
| `average` | 0.5 * old + 0.5 * new | Equal weighting |
| `conservative` | 0.7 * old + 0.3 * new | Stable, slow adaptation |

### Client-Server Protocol (gRPC)

1. **`Ready()`** — client handshake, resets server state
2. **`SendPolicyInstructions()`** — client sends policy type, path, device; server loads policy
3. **`SendObservations()`** — client streams observations to server
4. **`GetActions()`** — server runs inference, returns timestamped action chunk

## Policy Loading

### From Local Checkpoint

```python
from lerobot.policies.pretrained import PreTrainedPolicy
policy = PreTrainedPolicy.from_pretrained(
    "outputs/train/.../checkpoints/last/pretrained_model",
    device="cuda"
)
```

Or via CLI: `--policy.path=outputs/train/.../checkpoints/last/pretrained_model`

### From HuggingFace Hub

```python
policy = PreTrainedPolicy.from_pretrained("domrachev03/my_policy", device="cuda")
```

Or via CLI: `--policy.path=domrachev03/my_policy`

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
Observation -> Preprocessor -> Policy -> Postprocessor -> Action
```

Processors are loaded from the pretrained model directory:
- `preprocessor.json` / `postprocessor.json` — pipeline configuration
- Step-specific state files (e.g., normalization statistics)

Key inference-specific processing:
- **`NormalizerProcessorStep`** — normalize observations using dataset statistics
- **`UnnormalizerProcessorStep`** — unnormalize predicted actions
- **`DeviceProcessorStep`** — move tensors to/from GPU

## See Also

- [REPLAY.md](REPLAY.md) — Replaying recorded dataset trajectories on the real robot
- [TRAINING.md](TRAINING.md) — Training policies on real-world datasets
- [TELEOPERATION.md](TELEOPERATION.md) — Recording datasets with teleoperation
