# Replaying Recorded Trajectories

This document describes how to replay actions from a recorded dataset on a real robot using `lerobot-replay`.

## Quick Start

```bash
lerobot-replay \
    --robot.type=crisp_ws \
    --robot.ws_url=ws://localhost:8092/ws \
    --dataset.repo_id=domrachev03/franka_timing_belt_haply_static_v3 \
    --dataset.episode=0 \
    --dataset.fps=30
```

## ReplayConfig

```python
@dataclass
class ReplayConfig:
    robot: RobotConfig                  # Robot configuration
    dataset: DatasetReplayConfig        # Dataset + episode selection
    play_sounds: bool = True            # Vocal synthesis for events
    action_interpolation_steps: int = 0 # Sub-steps per action (0 = disabled)
```

### DatasetReplayConfig

```python
@dataclass
class DatasetReplayConfig:
    repo_id: str                        # HuggingFace dataset ID
    episode: int                        # Episode index to replay
    root: str | Path | None = None      # Local dataset root
    fps: int = 30                       # Replay frame rate
```

## Replay Flow

1. Load the dataset and filter to the specified episode
2. Create `robot_action_processor` for the robot type (e.g. `TwistToAbsolutePoseProcessor` for crisp robots)
3. Connect to the robot
4. For each frame in the episode:
   - Unpack the action vector into named fields (`linear_vel`, `angular_vel`, `gripper.pos`, etc.)
   - Get current robot observation
   - Process action through `robot_action_processor` (converts twist to absolute pose)
   - Send action to robot
   - Sleep to maintain target FPS

## Action Interpolation

When replaying a dataset recorded (or downsampled) at a lower FPS than the robot's control rate, action interpolation produces smoother trajectories.

### When to use

| Dataset FPS | Robot FPS | Interpolation steps | Result |
|---|---|---|---|
| 30 Hz | 30 Hz | 0 (disabled) | Direct replay |
| 10 Hz | 30 Hz | 3 | 10 × 3 = 30 Hz robot commands |
| 10 Hz | 30 Hz | 0 | Jerky motion (one command every 100ms) |

### How it works

```
Dataset frame (10Hz twist action)
    │
    ├── divide_twist(action, n=3)
    │   ├── linear_vel / 3
    │   ├── angular_vel / 3
    │   └── gripper.pos unchanged (absolute target)
    │
    ├── Sub-step 1 (at t + 0ms)
    │   ├── robot.get_observation()     ← fresh pose
    │   ├── robot_action_processor()    ← twist → absolute using live pose
    │   └── robot.send_action()
    │
    ├── Sub-step 2 (at t + 33ms)
    │   ├── robot.get_observation()
    │   ├── robot_action_processor()
    │   └── robot.send_action()
    │
    └── Sub-step 3 (at t + 66ms)
        ├── robot.get_observation()
        ├── robot_action_processor()
        └── robot.send_action()
```

Key details:
- **Velocity keys** (`linear_vel`, `angular_vel`) are divided by N so each sub-step covers 1/N of the total displacement
- **Absolute keys** (`gripper.pos`) are passed through unchanged — the robot moves to the same target each sub-step
- Each sub-step fetches a **fresh robot observation**, so the twist-to-absolute conversion uses the robot's actual current pose
- Sub-step timing: `1 / (fps × N)` seconds between sub-steps

### Usage

```bash
# Replay 10Hz dataset with 3× interpolation (effective 30Hz robot commands)
lerobot-replay \
    --robot.type=crisp_ws \
    --robot.ws_url=ws://localhost:8092/ws \
    --dataset.repo_id=domrachev03/franka_timing_belt_haply_static_v3_10hz \
    --dataset.episode=0 \
    --dataset.fps=10 \
    --action_interpolation_steps=3
```

## Franka Examples

### Replay 30Hz dataset directly

```bash
lerobot-replay \
    --robot.type=crisp_ws \
    --robot.ws_url=ws://localhost:8092/ws \
    --dataset.repo_id=domrachev03/franka_timing_belt_haply_static_v3 \
    --dataset.episode=0 \
    --dataset.fps=30
```

### Replay 10Hz dataset with interpolation

```bash
lerobot-replay \
    --robot.type=crisp_ws \
    --robot.ws_url=ws://localhost:8092/ws \
    --dataset.repo_id=domrachev03/franka_timing_belt_haply_static_v3_10hz \
    --dataset.episode=0 \
    --dataset.fps=10 \
    --action_interpolation_steps=3
```

## Action Format

The replay script unpacks the flat action vector from the dataset into named fields using the dataset's `features` metadata:

| Field | Dims | Description |
|---|---|---|
| `linear_vel` | 3 | Position displacement (m) |
| `angular_vel` | 3 | Rotation displacement (rotation vector, rad) |
| `gripper.pos` | 1 | Gripper width (m), absolute target |

For `crisp_ws` / `crisp_fastapi` robots, the `robot_action_processor` (`TwistToAbsolutePoseProcessor`) converts this twist to an absolute TCP target:
- `target_pos = current_pos + linear_vel`
- `target_rot = Rotation.from_rotvec(angular_vel) * current_rot`

## Comparison: Replay vs Policy Evaluation

| Aspect | `lerobot-replay` | `lerobot-record --policy` |
|---|---|---|
| Action source | Dataset file | Policy inference |
| Observations | Fetched but not saved | Fetched and saved to new dataset |
| Cameras | Not needed | Needed (for policy input) |
| Interpolation | `--action_interpolation_steps` | `--action_interpolation_steps` |
| Use case | Verify dataset quality, test robot setup | Evaluate trained policy |

## See Also

- [INFERENCE.md](INFERENCE.md) — Policy evaluation and deployment
- [TRAINING.md](TRAINING.md) — Training and dataset FPS guidelines
- [TELEOPERATION.md](TELEOPERATION.md) — Recording datasets
