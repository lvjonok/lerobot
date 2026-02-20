# LeRobot Teleoperation & Recording

This document describes how teleoperation and dataset recording work in LeRobot, and how to extend them with custom robots and teleoperators.

## Configuration System

LeRobot uses **draccus** (dataclass-based CLI parsing), not Hydra. All configuration is done via `@dataclass` classes with CLI overrides in the format `--field.subfield=value`.

Key concepts:
- **`RobotConfig`** and **`TeleoperatorConfig`** are abstract base classes using `draccus.ChoiceRegistry`
- Plugins register via `@RobotConfig.register_subclass("type_name")` decorators
- The `--robot.type=...` CLI argument selects which registered subclass to instantiate
- Third-party plugins are discovered via `--robot.discover_packages_path=my_package`

## Franka Teleoperation & Recording (Quick Start)

The `crisp_fastapi` robot connects to a running `franka_server_crisp.py` via HTTP REST. Hardware presets in `presets/` bundle robot, teleoperator, and dataset defaults for common setups.

### Prerequisites

1. Start the Franka crisp server on the robot control PC (see parent repo's `TELEOPERATION.md`).
2. Ensure the robot server is reachable at `http://192.168.50.67:8092` (or adjust the URL).
3. For camera presets, connect the RealSense cameras and verify serial numbers match.

### Teleoperation (no recording)

**SpaceMouse** (no cameras):
```bash
lerobot-teleoperate \
    --robot.type=crisp_fastapi \
    --robot.server_url=http://192.168.50.67:8092 \
    --robot.max_gripper_width=0.078 \
    --robot.gripper_velocity=0.15 \
    --robot.gripper_force=20.0 \
    --teleop.type=spacemouse \
    --teleop.translation_scale=0.0000125 \
    --teleop.rotation_scale=0.000025 \
    --teleop.teleop_mode=left_arm_3D_translation_Z_rotation \
    --teleop.translation_deadzone=120.0 \
    --teleop.rotation_deadzone=120.0 \
    --fps=30
```

**Haply Inverse3** (no cameras):
```bash
lerobot-teleoperate \
    --robot.type=crisp_fastapi \
    --robot.server_url=http://192.168.50.67:8092 \
    --robot.max_gripper_width=0.078 \
    --robot.gripper_velocity=0.15 \
    --robot.gripper_force=20.0 \
    --teleop.type=haply \
    --teleop.ws_uri=ws://localhost:10001 \
    --teleop.teleop_mode=left_arm_3D_translation_Z_rotation \
    --fps=30
```

**Meta Quest** (with two cameras):
```bash
lerobot-teleoperate \
    --robot.type=crisp_fastapi \
    --robot.server_url=http://192.168.50.67:8092 \
    --robot.max_gripper_width=0.078 \
    --robot.gripper_velocity=0.15 \
    --robot.gripper_force=20.0 \
    --robot.cameras='{"external_camera": {"type": "intelrealsense", "serial_number_or_name": "838212074376", "fps": 30, "width": 640, "height": 480}, "left_wrist_camera": {"type": "intelrealsense", "serial_number_or_name": "130322271369", "fps": 30, "width": 640, "height": 480}}' \
    --teleop.type=meta_quest \
    --teleop.multicast_group=224.1.1.1 \
    --teleop.control_hand=right \
    --teleop.teleop_mode=left_arm_3D_translation_Z_rotation \
    --fps=30 \
    --display_data=true
```

### Recording

**SpaceMouse, no cameras** (matches `presets/franka_no_camera.yaml`):
```bash
lerobot-record \
    --robot.type=crisp_fastapi \
    --robot.server_url=http://192.168.50.67:8092 \
    --robot.max_gripper_width=0.078 \
    --robot.gripper_velocity=0.15 \
    --robot.gripper_force=20.0 \
    --teleop.type=spacemouse \
    --teleop.translation_scale=0.0000125 \
    --teleop.rotation_scale=0.000025 \
    --teleop.teleop_mode=left_arm_3D_translation_Z_rotation \
    --teleop.translation_deadzone=120.0 \
    --teleop.rotation_deadzone=120.0 \
    --dataset.repo_id=local/franka_spacemouse_no_cam \
    --dataset.single_task="Pick and place" \
    --dataset.fps=30 \
    --dataset.episode_time_s=60 \
    --dataset.num_episodes=10 \
    --dataset.video=false \
    --dataset.push_to_hub=false
```

**SpaceMouse, two cameras** (matches `presets/franka_two_cameras.yaml`):
```bash
lerobot-record \
    --robot.type=crisp_fastapi \
    --robot.server_url=http://192.168.50.67:8092 \
    --robot.max_gripper_width=0.078 \
    --robot.gripper_velocity=0.15 \
    --robot.gripper_force=20.0 \
    --robot.cameras='{"external_camera": {"type": "intelrealsense", "serial_number_or_name": "838212074376", "fps": 30, "width": 640, "height": 480}, "left_wrist_camera": {"type": "intelrealsense", "serial_number_or_name": "130322271369", "fps": 30, "width": 640, "height": 480}}' \
    --teleop.type=spacemouse \
    --teleop.translation_scale=0.0000125 \
    --teleop.rotation_scale=0.000025 \
    --teleop.teleop_mode=left_arm_3D_translation_Z_rotation \
    --teleop.translation_deadzone=120.0 \
    --teleop.rotation_deadzone=120.0 \
    --dataset.repo_id=local/franka_spacemouse_two_cams \
    --dataset.single_task="Pick and place" \
    --dataset.fps=30 \
    --dataset.episode_time_s=60 \
    --dataset.num_episodes=10 \
    --dataset.video=true \
    --dataset.push_to_hub=false
```

**Haply, two cameras**:
```bash
lerobot-record \
    --robot.type=crisp_fastapi \
    --robot.server_url=http://192.168.50.67:8092 \
    --robot.max_gripper_width=0.078 \
    --robot.gripper_velocity=0.15 \
    --robot.gripper_force=20.0 \
    --robot.cameras='{"external_camera": {"type": "intelrealsense", "serial_number_or_name": "838212074376", "fps": 30, "width": 640, "height": 480}, "left_wrist_camera": {"type": "intelrealsense", "serial_number_or_name": "130322271369", "fps": 30, "width": 640, "height": 480}}' \
    --teleop.type=haply \
    --teleop.ws_uri=ws://localhost:10001 \
    --teleop.teleop_mode=left_arm_3D_translation_Z_rotation \
    --dataset.repo_id=local/franka_haply_two_cams \
    --dataset.single_task="Pick and place" \
    --dataset.fps=30 \
    --dataset.episode_time_s=60 \
    --dataset.num_episodes=10 \
    --dataset.video=true \
    --dataset.push_to_hub=false
```

**Meta Quest, two cameras**:
```bash
lerobot-record \
    --robot.type=crisp_fastapi \
    --robot.server_url=http://192.168.50.67:8092 \
    --robot.max_gripper_width=0.078 \
    --robot.gripper_velocity=0.15 \
    --robot.gripper_force=20.0 \
    --robot.cameras='{"external_camera": {"type": "intelrealsense", "serial_number_or_name": "838212074376", "fps": 30, "width": 640, "height": 480}, "left_wrist_camera": {"type": "intelrealsense", "serial_number_or_name": "130322271369", "fps": 30, "width": 640, "height": 480}}' \
    --teleop.type=meta_quest \
    --teleop.multicast_group=224.1.1.1 \
    --teleop.control_hand=right \
    --teleop.teleop_mode=left_arm_3D_translation_Z_rotation \
    --dataset.repo_id=local/franka_quest_two_cams \
    --dataset.single_task="Pick and place" \
    --dataset.fps=30 \
    --dataset.episode_time_s=60 \
    --dataset.num_episodes=10 \
    --dataset.video=true \
    --dataset.push_to_hub=false
```

### Processor Pipelines

The processor factory (`processor/factory.py`) auto-selects the correct processor pipeline for each `crisp_fastapi` + teleoperator combination. Each pipeline chains three steps:

1. **Teleop-specific processor** — converts raw teleop output to absolute TCP targets
2. **`GripperInterpolationProcessor`** — smoothly ramps the gripper width toward the target at 0.1 m/s instead of jumping instantly between open/closed
3. **`AbsoluteToTwistProcessor`** — converts absolute TCP target to twist (linear_vel + angular_vel) for the robot server

| Teleoperator | Teleop Processor | Description |
|---|---|---|
| `spacemouse` | `SpaceMouseDeltaToAbsoluteProcessor` | Accumulates per-frame delta_pos/delta_rot into absolute TCP targets |
| `haply` | `HaplyToCrispClutchProcessor` | Converts raw Haply positions with clutch logic (button 'b') into absolute TCP targets |
| `meta_quest` | `DeltaPoseToAbsoluteProcessor` | Applies delta position/quaternion from initial hand pose to initial robot pose |

No manual processor configuration is needed — the correct pipeline is injected automatically based on `--robot.type` and `--teleop.type`.

### Hardware Presets

YAML files in `presets/` document tested hardware configurations. They are **reference configurations** — the values should be passed as CLI arguments as shown above.

| Preset | Description | Teleoperators |
|---|---|---|
| `franka_no_camera.yaml` | Franka arm, no cameras, `video: false` | spacemouse, haply |
| `franka_two_cameras.yaml` | Franka arm + external D435i + wrist D405, `video: true` | spacemouse, haply, meta_quest |

## Recording

### RecordConfig

```python
@dataclass
class RecordConfig:
    robot: RobotConfig                    # Robot config
    dataset: DatasetRecordConfig          # Dataset config
    teleop: TeleoperatorConfig | None     # Teleoperator (for manual recording)
    policy: PreTrainedConfig | None       # Policy (for autonomous recording)
    display_data: bool = False            # Rerun visualization
    play_sounds: bool = True              # Audio cues for episode events
    resume: bool = False                  # Resume existing dataset
```

Either `teleop` or `policy` (or both) must be provided.

### DatasetRecordConfig

```python
@dataclass
class DatasetRecordConfig:
    repo_id: str                          # HuggingFace repo ID (user/dataset_name)
    single_task: str                      # Task description string
    root: str | Path | None = None        # Local dataset root
    fps: int = 30                         # Recording frame rate
    episode_time_s: int | float = 60      # Episode duration in seconds
    reset_time_s: int | float = 60        # Reset time between episodes
    num_episodes: int = 50                # Total episodes to record
    video: bool = True                    # Encode frames as video
    push_to_hub: bool = True              # Upload to HuggingFace Hub
    private: bool = False                 # Private Hub repository
    num_image_writer_processes: int = 0   # Image writer subprocesses
    num_image_writer_threads_per_camera: int = 4
    rename_map: dict[str, str] = ...      # Observation key renaming
```

### Recording Flow

1. Create robot and teleoperator from configs
2. Create default processor pipelines (teleop action, robot action, robot observation)
3. Aggregate dataset features from robot and teleoperator
4. Create or resume `LeRobotDataset`
5. Connect devices, initialize keyboard listener
6. **Episode loop** (for each episode):
   - **Record phase**: run `record_loop()` for `episode_time_s` with dataset writing
   - **Reset phase**: run `record_loop()` for `reset_time_s` without recording
   - Handle rerecord events (discard and retry)
   - `dataset.save_episode()`
7. Disconnect devices, push to Hub if enabled

### Keyboard Controls

- `Enter` — end current episode early
- `q` — stop recording session
- `r` — re-record current episode (discard buffer)

## Plugins

### `crisp_fastapi` / `crisp_ws` Robot

**Module:** `robots/crisp_fastapi/`

Thin HTTP REST client that communicates with a standalone `franka_server_crisp.py` (FastAPI/ROS2 server) running on the robot control PC. The server is **not** part of lerobot — it lives in `reactive_diffusion_policy/real_world/robot/franka_server_crisp.py`.

**Config fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `server_url` | `str` | `http://localhost:8092` | URL of the Franka crisp server |
| `timeout` | `float` | `5.0` | HTTP request timeout (seconds) |
| `cameras` | `dict[str, CameraConfig]` | `{}` | Camera configurations |
| `max_gripper_width` | `float` | `0.078` | Max gripper opening (meters) |
| `gripper_velocity` | `float` | `0.15` | Gripper movement speed (m/s) |
| `gripper_force` | `float` | `20.0` | Gripper grasp force (N) |

**Observation features:** `tcp.pos` (3), `tcp.quat` (4), `gripper.pos`, `wrench.force` (3), `wrench.torque` (3), `ft_sensor.force` (3), `ft_sensor.torque` (3), `joint.pos` (7), `joint.vel` (7), plus camera images.

**Action features:** `tcp.pos` (3), `tcp.quat` (4), `gripper.pos`.

**Usage:**

```bash
lerobot-record \
    --robot.type=crisp_fastapi \
    --robot.server_url=http://192.168.50.67:8092 \
    --robot.cameras='{"wrist": {"type": "intelrealsense", "serial_number_or_name": "130322271369", "width": 640, "height": 480, "fps": 30}}' \
    --teleop.type=spacemouse \
    --dataset.repo_id=user/franka_dataset \
    --dataset.single_task="Pick up object"
```

### `spacemouse` Teleoperator

**Module:** `teleoperators/spacemouse/`

3Dconnexion SpaceMouse teleoperator using libspnav. Provides 6DOF delta pose via a background reader thread at 100Hz. Supports deadzone filtering and per-axis mode selection.

**Config fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `translation_scale` | `float` | `1.0` | Translation sensitivity multiplier |
| `rotation_scale` | `float` | `1.0` | Rotation sensitivity multiplier |
| `teleop_mode` | `str` | `left_arm_6DOF` | Axis filter mode |
| `translation_deadzone` | `float` | `0.01` | Deadzone for translation axes |
| `rotation_deadzone` | `float` | `0.01` | Deadzone for rotation axes |
| `use_delta_control` | `bool` | `True` | Delta vs absolute mode |
| `max_gripper_width` | `float` | `0.08` | Max gripper opening (meters) |

**Teleop modes:** `left_arm_6DOF`, `left_arm_3D_translation`, `left_arm_3D_translation_Y_rotation`, `left_arm_3D_translation_Z_rotation`.

**Button mapping:** Button 0 toggles gripper, Button 1 resets reference position.

**Action features:** `delta_pos` (3), `delta_rot` (3), `gripper.pos`.

### `haply` Teleoperator

**Module:** `teleoperators/haply/`

Haply Inverse3 + VerseGrip with **built-in clutch logic** (unlike the base `haply` type which relies on external processors). Button 'b' engages the clutch — hold to control, release to reposition. Deltas accumulate across clutch sessions.

**Config fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `ws_uri` | `str` | `ws://localhost:10001` | Haply Inverse Service WebSocket URI |
| `translation_scale` | `float` | `1.0` | Translation sensitivity |
| `rotation_scale` | `float` | `1.0` | Rotation sensitivity |
| `teleop_mode` | `str` | `left_arm_6DOF` | Axis filter mode |
| `max_gripper_width` | `float` | `0.08` | Max gripper opening (meters) |
| `enable_feedback` | `bool` | `False` | Enable haptic force feedback |

**Button mapping:** Button 'a' toggles gripper, Button 'b' clutch.

**Action features:** `tcp.pos` (3), `tcp.quat` (4), `gripper.pos`.

### `meta_quest` Teleoperator

**Module:** `teleoperators/meta_quest/`

Meta Quest VR headset teleoperator using the `meta-teleop` UDP multicast client. Hand tracking provides wrist/palm pose; pinch gesture controls gripper with hysteresis. Supports engagement pinch (off-hand) for clutch control.

**Config fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `multicast_group` | `str` | `239.0.0.1` | UDP multicast group |
| `control_hand` | `str` | `right` | Hand for pose control |
| `translation_scale` | `float` | `1.0` | Translation sensitivity |
| `rotation_scale` | `float` | `1.0` | Rotation sensitivity |
| `teleop_mode` | `str` | `left_arm_6DOF` | Axis filter mode |
| `pinch_close_threshold` | `float` | `0.025` | Pinch distance to close gripper (m) |
| `pinch_open_threshold` | `float` | `0.045` | Pinch distance to open gripper (m) |
| `max_gripper_width` | `float` | `0.08` | Max gripper opening (meters) |
| `require_engagement_pinch` | `bool` | `False` | Require off-hand pinch to engage |
| `use_palm_joint` | `bool` | `False` | Use palm joint instead of wrist |
| `channel_timeout` | `float` | `10.0` | Timeout waiting for Quest channels |

**Action features:** `tcp.pos` (3), `tcp.quat` (4), `gripper.pos`.
