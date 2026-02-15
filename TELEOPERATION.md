# LeRobot Teleoperation & Recording

This document describes how teleoperation and dataset recording work in LeRobot, and how to extend them with custom robots and teleoperators.

## Configuration System

LeRobot uses **draccus** (dataclass-based CLI parsing), not Hydra. All configuration is done via `@dataclass` classes with CLI overrides in the format `--field.subfield=value`.

Key concepts:
- **`RobotConfig`** and **`TeleoperatorConfig`** are abstract base classes using `draccus.ChoiceRegistry`
- Plugins register via `@RobotConfig.register_subclass("type_name")` decorators
- The `--robot.type=...` CLI argument selects which registered subclass to instantiate
- Third-party plugins are discovered via `--robot.discover_packages_path=my_package`

## Teleoperation

Pure teleoperation (no recording) uses `lerobot-teleoperate`:

```bash
lerobot-teleoperate \
    --robot.type=slim_crisp \
    --robot.server_ip=127.0.0.1 \
    --teleop.type=haply \
    --fps=60 \
    --display_data=true
```

### TeleoperateConfig

```python
@dataclass
class TeleoperateConfig:
    teleop: TeleoperatorConfig        # Teleoperator device config
    robot: RobotConfig                # Robot config
    fps: int = 60                     # Control loop frequency
    teleop_time_s: float | None = None  # Duration (None = infinite)
    display_data: bool = False        # Rerun visualization
```

### Teleoperation Loop

The `teleop_loop()` function runs at the target FPS:

1. `robot.get_observation()` — read current robot state
2. `teleop.get_action()` — read teleoperator input
3. Process teleop action through `teleop_action_processor` pipeline
4. Process robot action through `robot_action_processor` pipeline
5. `robot.send_action(action)` — send to robot
6. `precise_sleep()` to maintain target FPS

Custom processor pipelines can be injected for specific robot-teleoperator pairs (e.g., `HaplyToSlimCrispClutchProcessor` for Haply + SlimCrisp).

## Recording

Dataset recording uses `lerobot-record`:

```bash
lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyUSB0 \
    --robot.cameras='{"front": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' \
    --teleop.type=gamepad \
    --dataset.repo_id=username/my_dataset \
    --dataset.single_task="Pick the cube" \
    --dataset.num_episodes=50 \
    --dataset.fps=30
```

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

### Policy-Based Recording

Record autonomous policy rollouts by providing `--policy.path` instead of (or in addition to) `--teleop`:

```bash
lerobot-record \
    --robot.type=so100_follower \
    --policy.path=lerobot/act_aloha_cube \
    --dataset.repo_id=username/eval_dataset \
    --dataset.single_task="Policy rollout" \
    --dataset.num_episodes=10
```

## Robot Interface

All robots implement the `Robot` abstract base class from `lerobot/robots/robot.py`.

### Required Properties

```python
@property
def observation_features(self) -> dict:
    """Structure of get_observation() output. Must work without connection."""
    return {
        "shoulder_pan.pos": float,
        "camera_front": (480, 640, 3),  # (H, W, C) for images
    }

@property
def action_features(self) -> dict:
    """Structure of send_action() input."""
    return {"shoulder_pan.pos": float, "gripper.pos": float}

@property
def is_connected(self) -> bool: ...
@property
def is_calibrated(self) -> bool: ...
```

### Required Methods

| Method | Description |
|---|---|
| `connect(calibrate=True)` | Establish communication, auto-calibrate if needed |
| `calibrate()` | Run calibration procedure |
| `configure()` | Apply runtime settings (motor params, control modes) |
| `get_observation() -> dict` | Return current state matching `observation_features` |
| `send_action(action: dict) -> dict` | Send action, return actual action sent |
| `disconnect()` | Clean up |

### RobotConfig

```python
@RobotConfig.register_subclass("my_robot")
@dataclass
class MyRobotConfig(RobotConfig):
    # Base fields:
    id: str | None = None
    calibration_dir: Path | None = None
    # Custom fields:
    port: str = "/dev/ttyUSB0"
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
```

## Teleoperator Interface

All teleoperators implement the `Teleoperator` abstract base class from `lerobot/teleoperators/teleoperator.py`.

### Required Properties

```python
@property
def action_features(self) -> dict:
    """Structure of get_action() output."""
    return {
        "dtype": "float32",
        "shape": (8,),
        "names": {"x": 0, "y": 1, "z": 2, "qw": 3, "qx": 4, "qy": 5, "qz": 6, "gripper": 7}
    }

@property
def feedback_features(self) -> dict:
    """Structure of send_feedback() input."""
    return {}

@property
def is_connected(self) -> bool: ...
@property
def is_calibrated(self) -> bool: ...
```

### Required Methods

| Method | Description |
|---|---|
| `connect(calibrate=True)` | Connect to device |
| `calibrate()` | Run calibration |
| `configure()` | Apply runtime settings |
| `get_action() -> dict` | Return current teleop action |
| `send_feedback(feedback: dict)` | Send feedback (e.g., force) to device |
| `disconnect()` | Clean up |

### TeleoperatorConfig

```python
@TeleoperatorConfig.register_subclass("my_teleop")
@dataclass
class MyTeleopConfig(TeleoperatorConfig):
    id: str | None = None
    calibration_dir: Path | None = None
    # Custom fields:
    use_gripper: bool = True
```

## Camera System

Cameras are configured as part of `RobotConfig.cameras` (a `dict[str, CameraConfig]`).

### Available Camera Types

**OpenCV** (`type: opencv`):
```python
OpenCVCameraConfig(
    index_or_path: int | Path,    # Camera index or video path
    fps: int, width: int, height: int,
    color_mode: ColorMode = RGB,
    rotation: Cv2Rotation = NO_ROTATION,
)
```

**RealSense** (`type: intelrealsense`):
```python
RealSenseCameraConfig(
    serial_number_or_name: str,   # Camera serial number
    fps: int, width: int, height: int,
    color_mode: ColorMode = RGB,
    use_depth: bool = False,
)
```

### CLI Camera Configuration

Cameras are passed as JSON via CLI:

```bash
--robot.cameras='{
  "front": {"type": "intelrealsense", "serial_number_or_name": "838212074376", "width": 640, "height": 480, "fps": 30},
  "wrist": {"type": "intelrealsense", "serial_number_or_name": "130322271369", "width": 640, "height": 480, "fps": 30}
}'
```

## Processor Pipelines

LeRobot uses a processor pipeline system to transform data between teleoperator, robot, and policy formats.

Three default pipelines are created for every recording/teleoperation session:
- **`teleop_action_processor`** — transforms teleop output before sending to robot
- **`robot_action_processor`** — final processing before `robot.send_action()`
- **`robot_observation_processor`** — transforms observations before saving to dataset

By default, all three use `IdentityProcessorStep` (passthrough). Custom processors can be injected for specific hardware combinations (e.g., clutch logic for Haply).

## Adding a Custom Robot Plugin

1. Create the package structure:

```
my_robot/
  __init__.py       # Exports MyRobot, MyRobotConfig
  config.py         # Config dataclass with @register_subclass
  robot.py          # Robot implementation
```

2. Register the config:

```python
# config.py
@RobotConfig.register_subclass("my_robot")
@dataclass
class MyRobotConfig(RobotConfig):
    server_url: str = "http://localhost:8092"
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
```

3. Implement the robot:

```python
# robot.py
class MyRobot(Robot):
    config_class = MyRobotConfig
    name = "my_robot"

    def __init__(self, config: MyRobotConfig): ...
    # Implement all abstract methods
```

4. Export in `__init__.py`:

```python
from .config import MyRobotConfig
from .robot import MyRobot
```

5. Use via CLI:

```bash
lerobot-record \
    --robot.discover_packages_path=my_robot \
    --robot.type=my_robot \
    --robot.server_url=http://192.168.50.67:8092 \
    --teleop.type=gamepad \
    --dataset.repo_id=user/dataset
```

## Existing Robot/Teleoperator Implementations

### Robots

| Type | Module | Description |
|---|---|---|
| `so100_follower` | `robots/so100_follower/` | SO-100 follower arm (Feetech servos) |
| `slim_crisp` | `robots/slim_crisp/` | Remote robot via ZMQ (slim-crisp-zmq bridge) |
| `koch_follower` | `robots/koch_follower/` | Koch v1.1 follower arm |
| `lekiwi` | `robots/lekiwi/` | LeKiwi mobile robot |
| `reachy2` | `robots/reachy2/` | Pollen Reachy 2 |
| `mock_robot` | (built-in) | Mock robot for testing |

### Teleoperators

| Type | Module | Description |
|---|---|---|
| `gamepad` | `teleoperators/gamepad/` | Gamepad controller (pygame/hidapi) |
| `haply` | `teleoperators/haply/` | Haply Inverse3 + VerseGrip (raw output, processor-based clutch) |
| `keyboard` | `teleoperators/keyboard/` | Keyboard teleoperation |
| `phone` | `teleoperators/phone/` | Phone-based teleoperation |
| `so100_leader` | `teleoperators/so100_leader/` | SO-100 leader arm |
| `mock_teleop` | (built-in) | Mock teleoperator for testing |

## RDP Custom Plugins

The following plugins are added by the Reactive Diffusion Policy project for Franka robot teleoperation and recording.

### `crisp_fastapi` Robot

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

**Observation features:** `tcp.pos` (3), `tcp.quat` (4), `gripper.pos`, `wrench.force` (3), `wrench.torque` (3), `ft_sensor.force` (3), `ft_sensor.torque` (3), plus camera images.

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
