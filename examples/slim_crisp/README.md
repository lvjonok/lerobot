# slim-crisp Robot Integration for LeRobot

This integration enables control of remote robots via the [slim-crisp-zmq](https://github.com/lvjonok/slim-crisp-zmq) bridge, allowing teleoperation with devices like Haply Inverse3 for recording demonstrations.

## Features

- **Cartesian Position Control**: Control robot end-effector position (x, y, z)
- **Remote Operation**: Communicate with robot over ZMQ protocol
- **Haply Teleoperation**: Compatible with Haply Inverse3 device for intuitive control
- **LeRobot Integration**: Full support for dataset recording and policy learning

## Prerequisites

1. **slim-crisp-zmq bridge** running on the robot machine
2. **Haply Inverse Service** (if using Haply teleop): `ws://localhost:10001`
3. **slim-crisp-zmq Python package** installed in LeRobot environment

## Installation

The slim-crisp-zmq package is included as an optional dependency:

```bash
# From lerobot directory
pip install -e ".[slim_crisp]"
```

Or install the slim-crisp-zmq package directly:

```bash
pip install -e /home/lev/github.com/lvjonok/slim-crisp-zmq
```

## Quick Start

### 1. Start the slim-crisp-zmq server

On the robot machine, start the ZMQ bridge server:

```bash
cd /path/to/slim-crisp-zmq
python server/launch.py
```

By default, this will listen on:
- State publisher: `tcp://*:5556`
- Command reply: `tcp://*:5557`

### 2. Start Haply Inverse Service (if using Haply)

Make sure the Haply Inverse Service is running and accessible at `ws://localhost:10001`.

### 3. Record demonstrations

#### Using the CLI:

```bash
lerobot-record \
    --robot.type=slim_crisp \
    --robot.server_ip=192.168.1.100 \
    --robot.id=my_robot \
    --teleop.type=haply \
    --teleop.use_gripper=true \
    --dataset.repo_id=<username>/<dataset_name> \
    --dataset.num_episodes=10 \
    --dataset.single_task="Pick and place object" \
    --display_data=true
```

#### Using the example script:

```bash
cd lerobot
python examples/slim_crisp/record_haply.py
```

Edit the script to configure:
- `server_ip`: IP address of the robot server
- `HF_REPO_ID`: Your HuggingFace dataset repository
- `NUM_EPISODES`, `FPS`, `EPISODE_TIME_SEC`: Recording parameters

## Configuration

### Robot Configuration

The `SlimCrispConfig` class supports the following parameters:

```python
from lerobot.robots.slim_crisp import SlimCrispConfig, SlimCrispRobot

config = SlimCrispConfig(
    # Connection settings
    server_ip="192.168.1.100",      # Robot server IP address
    state_pub_port=5556,            # ZMQ state publisher port
    cmd_rep_port=5557,              # ZMQ command reply port
    
    # Client settings
    command_timeout=5.0,            # Command timeout in seconds
    max_state_delay=1.0,            # Max acceptable state staleness
    
    # Controller settings
    default_controller="cartesian_impedance_controller",
    
    # Gripper (TODO: implementation pending)
    use_gripper=True,
    
    # Robot identification
    id="my_robot",
)

robot = SlimCrispRobot(config)
```

### YAML Configuration

You can also create a YAML config file:

```yaml
# config_slim_crisp.yaml
type: slim_crisp
id: slim_crisp_robot
server_ip: "192.168.1.100"
state_pub_port: 5556
cmd_rep_port: 5557
default_controller: "cartesian_impedance_controller"
use_gripper: true
cameras: {}
```

## Teleoperation with Haply

The Haply Inverse3 provides intuitive 3D position control:

### Controls:
- **Button 'b'**: Toggle intervention (start/stop robot control)
- **Button 'a'**: Toggle gripper open/close
- **Button 'c'**: Mark episode as successful
- **Keyboard 'R'**: Re-record current episode

### Control Mode:
When you press button 'b' to start controlling:
1. The Haply and robot initial positions are captured
2. As you move the Haply device, the robot tracks: `robot_target = robot_initial + (haply_current - haply_initial)`
3. This provides intuitive absolute position control with the delta computed from your motion

## Observation and Action Space

### Observations
```python
{
    "ee_pos_x": float,  # End-effector X position (meters)
    "ee_pos_y": float,  # End-effector Y position (meters)
    "ee_pos_z": float,  # End-effector Z position (meters)
    "gripper.pos": float,  # Gripper state 0-1 (TODO: implementation)
}
```

### Actions
```python
{
    "ee_pos_x": float,  # Target X position (meters)
    "ee_pos_y": float,  # Target Y position (meters)
    "ee_pos_z": float,  # Target Z position (meters)
    "gripper.pos": float,  # Target gripper state (TODO: implementation)
}
```

## Coordinate Frames

Both the slim-crisp robot and Haply device are assumed to use:
- **Units**: Meters
- **Coordinate system**: Compatible frames (adjust if needed)

If coordinate transformation is needed, update the mappings in the robot or teleoperator implementation.

## TODO

- [ ] Implement gripper observation and action
- [ ] Add camera support (local or remote streaming)
- [ ] Add orientation control (quaternion or rotation matrix)
- [ ] Add force/torque feedback to Haply
- [ ] Verify coordinate frame alignment between devices

## Troubleshooting

### Robot not connecting
- Check that slim-crisp-zmq server is running
- Verify IP address and ports are correct
- Check network connectivity: `ping <robot_ip>`

### Stale state warnings
- Network latency too high
- Server not publishing state updates
- Increase `max_state_delay` if needed

### Haply not connecting
- Ensure Haply Inverse Service is running: `ws://localhost:10001`
- Install dependencies: `pip install websockets orjson`
- Check device is connected and powered on

## Architecture

```
┌─────────────────┐         ZMQ          ┌──────────────────┐
│   LeRobot       │◄────────────────────►│ slim-crisp-zmq   │
│   (Record/      │   State (PUB/SUB)    │   Bridge Server  │
│    Train)       │   Commands (REQ/REP) │                  │
└─────────────────┘                      └──────────────────┘
        ▲                                          │
        │                                          │ ROS2
        │                                          ▼
        │                                 ┌──────────────────┐
        │                                 │  Physical Robot  │
        │                                 │  (crisp_py)      │
        │                                 └──────────────────┘
        │
┌───────▼─────────┐
│  Haply Inverse3 │
│  (Teleop)       │
└─────────────────┘
```

## References

- [slim-crisp-zmq](https://github.com/lvjonok/slim-crisp-zmq): ZMQ bridge for remote robot control
- [crisp_py](https://github.com/lvjonok/crisp_py): Python interface for robot control
- [LeRobot](https://github.com/huggingface/lerobot): Framework for robot learning
- [Haply Robotics](https://www.haply.co/): Inverse3 haptic device
