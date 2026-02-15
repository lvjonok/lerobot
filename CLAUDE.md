# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **fork of [HuggingFace LeRobot](https://github.com/huggingface/lerobot)** (v0.4.3) extended with custom hardware plugins and ported policy models for the Reactive Diffusion Policy (RDP) project (Xue et al., RSS 2025).

The fork adds:
- **Robot plugins** for crisp_py-based robots (Franka, Flexiv) via FastAPI/REST
- **Teleoperator plugins** for SpaceMouse, Haply Inverse3, and Meta Quest VR
- **Custom processors** for delta-to-absolute action conversion, clutch logic, etc.
- **RDP policy ports** — Asymmetric Tokenizer (rdp_tokenizer) and Latent Diffusion Policy (rdp_latent_diffusion)
- **Hardware presets** for quick recording configuration

## Environment & Build

This repo is installed as an editable package from the parent `reactive_diffusion_policy` project via **pixi**:

```bash
# From the parent reactive_diffusion_policy/ directory:
pixi install -e cuda     # GPU environment (training)
pixi install             # CPU-only environment
pixi shell -e cuda       # activate environment
```

Standalone installation (without pixi):

```bash
pip install -e ".[dev]"
```

Python version: 3.10+ (3.11 when used with parent pixi, pinned by ROS2 Jazzy).

## Operational Guides

- **[TELEOPERATION.md](TELEOPERATION.md)** — Teleoperation, recording, robot/teleoperator interfaces, custom plugin guide
- **[TRAINING.md](TRAINING.md)** — Training configuration, training loop, RDP two-stage training, checkpoints
- **[INFERENCE.md](INFERENCE.md)** — Simulation evaluation, real robot inference, async inference

## Running Tests

```bash
pytest tests/
pytest tests/test_some_module.py   # single test
```

## Architecture

### Configuration System

LeRobot uses **draccus** (dataclass-based CLI parsing), NOT Hydra. All configuration is done via `@dataclass` classes with CLI overrides: `--field.subfield=value`.

Key concepts:
- `RobotConfig` and `TeleoperatorConfig` use `draccus.ChoiceRegistry` for plugin discovery
- Plugins register via `@RobotConfig.register_subclass("type_name")` decorators
- `--robot.type=...` selects which registered subclass to instantiate
- Third-party plugins discovered via `--robot.discover_packages_path=my_package`

### Source Layout (`src/lerobot/`)

| Directory | Description |
|---|---|
| `policies/` | Policy implementations (diffusion, act, pi0, rdp_tokenizer, rdp_latent_diffusion, etc.) |
| `robots/` | Robot plugins (so100, koch, lekiwi, crisp_fastapi, slim_crisp, etc.) |
| `teleoperators/` | Teleoperator plugins (gamepad, keyboard, haply, spacemouse, meta_quest, etc.) |
| `processor/` | Data processor pipeline (normalization, delta actions, clutch, device transfer) |
| `cameras/` | Camera implementations (OpenCV, Intel RealSense) |
| `datasets/` | LeRobotDataset format, data loading, streaming |
| `scripts/` | CLI entry points (lerobot_train, lerobot_eval, lerobot_record, etc.) |
| `configs/` | Default configurations |
| `envs/` | Simulation environment wrappers (pusht, aloha, libero, metaworld) |
| `async_inference/` | Client-server async inference (gRPC policy server + robot client) |
| `model/` | Shared model components |
| `optim/` | Optimizer and LR scheduler configs |
| `utils/` | Shared utilities |

### CLI Entry Points

| Command | Script | Description |
|---|---|---|
| `lerobot-record` | `scripts/control_robot.py` | Record datasets via teleoperation or policy rollout |
| `lerobot-teleoperate` | `scripts/control_robot.py` | Pure teleoperation (no recording) |
| `lerobot-train` | `scripts/lerobot_train.py` | Train a policy |
| `lerobot-eval` | `scripts/lerobot_eval.py` | Evaluate a policy in simulation |
| `lerobot-info` | `scripts/lerobot_info.py` | Show system and package info |

## Custom Plugins (RDP-Specific)

### Robots

**`crisp_fastapi`** (`robots/crisp_fastapi/`) — HTTP REST client for crisp_py-based robot servers (Franka, Flexiv). Communicates with a standalone `franka_server_crisp.py` FastAPI server (lives in the parent `reactive_diffusion_policy/real_world/robot/` repo).

- Observations: `tcp.pos` (3), `tcp.quat` (4), `gripper.pos`, `wrench.force` (3), `wrench.torque` (3), `ft_sensor.force` (3), `ft_sensor.torque` (3), plus camera images
- Actions: `tcp.pos` (3), `tcp.quat` (4), `gripper.pos`

**`slim_crisp`** (`robots/slim_crisp/`) — ZMQ-based remote robot client (slim-crisp-zmq bridge).

### Teleoperators

**`spacemouse`** (`teleoperators/spacemouse/`) — 3Dconnexion SpaceMouse via libspnav. 6DOF delta pose, background reader at 100Hz, deadzone filtering, per-axis mode selection. Actions: `delta_pos` (3), `delta_rot` (3), `gripper.pos`.

**`haply`** (`teleoperators/haply/`) — Haply Inverse3 + VerseGrip with built-in clutch logic. Button 'b' = clutch (hold to control), Button 'a' = gripper toggle. Actions: `tcp.pos` (3), `tcp.quat` (4), `gripper.pos`.

**`meta_quest`** (`teleoperators/meta_quest/`) — Meta Quest VR via `meta-teleop` UDP multicast. Hand tracking, pinch-to-grip with hysteresis, optional engagement pinch for clutch. Actions: `tcp.pos` (3), `tcp.quat` (4), `gripper.pos`.

### Processors

- `crisp_fastapi_processors.py` — SpaceMouse delta-to-absolute, Haply clutch, Meta Quest delta-to-absolute processors for crisp_fastapi robots
- `haply_clutch_processor.py` — Haply clutch processor for slim_crisp robots

### Policies (RDP Ports)

**`rdp_tokenizer`** (`policies/rdp_tokenizer/`) — Asymmetric Tokenizer (AT). VAE/VQ-VAE that compresses action chunks into latent space. Encoder: MLP or Conv1D. Decoder: MLP or RNN with temporal conditioning on per-step observations (e.g. force/torque).

**`rdp_latent_diffusion`** (`policies/rdp_latent_diffusion/`) — Latent Diffusion Policy (LDP). Conditional 1-D UNet diffusion in the latent action space of a frozen AT. Uses vision encoder for image conditioning.

See [TRAINING.md](TRAINING.md) for the two-stage training guide.

### Hardware Presets

YAML presets in `presets/` for quick recording configuration:
- `franka_no_camera.yaml` — Franka robot without cameras
- `franka_two_cameras.yaml` — Franka with external + wrist RealSense cameras

## Robot Communication Pattern

The robot server (`franka_server_crisp.py`) runs as a standalone FastAPI process on the robot control PC with a control loop that:
1. Receives TCP pose targets and gripper commands via HTTP
2. Interpolates trajectories using `PoseTrajectoryInterpolator`
3. Sends commands to the robot via crisp_py ROS2 interface

The `crisp_fastapi` robot plugin in this repo is a thin HTTP client that talks to that server. The server itself lives in the parent `reactive_diffusion_policy/real_world/robot/` repository.

## Issue Labels

Every beads issue must carry two kinds of labels: **workstation** (`ws:`) and **area** (`area:`).

### Workstation labels (`ws:`)

| Workstation | Label | What runs here |
|---|---|---|
| **Personal PC** | `ws:any` | Development, code review, documentation |
| **Inference PC** | `ws:inference-pc` | Teleoperation, dataset collection, inference, robot experiments |
| **DL Server** | `ws:dl-server` | Model training (GPU) |

**Rules:**
- Development-only tasks (coding, refactoring, docs) get label `ws:any`
- Tasks requiring the robot or sensors get `ws:inference-pc`
- Training tasks get `ws:dl-server`
- A task can have multiple `ws:` labels if executable on several machines
- Every issue MUST have at least one `ws:` label

### Area labels (`area:`)

| Area | Label | Scope |
|---|---|---|
| **Teleoperation** | `area:teleoperation` | Robot plugins, teleop devices, dataset collection, processor pipelines |
| **Training** | `area:training` | Policy training, dataset format, training loop |
| **Inference** | `area:inference` | Policy deployment, evaluation, real-robot rollouts, async inference |

**Rules:**
- Every issue MUST have at least one `area:` label
- An issue can have multiple `area:` labels if it spans areas

## Coding Conventions
- Commit format: `<type>: <description>` where type is feat/fix/docs/refactor/test/chore
- Branches: `main`, `feature/*`, `fix/*`, `docs/*`
- All imports at the beginning of the file, grouped by standard library, third-party, and local
- Do not check if import exists; assume all dependencies are installed via pixi

## Git Rules
- All commits must be authored by `Domrachev Ivan <domrachev03@mail.ru>` — do NOT add Co-Authored-By lines or any other attribution
- NEVER push commits automatically — only commit when asked, never push unless explicitly told to
- Follow the commit format above: `<type>: <description>`
