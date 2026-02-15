"""Configuration for CrispWSRobot — WebSocket-based crisp_py robot client."""

from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("crisp_ws")
@dataclass
class CrispWSConfig(RobotConfig):
    """Configuration for a robot controlled via crisp_py WebSocket server.

    The robot communicates with a running crisp_py-based WS server instance
    over a persistent WebSocket connection, eliminating per-message HTTP
    overhead for lower-latency control.
    """

    ws_url: str = "ws://localhost:8092/ws"

    # Camera configurations
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Control parameters
    max_gripper_width: float = 0.08
    gripper_velocity: float = 0.1
    gripper_force: float = 40.0
