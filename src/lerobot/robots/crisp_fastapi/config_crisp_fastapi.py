"""Configuration for CrispFastAPIRobot — generic crisp_py REST API robot client."""

from dataclasses import dataclass, field
from typing import Any

from lerobot.cameras.configs import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("crisp_fastapi")
@dataclass
class CrispFastAPIConfig(RobotConfig):
    """Configuration for a robot controlled via crisp_py FastAPI REST server.

    The robot communicates with a running crisp_py-based server instance
    over HTTP REST API. Works with any robot that exposes the crisp_py
    FastAPI interface (Franka, Flexiv, etc.).
    """

    # Robot server connection
    server_url: str = "http://localhost:8092"
    timeout: float = 5.0

    # Camera configurations (keys are camera names, e.g., "front", "wrist")
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Control parameters
    max_gripper_width: float = 0.08  # meters
    gripper_velocity: float = 0.1  # m/s
    gripper_force: float = 40.0  # N
