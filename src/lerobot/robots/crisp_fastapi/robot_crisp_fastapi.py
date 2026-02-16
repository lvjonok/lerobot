"""Generic crisp_py FastAPI robot implementation for LeRobot.

This robot communicates with a crisp_py-based FastAPI server via HTTP requests,
allowing integration with lerobot's teleoperation and recording infrastructure.
Works with any robot that exposes the crisp_py REST interface.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from typing import Any
import time

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_crisp_fastapi import CrispFastAPIConfig

logger = logging.getLogger(__name__)

# Suppress verbose httpx HTTP request logging (INFO level logs every request)
logging.getLogger("httpx").setLevel(logging.WARNING)


class CrispFastAPIRobot(Robot):
    """LeRobot-compatible robot using crisp_py FastAPI server.

    Communicates with a crisp_py-based server via HTTP REST API.
    Supports Cartesian pose control, gripper control, force-torque sensor
    readings, and camera integration via lerobot's camera system.
    """

    config_class = CrispFastAPIConfig
    name = "crisp_fastapi"

    def __init__(self, config: CrispFastAPIConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._client = None
        self._pool = None

        # Initialize cameras
        self.cameras = {}
        if config.cameras:
            self.cameras = make_cameras_from_configs(config.cameras)

    def __str__(self) -> str:
        return f"CrispFastAPIRobot({self.config.id})"

    @cached_property
    def observation_features(self) -> dict:
        """Dictionary describing observation structure.

        Observations are split into named groups so that each group becomes a
        separate dataset column (``observation.state``, ``observation.effort``,
        etc.).  This allows policies to selectively include/exclude groups via
        ``input_features``.
        """
        features = {
            "state": {
                "tcp.pos": (3,),
                "tcp.quat": (4,),
                "gripper.pos": float,
            },
            "effort": {
                "ft_sensor.force": (3,),
                "ft_sensor.torque": (3,),
            },
            "joints": {
                "joint.pos": (7,),
            },
            "joint_vel": {
                "joint.vel": (7,),
            },
        }

        for cam_key, cam in self.cameras.items():
            if hasattr(cam, "height") and hasattr(cam, "width"):
                features[cam_key] = (cam.height, cam.width, 3)
            else:
                features[cam_key] = (480, 640, 3)

        return features

    @cached_property
    def action_features(self) -> dict:
        """Dictionary describing action structure (6DOF twist + gripper)."""
        return {
            "linear_vel": (3,),
            "angular_vel": (3,),
            "gripper.pos": float,
        }

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        """Establish connection to the robot server."""
        if self._connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "CrispFastAPIRobot requires httpx. Install with: pip install httpx"
            ) from e

        logger.info(f"Connecting to crisp server at {self.config.server_url}")

        self._client = httpx.Client(
            base_url=self.config.server_url,
            timeout=self.config.timeout,
        )
        self._pool = ThreadPoolExecutor(max_workers=2)

        try:
            response = self._client.get("/get_current_robot_states")
            response.raise_for_status()
            logger.info("Successfully connected to crisp server")
        except Exception as e:
            self._client.close()
            self._client = None
            raise ConnectionError(f"Failed to connect to crisp server: {e}")

        for cam_key, cam in self.cameras.items():
            try:
                cam.connect()
                logger.info(f"Connected camera: {cam_key}")
            except Exception as e:
                logger.warning(f"Failed to connect camera {cam_key}: {e}")

        self._connected = True
        logger.info(f"{self} connected successfully")

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        """Retrieve current observation from the robot."""
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        t0 = time.perf_counter()
        response = self._client.get("/get_current_robot_states")
        response.raise_for_status()
        state = response.json()
        # logger.info(f"Time: {(time.perf_counter() - t0)*1e3 :0.1f} ms")

        tcp = state.get("leftRobotTCP", [0, 0, 0, 1, 0, 0, 0])
        tcp_pos = np.array(tcp[:3], dtype=np.float32)
        tcp_quat = np.array(tcp[3:], dtype=np.float32)

        gripper_state = state.get("leftGripperState", [0, 0])
        gripper_pos = float(gripper_state[0])

        ft_wrench = state.get("leftFTSensorWrench", state.get("FTSensorWrench", [0, 0, 0, 0, 0, 0]))
        ft_force = np.array(ft_wrench[:3], dtype=np.float32)
        ft_torque = np.array(ft_wrench[3:], dtype=np.float32)

        joint_pos = np.array(
            state.get("leftJointPositions", [0] * 7), dtype=np.float32
        )
        joint_vel = np.array(
            state.get("leftJointVelocities", [0] * 7), dtype=np.float32
        )

        obs = {
            "tcp.pos": tcp_pos,
            "tcp.quat": tcp_quat,
            "gripper.pos": gripper_pos,
            "ft_sensor.force": ft_force,
            "ft_sensor.torque": ft_torque,
            "joint.pos": joint_pos,
            "joint.vel": joint_vel,
        }

        for cam_key, cam in self.cameras.items():
            try:
                obs[cam_key] = cam.async_read()
            except Exception as e:
                logger.warning(f"Failed to read camera {cam_key}: {e}")
                if cam_key in self.observation_features:
                    shape = self.observation_features[cam_key]
                    obs[cam_key] = np.zeros(shape, dtype=np.uint8)

        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send action command to the robot via a single combined request."""
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        tcp_pos = np.array(action.get("tcp.pos", [0, 0, 0]), dtype=np.float32)
        tcp_quat = np.array(action.get("tcp.quat", [1, 0, 0, 0]), dtype=np.float32)
        gripper_pos = float(action.get("gripper.pos", 0.04))

        target_tcp = np.eoncatenate([tcp_pos, tcp_quat]).tolist()

        try:
            r = self._client.post(
                "/send_action/left",
                json={
                    "target_tcp": target_tcp,
                    "gripper_width": gripper_pos,
                    "gripper_velocity": self.config.gripper_velocity,
                    "gripper_force": self.config.gripper_force,
                    "feedforward_wrench": None,
                },
            )
            r.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send action: {e}")

        f_tcp = self._pool.submit(_send_tcp)
        f_grip = self._pool.submit(_send_gripper)
        f_tcp.result()
        f_grip.result()
        return {
            "tcp.pos": tcp_pos,
            "tcp.quat": tcp_quat,
            "gripper.pos": gripper_pos,
        }

    def go_home(self, timeout: float = 60.0) -> None:
        """Move the robot to its home position."""
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        logger.info("Moving robot to home position")
        response = self._client.post("/birobot_go_home", timeout=timeout)
        response.raise_for_status()
        logger.info("Robot at home position")

    def disconnect(self) -> None:
        """Disconnect from the robot and cleanup resources."""
        if not self._connected:
            logger.warning(f"{self} is not connected.")
            return

        logger.info(f"Disconnecting {self}")

        for cam_key, cam in self.cameras.items():
            try:
                cam.disconnect()
                logger.info(f"Disconnected camera: {cam_key}")
            except Exception as e:
                logger.warning(f"Failed to disconnect camera {cam_key}: {e}")

        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

        if self._client is not None:
            self._client.close()
            self._client = None

        self._connected = False
        logger.info(f"{self} disconnected")
