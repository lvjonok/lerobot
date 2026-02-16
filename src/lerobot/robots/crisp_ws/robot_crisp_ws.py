"""WebSocket-based crisp_py robot implementation for LeRobot.

This robot communicates with a crisp_py-based WebSocket server, providing
lower latency than the HTTP-based CrispFastAPIRobot by using a persistent
connection with binary orjson frames.
"""

import asyncio
import logging
import threading
from concurrent.futures import Future
from functools import cached_property
from typing import Any

import numpy as np
import orjson
import websockets

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_crisp_ws import CrispWSConfig

logger = logging.getLogger(__name__)


class CrispWSRobot(Robot):
    """LeRobot-compatible robot using crisp_py WebSocket server.

    Uses a background thread running an asyncio event loop to maintain a
    persistent WebSocket connection. The ``send_action`` method performs a
    single WS round-trip that sends an action and receives the resulting
    observation, merging two HTTP calls into one exchange.
    """

    config_class = CrispWSConfig
    name = "crisp_ws"

    def __init__(self, config: CrispWSConfig):
        super().__init__(config)
        self.config = config
        self._connected = False

        # Background asyncio event loop for WebSocket
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._ws: websockets.ClientConnection | None = None

        # Cameras
        self.cameras = {}
        if config.cameras:
            self.cameras = make_cameras_from_configs(config.cameras)

    def __str__(self) -> str:
        return f"CrispWSRobot({self.config.id})"

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

    # ── Background event loop ────────────────────────────────────────────

    def _run_event_loop(self):
        """Run asyncio event loop in a background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        self._loop.run_forever()
        self._loop.close()

    def _submit(self, coro) -> Any:
        """Submit a coroutine to the background loop and block until done."""
        future: Future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=10.0)

    async def _ws_connect(self, url: str):
        """Open a WebSocket connection (awaitable wrapper)."""
        return await websockets.connect(url)

    async def _ws_send_recv(self, msg: dict) -> dict:
        """Send a message and receive the response over WebSocket."""
        await self._ws.send(orjson.dumps(msg))
        raw = await self._ws.recv()
        return orjson.loads(raw)

    async def _ws_close(self):
        """Close the WebSocket connection."""
        await self._ws.close()

    # ── Lifecycle ────────────────────────────────────────────────────────

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info(f"Connecting to crisp WS server at {self.config.ws_url}")

        # Start background event loop thread
        self._loop_ready.clear()
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()
        self._loop_ready.wait(timeout=5.0)

        # Open WebSocket connection
        try:
            self._ws = self._submit(self._ws_connect(self.config.ws_url))
        except Exception as e:
            self._stop_loop()
            raise ConnectionError(f"Failed to connect to crisp WS server: {e}")

        # Verify connection with an observation request
        try:
            resp = self._submit(self._ws_send_recv({"type": "get_observation"}))
            if resp.get("type") != "observation":
                raise ConnectionError(f"Unexpected response: {resp}")
            logger.info("Successfully connected to crisp WS server")
        except Exception as e:
            self._submit(self._ws_close())
            self._stop_loop()
            raise ConnectionError(f"Failed initial observation: {e}")

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

    def _stop_loop(self):
        """Stop the background event loop and join the thread."""
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._loop = None

    def disconnect(self) -> None:
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

        if self._ws is not None:
            try:
                self._submit(self._ws_close())
            except Exception:
                pass
            self._ws = None

        self._stop_loop()
        self._connected = False
        logger.info(f"{self} disconnected")

    # ── Observation / Action ─────────────────────────────────────────────

    def _parse_observation(self, data: dict) -> dict[str, Any]:
        """Parse a server observation message into the lerobot obs dict."""
        tcp = data.get("tcp", [0, 0, 0, 1, 0, 0, 0])
        tcp_pos = np.array(tcp[:3], dtype=np.float32)
        tcp_quat = np.array(tcp[3:], dtype=np.float32)

        gripper_state = data.get("gripper", [0, 0])
        gripper_pos = float(gripper_state[0])

        ft = data.get("ft_sensor", [0, 0, 0, 0, 0, 0])
        ft_force = np.array(ft[:3], dtype=np.float32)
        ft_torque = np.array(ft[3:], dtype=np.float32)

        joint_pos = np.array(data.get("joint_pos", [0] * 7), dtype=np.float32)
        joint_vel = np.array(data.get("joint_vel", [0] * 7), dtype=np.float32)

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

    def get_observation(self) -> dict[str, Any]:
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        resp = self._submit(self._ws_send_recv({"type": "get_observation"}))
        return self._parse_observation(resp)

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send action and receive the resulting observation in one WS round-trip."""
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        tcp_pos = np.array(action.get("tcp.pos", [0, 0, 0]), dtype=np.float32)
        tcp_quat = np.array(action.get("tcp.quat", [1, 0, 0, 0]), dtype=np.float32)
        gripper_pos = float(action.get("gripper.pos", 0.04))

        target_tcp = np.concatenate([tcp_pos, tcp_quat]).tolist()

        msg = {
            "type": "action",
            "target_tcp": target_tcp,
            "gripper_width": gripper_pos,
            "gripper_velocity": self.config.gripper_velocity,
            "gripper_force": self.config.gripper_force,
            "feedforward_wrench": None,
        }

        self._submit(self._ws_send_recv(msg))

        return {
            "tcp.pos": tcp_pos,
            "tcp.quat": tcp_quat,
            "gripper.pos": gripper_pos,
        }

    def go_home(self, timeout: float = 60.0) -> None:
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        logger.info("Moving robot to home position")
        future: Future = asyncio.run_coroutine_threadsafe(
            self._ws_send_recv({"type": "go_home"}), self._loop
        )
        resp = future.result(timeout=timeout)
        if resp.get("status") != "ok":
            raise RuntimeError(f"go_home failed: {resp}")
        logger.info("Robot at home position")
