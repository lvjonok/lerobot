"""SpaceMouse teleoperator implementation for LeRobot.

Provides 6DOF teleoperation input from a 3Dconnexion SpaceMouse device.

Button mapping:
- Button 0 (left):  Toggle gripper open/close
- Button 1 (right): Reset reference position to current robot pose
"""

import logging
import threading
import time
from functools import cached_property
from typing import Any, Optional

import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator

from .config_spacemouse import SpaceMouseTeleopConfig

logger = logging.getLogger(__name__)


class SpaceMouseTeleop(Teleoperator):
    """LeRobot-compatible SpaceMouse teleoperator.

    Reads input from a 3Dconnexion SpaceMouse and converts it to
    robot actions (pose deltas and gripper commands).
    """

    config_class = SpaceMouseTeleopConfig
    name = "spacemouse"

    def __init__(self, config: SpaceMouseTeleopConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._teleop_mode = config.teleop_mode

        # State
        self._latest_motion = None
        self._latest_button = None
        self._gripper_open = True
        self._reset_reference = False
        self._lock = threading.Lock()

        # Reader thread
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def __str__(self) -> str:
        return f"SpaceMouseTeleop({self.config.id})"

    def reset(self):
        """Reset state for a new episode."""
        with self._lock:
            self._gripper_open = True
            self._latest_motion = None
            self._reset_reference = False

    def consume_reset_reference(self) -> bool:
        """Check and clear the reset-reference flag (right button)."""
        with self._lock:
            if self._reset_reference:
                self._reset_reference = False
                return True
            return False

    @cached_property
    def action_features(self) -> dict:
        return {
            "delta_pos": (3,),
            "delta_rot": (3,),
            "gripper.pos": float,
        }

    @cached_property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the SpaceMouse device."""
        if self._connected:
            logger.warning(f"{self} is already connected.")
            return

        try:
            import spnav
        except ImportError as e:
            raise ImportError(
                "SpaceMouseTeleop requires spnav. Install with: pip install spnav"
            ) from e

        logger.info("Connecting to SpaceMouse...")

        try:
            spnav.spnav_open()
            logger.info("SpaceMouse connected successfully")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to SpaceMouse: {e}")

        self._stop_event.clear()
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

        self._connected = True
        logger.info(f"{self} connected")

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _read_loop(self):
        """Background thread to continuously read SpaceMouse events."""
        import spnav

        while not self._stop_event.is_set():
            try:
                event = spnav.spnav_poll_event()
                if event is not None:
                    with self._lock:
                        if event.ev_type == spnav.SPNAV_EVENT_MOTION:
                            self._latest_motion = event
                        elif event.ev_type == spnav.SPNAV_EVENT_BUTTON:
                            self._process_button(event.bnum, event.press)
            except Exception as e:
                logger.warning(f"SpaceMouse read error: {e}")
            time.sleep(0.01)

    def _process_button(self, bnum: int, pressed: bool):  # noqa: FBT001
        """Handle button press/release (called under lock)."""
        if bnum == 0 and pressed:
            self._gripper_open = not self._gripper_open
            state_str = "open" if self._gripper_open else "closed"
            logger.info(f"Gripper toggled: {state_str}")
        elif bnum == 1 and pressed:
            self._reset_reference = True
            logger.info("Reference position reset requested")

    def get_action(self) -> dict[str, Any]:
        """Get the current action from the SpaceMouse."""
        if not self._connected:
            raise RuntimeError(f"{self} is not connected. Call connect() first.")

        delta_pos = np.zeros(3, dtype=np.float32)
        delta_rot = np.zeros(3, dtype=np.float32)

        with self._lock:
            gripper_width = 0.08 if self._gripper_open else 0.0

            if self._latest_motion is not None:
                motion = self._latest_motion

                # spnav [x, y, z] -> robot [z, -x, y]
                raw_trans = np.array(
                    [motion.translation[2], -motion.translation[0], motion.translation[1]],
                    dtype=np.float32,
                )
                raw_rot = np.array(
                    [motion.rotation[2], -motion.rotation[0], motion.rotation[1]],
                    dtype=np.float32,
                )

                if np.linalg.norm(raw_trans) > self.config.translation_deadzone:
                    delta_pos = raw_trans * self.config.translation_scale
                if np.linalg.norm(raw_rot) > self.config.rotation_deadzone:
                    delta_rot = raw_rot * self.config.rotation_scale

                # Filter rotation axes based on teleop_mode
                if self._teleop_mode == "left_arm_3D_translation":
                    delta_rot[:] = 0.0
                elif self._teleop_mode == "left_arm_3D_translation_Y_rotation":
                    delta_rot[0] = 0.0
                    delta_rot[2] = 0.0
                elif self._teleop_mode == "left_arm_3D_translation_Z_rotation":
                    delta_rot[0] = 0.0
                    delta_rot[1] = 0.0

        return {
            "delta_pos": delta_pos,
            "delta_rot": delta_rot,
            "gripper.pos": gripper_width,
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        """Disconnect from the SpaceMouse."""
        if not self._connected:
            logger.warning(f"{self} is not connected.")
            return

        logger.info(f"Disconnecting {self}")

        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None

        try:
            import spnav

            spnav.spnav_close()
        except Exception as e:
            logger.warning(f"Error closing spnav: {e}")

        self._connected = False
        logger.info(f"{self} disconnected")
