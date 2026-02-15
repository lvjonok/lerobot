# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional

from pynput import keyboard

from ..config import TeleoperatorConfig
from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .haply_utils import HaplyController


class GripperAction(IntEnum):
    CLOSE = 0
    STAY = 1
    OPEN = 2


gripper_action_map = {
    "close": GripperAction.CLOSE.value,
    "open": GripperAction.OPEN.value,
    "stay": GripperAction.STAY.value,
}


@TeleoperatorConfig.register_subclass("haply")
@dataclass
class HaplyTeleopConfig(TeleoperatorConfig):
    use_gripper: bool = True

    # Haply Inverse Service WebSocket URI
    ws_uri: str = "ws://localhost:10001"

    # Control scaling (used by processor, stored here for preset convenience)
    translation_scale: float = 1.0
    rotation_scale: float = 1.0

    # Teleop mode — which axes are actuated (used by processor)
    teleop_mode: str = "left_arm_6DOF"

    # Gripper
    max_gripper_width: float = 0.08  # meters

    # Force feedback
    enable_feedback: bool = False


class HaplyTeleop(Teleoperator):
    """
    Teleop class to use Haply Inverse3 with VerseGrip for control.
    """

    config_class = HaplyTeleopConfig
    name = "haply"

    def __init__(self, config: HaplyTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.haply_device = None
        self.keyboard_listener = None

        # Button state tracking for gripper toggle
        self.prev_button_a = False

        # Gripper state (toggled by button 'a')
        self.gripper_closed = False

        # Keyboard events
        self.rerecord_requested = False

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (8,),
                "names": {"x": 0, "y": 1, "z": 2, "qw": 3, "qx": 4, "qy": 5, "qz": 6, "gripper": 7},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (7,),
                "names": {"x": 0, "y": 1, "z": 2, "qw": 3, "qx": 4, "qy": 5, "qz": 6},
            }

    @property
    def feedback_features(self) -> dict:
        # No feedback needed - we output raw Haply positions
        return {
            "dtype": "float32",
            "shape": (0,),
            "names": {},
        }

    def connect(self, calibrate: bool = False) -> None:
        """Connect to the Haply device and start keyboard listener."""

        self.haply_device = HaplyController(uri=self.config.ws_uri)
        self.haply_device.start()

        # Wait for connection to establish
        timeout = 5.0
        start = time.time()
        while time.time() - start < timeout:
            if self.haply_device.running and self.haply_device.inverse3_device_id is not None:
                break
            time.sleep(0.1)

        if not self.haply_device.running:
            self.haply_device.stop()
            self.haply_device = None
            raise ConnectionError(
                f"Failed to connect to Haply Inverse Service at {self.config.ws_uri}. "
                "Make sure the Inverse Service is running."
            )

        # Start keyboard listener for 'R' key
        self._start_keyboard_listener()

    def is_connected(self) -> bool:
        """Check if Haply device is connected."""
        return self.haply_device is not None and self.haply_device.running

    def _update_buttons(self) -> dict[str, bool]:
        """Update button states and handle gripper toggle."""
        state = self.haply_device.get_state()
        buttons = state["buttons"]

        # Toggle gripper state with button 'a'
        current_button_a = buttons.get('a', False)
        if current_button_a and not self.prev_button_a:
            self.gripper_closed = not self.gripper_closed
        self.prev_button_a = current_button_a

        return buttons

    def get_action(self) -> dict[str, Any]:
        """Get raw Haply state and button values."""
        state = self.haply_device.get_state()
        buttons = self._update_buttons()

        # Output RAW position and orientation from Haply (processor will handle clutching and deltas)
        current_pos = state["xyz"]
        current_quat = state["quat"]  # Get raw orientation

        action_dict = {
            "x": float(current_pos["x"]),
            "y": float(current_pos["y"]),
            "z": float(current_pos["z"]),
            "qw": float(current_quat["w"]),
            "qx": float(current_quat["x"]),
            "qy": float(current_quat["y"]),
            "qz": float(current_quat["z"]),
            "is_controlling": buttons.get("b", False),  # Raw button state
        }

        # Handle gripper with button 'a' using toggle logic
        if self.config.use_gripper:
            gripper_action = GripperAction.CLOSE.value if self.gripper_closed else GripperAction.OPEN.value
            action_dict["gripper"] = gripper_action

        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        """Get teleoperation events from buttons and keyboard."""

        if self.haply_device is None:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        buttons = self._update_buttons()

        # Button 'b' indicates intervention (held to control)
        is_intervention = buttons.get("b", False)
        success = buttons.get("c", False)
        rerecord_episode = self.rerecord_requested

        # Reset rerecord flag after reading
        self.rerecord_requested = False

        # Terminate episode if success or rerecord requested
        terminate_episode = success or rerecord_episode

        return {
            TeleopEvents.IS_INTERVENTION.value: is_intervention,
            TeleopEvents.TERMINATE_EPISODE.value: terminate_episode,
            TeleopEvents.SUCCESS.value: success,
            TeleopEvents.RERECORD_EPISODE.value: rerecord_episode,
        }

    def calibrate(self) -> None:
        pass

    def disconnect(self) -> None:
        """Disconnect from the Haply device and stop keyboard listener."""
        if self.haply_device is not None:
            self.haply_device.stop()
            self.haply_device = None

        # Stop keyboard listener
        self._stop_keyboard_listener()

    def is_calibrated(self) -> bool:
        return True

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def _start_keyboard_listener(self) -> None:
        """Start listening for keyboard events."""
        def on_press(key):
            try:
                if hasattr(key, "char") and key.char == "r":
                    self.rerecord_requested = True
            except AttributeError:
                pass

        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.start()

    def _stop_keyboard_listener(self) -> None:
        """Stop the keyboard listener."""
        if self.keyboard_listener is not None:
            self.keyboard_listener.stop()
            self.keyboard_listener = None

    def get_orientation(self) -> Optional[dict]:
        """Get the raw orientation as quaternion."""
        if self.haply_device is None:
            return None

        state = self.haply_device.get_state()
        return state["quat"].copy()
