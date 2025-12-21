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

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional

from ..config import TeleoperatorConfig
from ..teleoperator import Teleoperator
from ..utils import TeleopEvents


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

        # Control state
        self.is_controlling = False  # Toggled by button 'b'
        self.initial_position: dict | None = None
        self.initial_orientation: dict | None = None

        # Button state tracking
        self.prev_button_b = False
        self.prev_button_a = False
        self.prev_button_c = False

        # Gripper state (toggled by button 'a')
        self.gripper_closed = False

        # Keyboard events
        self.rerecord_requested = False

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"x": 0, "y": 1, "z": 2, "gripper": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"x": 0, "y": 1, "z": 2},
            }

    @property
    def feedback_features(self) -> dict:
        # No feedback needed - we only output deltas from initial Haply position
        return {
            "dtype": "float32",
            "shape": (0,),
            "names": {},
        }

    def connect(self, calibrate: bool = False) -> None:
        """Connect to the Haply device and start keyboard listener."""
        # Try importing here to avoid dependency if not used
        try:
            from .haply_utils import HaplyController
        except ImportError as e:
            raise ImportError(
                "HaplyTeleop requires the websockets and orjson packages. "
                "Please install them: pip install websockets orjson"
            ) from e

        self.haply_device = HaplyController()
        self.haply_device.start()

        # Start keyboard listener for 'R' key
        self._start_keyboard_listener()

    def is_connected(self) -> bool:
        """Check if Haply device is connected."""
        return self.haply_device is not None and self.haply_device.running

    def _update_buttons(self) -> dict[str, bool]:
        # all the logic handling buttons
        state = self.haply_device.get_state()
        buttons = state["buttons"]

        # toggle of active control with button 'b'
        current_button_b = buttons.get('b', False)
        if current_button_b and not self.prev_button_b:
            self.is_controlling = not self.is_controlling

            if self.is_controlling:
                # Start controlling - capture initial pose from Haply
                self.initial_position = state["xyz"].copy()
                self.initial_orientation = state["quat"].copy()
            else:
                # Stop controlling - reset initial pose
                self.initial_position = None
                self.initial_orientation = None

        self.prev_button_b = current_button_b

        # toggle of gripper state with button 'a'
        current_button_a = buttons.get(0, False)
        if current_button_a and not self.prev_button_a:
            self.gripper_closed = not self.gripper_closed
        self.prev_button_a = current_button_a

        return buttons

    def get_action(self) -> dict[str, Any]:
        state = self.haply_device.get_state()

        buttons = self._update_buttons()

        # Compute delta position from initial Haply position
        if self.is_controlling and self.initial_position is not None:
            current_pos = state["xyz"]
            delta_x = current_pos["x"] - self.initial_position["x"]
            delta_y = current_pos["y"] - self.initial_position["y"]
            delta_z = current_pos["z"] - self.initial_position["z"]
        else:
            # When not controlling, return zero deltas
            delta_x = delta_y = delta_z = 0.0

        action_dict = {
            "x": float(delta_x),
            "y": float(delta_y),
            "z": float(delta_z),
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

        success = buttons.get("c", False)
        rerecord_episode = self.rerecord_requested

        # Keyboard 'R' for rerecord
        # rerecord_episode = self.rerecord_requested
        self.rerecord_requested = False  # Reset after reading

        # Terminate episode if success or rerecord requested
        terminate_episode = success or rerecord_episode

        return {
            TeleopEvents.IS_INTERVENTION.value: self.is_controlling,
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
        try:
            from pynput import keyboard

            def on_press(key):
                try:
                    if hasattr(key, "char") and key.char == "r":
                        self.rerecord_requested = True
                except AttributeError:
                    pass

            self.keyboard_listener = keyboard.Listener(on_press=on_press)
            self.keyboard_listener.start()
        except ImportError:
            import logging

            logging.warning("pynput not installed. Keyboard 'R' for rerecord will not work.")

    def _stop_keyboard_listener(self) -> None:
        """Stop the keyboard listener."""
        if self.keyboard_listener is not None:
            self.keyboard_listener.stop()
            self.keyboard_listener = None

    def get_orientation_delta(self) -> Optional[dict]:
        """Get the orientation delta as quaternion difference."""
        if not self.is_controlling or self.initial_orientation is None:
            return None

        state = self.haply_device.get_state()
        current_quat = state["quat"]

        return {
            "initial": self.initial_orientation.copy(),
            "current": current_quat.copy(),
        }
