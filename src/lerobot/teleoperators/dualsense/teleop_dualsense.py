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

from enum import IntEnum
from typing import Any

import numpy as np

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_dualsense import DualSenseTeleopConfig


class GripperAction(IntEnum):
    CLOSE = 0
    STAY = 1
    OPEN = 2


gripper_action_map = {
    "close": GripperAction.CLOSE.value,
    "open": GripperAction.OPEN.value,
    "stay": GripperAction.STAY.value,
}


class DualSenseTeleop(Teleoperator):
    """
    Teleop class to use DualSense controller inputs for control.
    """

    config_class = DualSenseTeleopConfig
    name = "dualsense"

    def __init__(self, config: DualSenseTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.dualsense = None

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
            }

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self) -> None:
        from .dualsense_utils import DualSenseController as DualSense

        self.dualsense = DualSense()
        self.dualsense.start()

    def get_action(self) -> dict[str, Any]:
        # Update the controller to get fresh inputs
        self.dualsense.update()

        # Get movement deltas from the controller
        delta_x, delta_y, delta_z = self.dualsense.get_deltas()

        # Create action from dualsense input
        dualsense_action = np.array([delta_x, delta_y, delta_z], dtype=np.float32)

        action_dict = {
            "delta_x": dualsense_action[0],
            "delta_y": dualsense_action[1],
            "delta_z": dualsense_action[2],
        }

        # Default gripper action is to stay
        gripper_action = GripperAction.STAY.value
        if self.config.use_gripper:
            gripper_command = self.dualsense.gripper_command()
            gripper_action = gripper_action_map[gripper_command]
            action_dict["gripper"] = gripper_action

        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the dualsense such as intervention status,
        episode termination, success indicators, etc.

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
        """
        if self.dualsense is None:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # Update dualsense state to get fresh inputs
        self.dualsense.update()

        # Check if intervention is active
        is_intervention = self.dualsense.should_intervene()

        # Get episode end status
        episode_end_status = self.dualsense.get_episode_end_status()
        terminate_episode = episode_end_status in [
            TeleopEvents.RERECORD_EPISODE,
            TeleopEvents.FAILURE,
        ]
        success = episode_end_status == TeleopEvents.SUCCESS
        rerecord_episode = episode_end_status == TeleopEvents.RERECORD_EPISODE

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }

    def disconnect(self) -> None:
        """Disconnect from the dualsense."""
        if self.dualsense is not None:
            self.dualsense.stop()
            self.dualsense = None

    def is_connected(self) -> bool:
        """Check if dualsense is connected."""
        return self.dualsense is not None

    def calibrate(self) -> None:
        """Calibrate the dualsense."""
        # No calibration needed for dualsense
        pass

    def is_calibrated(self) -> bool:
        """Check if dualsense is calibrated."""
        # dualsense doesn't require calibration
        return True

    def configure(self) -> None:
        """Configure the dualsense."""
        # No additional configuration needed
        pass

    def send_feedback(self, feedback: dict) -> None:
        """Send feedback to the dualsense."""
        # dualsense doesn't support feedback
        pass
