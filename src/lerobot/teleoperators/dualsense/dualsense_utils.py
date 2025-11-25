#!/usr/bin/env python

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

import logging
import time
from ..utils import TeleopEvents
from .gamepad_utils import InputController
from dualsense_controller import DualsenseController


class DualSenseController(InputController):
    """Generate motion deltas from dualsense input."""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0, deadzone=0.1):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.dualsense = None
        self.intervention_flag = False

    def start(self):
        """Initialize dualsense and the gamepad."""
        try:
            self.dualsense = DualsenseController()
            self.dualsense.activate()
        except FileNotFoundError:
            logging.error("No dualsense detected. Please connect a dualsense and try again.")
            self.running = False
            return

        logging.info(f"Initialized dualsense: {self.dualsense.device.name}")

        print("DualSense controls:")
        print("  Left analog stick: Move in X-Y plane")
        print("  Right analog stick (vertical): Move in Z axis")
        print("  Triangle button: End episode with SUCCESS")
        print("  Cross button: End episode with FAILURE")
        print("  Square button: Rerecord episode")

    def stop(self):
        """Clean up dualsense resources."""
        if self.dualsense:
            self.dualsense.deactivate()

    def update(self):
        """Process events to get fresh dualsense readings."""
        if not self.dualsense:
            return

        # BTN_TRIANGLE for success
        if self.dualsense.state.triangle:
            self.episode_end_status = TeleopEvents.SUCCESS
        # BTN_CROSS for failure
        elif self.dualsense.state.cross:
            self.episode_end_status = TeleopEvents.FAILURE
        # BTN_SQUARE for rerecord
        elif self.dualsense.state.square:
            self.episode_end_status = TeleopEvents.RERECORD_EPISODE
        else:
            self.episode_end_status = None

        # R1 for closing gripper
        self.close_gripper_command = self.dualsense.state.R1

        # L1 for opening gripper
        self.open_gripper_command = self.dualsense.state.L1

        # Check for R2 button for intervention flag
        self.intervention_flag = self.dualsense.state.R2 > 0

    def get_deltas(self):
        """Get the current movement deltas from dualsense state."""
        if not self.dualsense:
            return 0.0, 0.0, 0.0

        # Read joystick axes
        y_input = self.dualsense.state.LX
        x_input = self.dualsense.state.LY

        # Right stick Y
        z_input = self.dualsense.state.RY

        # Apply deadzone to avoid drift
        x_input = 0 if abs(x_input) < self.deadzone * 255 else x_input
        y_input = 0 if abs(y_input) < self.deadzone * 255 else y_input
        z_input = 0 if abs(z_input) < self.deadzone * 255 else z_input

        # Calculate deltas and normalize
        delta_x = -x_input / 255.0 * self.x_step_size
        delta_y = -y_input / 255.0 * self.y_step_size
        delta_z = -z_input / 255.0 * self.z_step_size

        return delta_x, delta_y, delta_z
