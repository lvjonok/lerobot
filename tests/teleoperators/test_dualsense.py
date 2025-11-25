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

import unittest
from unittest.mock import MagicMock, patch

from lerobot.teleoperators.dualsense.configuration_dualsense import DualSenseTeleopConfig
from lerobot.teleoperators.dualsense.teleop_dualsense import DualSenseTeleop


class TestDualSenseTeleop(unittest.TestCase):
    def test_smoke(self):
        with patch("lerobot.teleoperators.dualsense.dualsense_utils.DualsenseController") as MockDualSenseController:
            # Arrange
            mock_dualsense_instance = MagicMock()
            MockDualSenseController.return_value = mock_dualsense_instance

            config = DualSenseTeleopConfig()
            teleop = DualSenseTeleop(config)

            # Act
            teleop.connect()
            action = teleop.get_action()
            events = teleop.get_teleop_events()
            teleop.disconnect()

            # Assert
            mock_dualsense_instance.activate.assert_called_once()
            mock_dualsense_instance.deactivate.assert_called_once()
            self.assertIn("delta_x", action)
            self.assertIn("delta_y", action)
            self.assertIn("delta_z", action)
            self.assertIn("gripper", action)
            self.assertIn("is_intervention", events)
            self.assertIn("terminate_episode", events)
            self.assertIn("success", events)
            self.assertIn("rerecord_episode", events)

if __name__ == "__main__":
    unittest.main()
