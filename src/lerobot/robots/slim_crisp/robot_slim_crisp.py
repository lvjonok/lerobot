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
from functools import cached_property
from typing import Any

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_slim_crisp import SlimCrispConfig

logger = logging.getLogger(__name__)


class SlimCrispRobot(Robot):
    """
    Robot wrapper for remote control via slim-crisp-zmq bridge.
    
    This class provides LeRobot-compatible interface for controlling a remote robot
    through ZMQ protocol. It supports Cartesian space control (x, y, z positioning)
    and is designed to work with teleoperation devices like Haply Inverse3.
    
    The robot communicates with a remote server running the slim-crisp-zmq bridge,
    which in turn controls the physical robot hardware.
    """

    config_class = SlimCrispConfig
    name = "slim_crisp"

    def __init__(self, config: SlimCrispConfig):
        super().__init__(config)
        self.config = config
        
        # Will be initialized in connect()
        self._robot = None
        self._gripper = None
        self._is_connected = False

    @cached_property
    def observation_features(self) -> dict[str, type]:
        """Define observation structure: Cartesian position + gripper state."""
        features = {
            "ee_pos_x": float,
            "ee_pos_y": float,
            "ee_pos_z": float,
        }
        
        if self.config.use_gripper:
            # TODO(lvjonok): Implement actual gripper observation
            features["gripper.pos"] = float
        
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Define action structure: matches observation for Cartesian control."""
        features = {
            "ee_pos_x": float,
            "ee_pos_y": float,
            "ee_pos_z": float,
        }
        
        if self.config.use_gripper:
            # TODO(lvjonok): Implement actual gripper action
            features["gripper.pos"] = float
        
        return features

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected and state is fresh."""
        if not self._is_connected or self._robot is None:
            return False
        
        # Check state freshness and warn if stale
        try:
            self._robot._check_state_freshness()
            return True
        except Exception as e:
            logger.warning(f"Robot connection issue: {e}")
            return False

    def connect(self, calibrate: bool = True) -> None:
        """
        Establish connection to remote robot via ZMQ.
        
        Args:
            calibrate: Ignored for remote robot (calibration handled server-side)
        """
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        
        # Import here to avoid dependency if not used
        try:
            from client import SlimRobot, SlimGripper
            from client.config import ClientConfig
        except ImportError as e:
            raise ImportError(
                "SlimCrispRobot requires slim-crisp-zmq package. "
                "Please install it from the local path or repository."
            ) from e
        
        # Create client configuration
        client_config = ClientConfig(
            server_ip=self.config.server_ip,
            state_pub_port=self.config.state_pub_port,
            cmd_rep_port=self.config.cmd_rep_port,
            command_timeout=self.config.command_timeout,
            max_state_delay=self.config.max_state_delay,
        )
        
        # Initialize robot client
        self._robot = SlimRobot(client_config)
        logger.info(f"Connected to robot at {self.config.server_ip}")
        
        # Initialize gripper client if needed
        if self.config.use_gripper:
            self._gripper = SlimGripper(client_config)
            logger.info("Gripper client initialized")
        
        self._is_connected = True

        # move to home
        self._robot.home()
        
        # Configure robot (switch controller if needed)
        self.configure()
        
        # Wait for initial state
        logger.info("Waiting for initial robot state...")
        import time
        for _ in range(10):
            self._robot.update()
            if self._robot._latest_state is not None:
                break
            time.sleep(0.1)
        
        if self._robot._latest_state is None:
            logger.warning("No initial state received from robot server")
        else:
            logger.info(f"{self} connected and ready")

    @property
    def is_calibrated(self) -> bool:
        """Remote robot calibration is handled server-side."""
        return True

    def calibrate(self) -> None:
        """No-op: calibration is handled by the remote robot server."""
        pass

    def configure(self) -> None:
        """Apply runtime configuration (switch controller if needed)."""
        if self._robot is None:
            return
        
        # Switch to desired controller
        try:
            logger.info(f"Switching controller to {self.config.default_controller}")
            self._robot.switch_controller("cartesian_impedance_controller")
        except Exception as e:
            logger.warning(f"Could not switch controller: {e}")

    def get_observation(self) -> dict[str, Any]:
        """
        Get current robot observation.
        
        Returns:
            Dictionary with end-effector position (ee_pos_x, ee_pos_y, ee_pos_z)
            and gripper state (gripper.pos) if enabled.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        # Update state from ZMQ
        self._robot.update()
        
        # Get end-effector pose
        ee_pose = self._robot.end_effector_pose
        
        obs = {
            "ee_pos_x": float(ee_pose.position[0]),
            "ee_pos_y": float(ee_pose.position[1]),
            "ee_pos_z": float(ee_pose.position[2]),
        }
        
        # Get gripper state if enabled
        if self.config.use_gripper and self._gripper is not None:
            # TODO(lvjonok): Implement actual gripper observation
            # For now, return placeholder value
            gripper_value = self._gripper.value
            obs["gripper.pos"] = float(gripper_value) if gripper_value is not None else 0.5
        
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Send action command to robot.
        
        Args:
            action: Dictionary with target end-effector position
                   (ee_pos_x, ee_pos_y, ee_pos_z) and optionally gripper.pos
        
        Returns:
            The action that was sent (potentially modified for safety)
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        # Extract Cartesian target position
        target_position = [
            action["ee_pos_x"],
            action["ee_pos_y"],
            action["ee_pos_z"],
        ]
        
        # Send Cartesian target to robot (non-blocking)
        # NOTE: don't really allow to send action yet
        self._robot.set_target(position=target_position)
        
        # Handle gripper if enabled
        if self.config.use_gripper and self._gripper is not None and "gripper.pos" in action:
            # TODO(lvjonok): Implement actual gripper action
            # For now, just log the gripper target
            gripper_target = action["gripper.pos"]
            # self._gripper.set_target(float(gripper_target))
            logger.debug(f"Gripper target: {gripper_target} (not implemented)")
        
        # Return the action that was sent
        return action

    def disconnect(self) -> None:
        """Disconnect from remote robot."""
        if self._robot is not None:
            self._robot.disconnect()
            self._robot = None
        
        if self._gripper is not None:
            self._gripper.disconnect()
            self._gripper = None
        
        self._is_connected = False
        logger.info(f"{self} disconnected")
