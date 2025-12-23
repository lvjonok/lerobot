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

"""
Processor for converting Haply delta actions to robot absolute positions with cumulative clutching.

This processor implements a clutch mechanism that allows the operator to control the robot
in multiple segments when the teleoperator's workspace is smaller than the robot's workspace.

Orientation control uses pure quaternion mathematics without Euler angle conversions:
- Compute quaternion delta: q_delta = q_current * q_ref^-1
- Apply frame transform: q_delta_robot = q_transform * q_delta * q_transform^-1
- Apply to robot: q_target = q_robot_ref * q_delta_robot
"""

from dataclasses import dataclass, field
from typing import Any

from .core import RobotAction, TransitionKey
from .pipeline import ProcessorStepRegistry, RobotActionProcessorStep


@ProcessorStepRegistry.register("haply_to_slim_crisp_clutch")
@dataclass
class HaplyToSlimCrispClutchProcessor(RobotActionProcessorStep):
    """Convert Haply raw positions/orientations to robot absolute pose with cumulative clutching.

    This processor:
    1. Receives raw Haply positions/quaternions and button state (is_controlling flag)
    2. When clutch engages: captures both Haply and robot poses as references
    3. During clutch: computes delta (position and quaternion) and applies to robot reference
    4. When clutch disengages: robot maintains last commanded pose
    5. Next clutch engagement: starts from current poses (cumulative)

    Position Control:
    - Maps Haply axes to robot axes (e.g., Haply X → Robot Y)
    - Applies scale factors for each axis

    Orientation Control (Pure Quaternion):
    - Computes quaternion delta: q_delta_haply = q_haply_current * q_haply_ref^-1
    - Transforms to robot frame: q_delta_robot = q_frame * q_delta_haply * q_frame^-1
    - Applies to robot: q_robot_target = q_robot_ref * q_delta_robot
    - No Euler angle conversions = no ambiguity, no gimbal lock

    Args:
        teleop_position_keys: Keys in teleop action for RAW positions (default: ['x', 'y', 'z'])
        robot_position_keys: Keys in robot observation/action for absolute position
        axis_mapping: Maps teleop axis index to robot axis index (default: [0, 1, 2] = identity)
        axis_scales: Scale factor for each teleop axis (default: [1.0, 1.0, 1.0])
        teleop_gripper_key: Key in teleop action for gripper (default: 'gripper')
        robot_gripper_key: Key in robot action for gripper (default: 'gripper.pos')
        clutch_flag_key: Key in teleop action for clutch state (default: 'is_controlling')
        delta_deadband: Ignore position deltas smaller than this (default: 1e-4 meters)
        teleop_rotation_keys: Keys for Haply quaternion [w,x,y,z] (default: ['qw','qx','qy','qz'])
        robot_rotation_keys: Keys for robot quaternion [w,x,y,z] (default: ['ee_quat_w', ...])
        enable_orientation: Enable orientation control (default: True)
        rotation_deadband: Ignore rotations smaller than this angle in radians (default: 0.01)
        orientation_frame_transform: Quaternion [x,y,z,w] representing rotation from Haply frame
            to robot frame (default: [0,0,0,1] = identity). Use this to align coordinate systems.

    Example:
        # Identity mapping
        processor = HaplyToSlimCrispClutchProcessor()

        # With frame alignment (e.g., 90° rotation around Z)
        import numpy as np
        processor = HaplyToSlimCrispClutchProcessor(
            orientation_frame_transform=[0, 0, np.sin(np.pi/4), np.cos(np.pi/4)],  # 90° around Z
        )
    """

    # Teleop action keys
    teleop_position_keys: list[str] = field(default_factory=lambda: ["x", "y", "z"])
    teleop_gripper_key: str = "gripper"
    clutch_flag_key: str = "is_controlling"
    teleop_rotation_keys: list[str] = field(default_factory=lambda: ["qw", "qx", "qy", "qz"])

    # Robot observation/action keys
    robot_position_keys: list[str] = field(default_factory=lambda: ["ee_pos_x", "ee_pos_y", "ee_pos_z"])
    robot_gripper_key: str = "gripper.pos"
    robot_rotation_keys: list[str] = field(
        default_factory=lambda: ["ee_quat_w", "ee_quat_x", "ee_quat_y", "ee_quat_z"]
    )

    # Position coordinate transformation
    axis_mapping: list[int] = field(default_factory=lambda: [0, 1, 2])
    axis_scales: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    # Orientation control
    enable_orientation: bool = True
    orientation_frame_transform: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 1.0]
    )  # Quaternion [x, y, z, w]

    # Drift prevention
    delta_deadband: float = 1e-4  # meters
    rotation_deadband: float = 0.01  # radians

    # Internal state (not serialized in config)
    initial_robot_position: dict[str, float] | None = field(default=None, init=False, repr=False)
    initial_robot_orientation: Any = field(default=None, init=False, repr=False)
    haply_position_at_clutch_start: dict[str, float] | None = field(default=None, init=False, repr=False)
    haply_orientation_at_clutch_start: Any = field(default=None, init=False, repr=False)
    robot_position_at_clutch_start: dict[str, float] | None = field(default=None, init=False, repr=False)
    robot_orientation_at_clutch_start: Any = field(default=None, init=False, repr=False)
    last_commanded_position: dict[str, float] | None = field(default=None, init=False, repr=False)
    last_commanded_orientation: Any = field(default=None, init=False, repr=False)
    was_clutch_active: bool = field(default=False, init=False, repr=False)
    frame_transform_rotation: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Validate configuration and precompute transforms."""
        if len(self.teleop_position_keys) != len(self.robot_position_keys):
            raise ValueError(
                f"teleop_position_keys ({len(self.teleop_position_keys)}) and "
                f"robot_position_keys ({len(self.robot_position_keys)}) must have same length"
            )

        if len(self.axis_mapping) != len(self.teleop_position_keys):
            raise ValueError(
                f"axis_mapping ({len(self.axis_mapping)}) must match "
                f"teleop_position_keys ({len(self.teleop_position_keys)})"
            )

        if len(self.axis_scales) != len(self.teleop_position_keys):
            raise ValueError(
                f"axis_scales ({len(self.axis_scales)}) must match "
                f"teleop_position_keys ({len(self.teleop_position_keys)})"
            )

        # Validate axis_mapping indices
        max_axis = len(self.robot_position_keys) - 1
        for i, axis_idx in enumerate(self.axis_mapping):
            if axis_idx < 0 or axis_idx > max_axis:
                raise ValueError(
                    f"axis_mapping[{i}] = {axis_idx} is out of range. Must be between 0 and {max_axis}"
                )

        # Validate orientation_frame_transform
        if len(self.orientation_frame_transform) != 4:
            raise ValueError(
                f"orientation_frame_transform must be a quaternion [x,y,z,w] with 4 elements, "
                f"got {len(self.orientation_frame_transform)}"
            )

        # Precompute frame transformation as Rotation object
        from scipy.spatial.transform import Rotation
        import numpy as np

        self.frame_transform_rotation = Rotation.from_quat(np.array(self.orientation_frame_transform))

    def action(self, action: RobotAction) -> RobotAction:
        """Convert raw Haply pose to absolute robot pose using cumulative clutching.

        Args:
            action: Teleop action dict with RAW positions/quaternions and clutch flag

        Returns:
            Robot action dict with absolute positions/quaternions
        """
        from scipy.spatial.transform import Rotation
        import numpy as np

        # Get observation from the current transition
        observation = self.transition.get(TransitionKey.OBSERVATION, {})

        # Capture initial robot position on first call
        if self.initial_robot_position is None:
            self.initial_robot_position = {key: observation.get(key, 0.0) for key in self.robot_position_keys}

        # Capture initial robot orientation on first call
        if self.enable_orientation and self.initial_robot_orientation is None:
            self.initial_robot_orientation = np.array([
                observation.get(self.robot_rotation_keys[0], 1.0),  # w
                observation.get(self.robot_rotation_keys[1], 0.0),  # x
                observation.get(self.robot_rotation_keys[2], 0.0),  # y
                observation.get(self.robot_rotation_keys[3], 0.0),  # z
            ])

        # Check clutch state
        clutch_active = action.get(self.clutch_flag_key, False)

        # Get RAW positions from Haply
        raw_haply_positions = [action.get(key, 0.0) for key in self.teleop_position_keys]

        # === Clutch State Machine ===

        # Detect clutch engagement
        if clutch_active and not self.was_clutch_active:
            # Capture Haply position reference
            self.haply_position_at_clutch_start = {
                key: action.get(key, 0.0) for key in self.teleop_position_keys
            }

            # Capture robot position reference (prefer last commanded)
            if self.last_commanded_position is not None:
                self.robot_position_at_clutch_start = self.last_commanded_position.copy()
            else:
                self.robot_position_at_clutch_start = {
                    key: observation.get(key, 0.0) for key in self.robot_position_keys
                }

            # Capture orientation references
            if self.enable_orientation:
                # Haply orientation reference
                haply_quat_wxyz = np.array([
                    action.get(self.teleop_rotation_keys[0], 1.0),
                    action.get(self.teleop_rotation_keys[1], 0.0),
                    action.get(self.teleop_rotation_keys[2], 0.0),
                    action.get(self.teleop_rotation_keys[3], 0.0),
                ])
                haply_quat_xyzw = np.array([haply_quat_wxyz[1], haply_quat_wxyz[2], 
                                           haply_quat_wxyz[3], haply_quat_wxyz[0]])
                self.haply_orientation_at_clutch_start = Rotation.from_quat(haply_quat_xyzw)

                # Robot orientation reference (prefer last commanded)
                if self.last_commanded_orientation is not None:
                    robot_quat_wxyz = self.last_commanded_orientation
                else:
                    robot_quat_wxyz = np.array([
                        observation.get(self.robot_rotation_keys[0], 1.0),
                        observation.get(self.robot_rotation_keys[1], 0.0),
                        observation.get(self.robot_rotation_keys[2], 0.0),
                        observation.get(self.robot_rotation_keys[3], 0.0),
                    ])

                robot_quat_xyzw = np.array([robot_quat_wxyz[1], robot_quat_wxyz[2], 
                                           robot_quat_wxyz[3], robot_quat_wxyz[0]])
                self.robot_orientation_at_clutch_start = Rotation.from_quat(robot_quat_xyzw)

        # Detect clutch disengagement
        elif not clutch_active and self.was_clutch_active:
            pass  # Hold last commanded pose

        self.was_clutch_active = clutch_active

        # === Compute Position Deltas ===

        if clutch_active and self.haply_position_at_clutch_start is not None:
            # Compute raw deltas
            teleop_deltas = [
                current - self.haply_position_at_clutch_start[key]
                for current, key in zip(raw_haply_positions, self.teleop_position_keys, strict=True)
            ]
            # Apply deadband
            teleop_deltas = [delta if abs(delta) > self.delta_deadband else 0.0 for delta in teleop_deltas]
        else:
            teleop_deltas = [0.0] * len(self.teleop_position_keys)

        # === Compute Target Position ===

        if clutch_active and self.robot_position_at_clutch_start is not None:
            # Transform deltas: scale and map axes
            robot_deltas = [0.0] * len(self.robot_position_keys)
            for delta, scale, robot_idx in zip(teleop_deltas, self.axis_scales, self.axis_mapping, strict=True):
                robot_deltas[robot_idx] = delta * scale

            # Compute target = reference + deltas
            target_positions = {
                robot_key: self.robot_position_at_clutch_start[robot_key] + robot_delta
                for robot_key, robot_delta in zip(self.robot_position_keys, robot_deltas, strict=True)
            }
            self.last_commanded_position = target_positions.copy()
        else:
            # Clutch inactive: hold last commanded or initial position
            if self.last_commanded_position is not None:
                target_positions = self.last_commanded_position.copy()
            else:
                target_positions = self.initial_robot_position.copy()

        # Build output action
        robot_action = target_positions.copy()
        
        # === Compute Target Orientation (Pure Quaternion Delta) ===

        if self.enable_orientation:
            if clutch_active and self.haply_orientation_at_clutch_start is not None and self.robot_orientation_at_clutch_start is not None:
                # Get current Haply orientation
                haply_quat_wxyz = np.array([
                    action.get(self.teleop_rotation_keys[0], 1.0),
                    action.get(self.teleop_rotation_keys[1], 0.0),
                    action.get(self.teleop_rotation_keys[2], 0.0),
                    action.get(self.teleop_rotation_keys[3], 0.0),
                ])
                haply_quat_xyzw = np.array([haply_quat_wxyz[1], haply_quat_wxyz[2], 
                                           haply_quat_wxyz[3], haply_quat_wxyz[0]])
                haply_current = Rotation.from_quat(haply_quat_xyzw)

                # Compute quaternion delta in Haply frame
                # q_delta = q_current * q_ref^-1
                quat_delta_haply = haply_current * self.haply_orientation_at_clutch_start.inv()

                # Transform delta to robot frame using conjugation
                # q_delta_robot = q_frame * q_delta_haply * q_frame^-1
                quat_delta_robot = (
                    self.frame_transform_rotation * quat_delta_haply * self.frame_transform_rotation.inv()
                )

                # Check if rotation is significant (above deadband)
                rotation_angle = quat_delta_robot.magnitude()  # radians

                if rotation_angle > self.rotation_deadband:
                    # Apply delta to robot reference: q_target = q_robot_ref * q_delta_robot
                    quat_target = self.robot_orientation_at_clutch_start * quat_delta_robot
                else:
                    # Below deadband: use reference
                    quat_target = self.robot_orientation_at_clutch_start

                # Convert to [w,x,y,z] for output
                target_quat_xyzw = quat_target.as_quat()  # [x,y,z,w]
                target_quat_wxyz = np.array([
                    target_quat_xyzw[3], target_quat_xyzw[0], 
                    target_quat_xyzw[1], target_quat_xyzw[2]
                ])

                self.last_commanded_orientation = target_quat_wxyz.copy()
            else:
                # Clutch inactive: hold last commanded or initial orientation
                if self.last_commanded_orientation is not None:
                    target_quat_wxyz = self.last_commanded_orientation
                else:
                    target_quat_wxyz = self.initial_robot_orientation

            # Add orientation to output
            robot_action[self.robot_rotation_keys[0]] = float(target_quat_wxyz[0])  # w
            robot_action[self.robot_rotation_keys[1]] = float(target_quat_wxyz[1])  # x
            robot_action[self.robot_rotation_keys[2]] = float(target_quat_wxyz[2])  # y
            robot_action[self.robot_rotation_keys[3]] = float(target_quat_wxyz[3])  # z

        # Pass through gripper (no transformation needed)
        if self.teleop_gripper_key in action:
            robot_action[self.robot_gripper_key] = action[self.teleop_gripper_key]

        return robot_action

    def get_config(self) -> dict[str, Any]:
        """Return configuration for serialization."""
        return {
            "teleop_position_keys": self.teleop_position_keys,
            "teleop_gripper_key": self.teleop_gripper_key,
            "clutch_flag_key": self.clutch_flag_key,
            "teleop_rotation_keys": self.teleop_rotation_keys,
            "robot_position_keys": self.robot_position_keys,
            "robot_gripper_key": self.robot_gripper_key,
            "robot_rotation_keys": self.robot_rotation_keys,
            "axis_mapping": self.axis_mapping,
            "axis_scales": self.axis_scales,
            "enable_orientation": self.enable_orientation,
            "orientation_frame_transform": self.orientation_frame_transform,
            "delta_deadband": self.delta_deadband,
            "rotation_deadband": self.rotation_deadband,
        }

    def reset(self) -> None:
        """Reset processor state between episodes."""
        self.initial_robot_position = None
        self.initial_robot_orientation = None
        self.haply_position_at_clutch_start = None
        self.haply_orientation_at_clutch_start = None
        self.robot_position_at_clutch_start = None
        self.robot_orientation_at_clutch_start = None
        self.last_commanded_position = None
        self.last_commanded_orientation = None
        self.was_clutch_active = False

    def transform_features(self, features: dict) -> dict:
        """Transform feature names from teleop format to robot format."""
        transformed = {}
        for key, value in features.items():
            if isinstance(value, dict) and "names" in value:
                new_names = {}

                # Map position keys
                for teleop_key, robot_key in zip(
                    self.teleop_position_keys, self.robot_position_keys, strict=True
                ):
                    if teleop_key in value["names"]:
                        teleop_idx = value["names"][teleop_key]
                        new_names[robot_key] = teleop_idx

                # Map gripper key
                if self.teleop_gripper_key in value["names"]:
                    new_names[self.robot_gripper_key] = value["names"][self.teleop_gripper_key]

                # Map rotation keys
                for teleop_key, robot_key in zip(
                    self.teleop_rotation_keys, self.robot_rotation_keys, strict=True
                ):
                    if teleop_key in value["names"]:
                        teleop_idx = value["names"][teleop_key]
                        new_names[robot_key] = teleop_idx

                transformed[key] = {**value, "names": new_names}
            else:
                transformed[key] = value

        return transformed
