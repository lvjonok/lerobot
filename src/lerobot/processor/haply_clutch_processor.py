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
"""

from dataclasses import dataclass, field
from typing import Any

from .core import RobotAction, TransitionKey
from .pipeline import ProcessorStepRegistry, RobotActionProcessorStep


@ProcessorStepRegistry.register("haply_to_slim_crisp_clutch")
@dataclass
class HaplyToSlimCrispClutchProcessor(RobotActionProcessorStep):
    """Convert Haply raw positions to robot absolute positions with cumulative clutching.

    This processor:
    1. Receives raw Haply positions and button state (is_controlling flag)
    2. When clutch engages: captures both Haply and robot positions as references
    3. During clutch: computes delta = current_haply - haply_reference, then outputs robot_reference + delta (scaled and mapped)
    4. When clutch disengages: robot maintains last position
    5. Next clutch engagement: starts from current positions (cumulative)

    Coordinate mapping and scaling:
    - Maps Haply axes to robot axes (e.g., Haply X → Robot Y)
    - Applies scale factors for each axis
    - Handles gripper pass-through

    Args:
        teleop_position_keys: Keys in teleop action for RAW positions (default: ['x', 'y', 'z'])
        robot_position_keys: Keys in robot observation/action for absolute position (default: ['ee_pos_x', 'ee_pos_y', 'ee_pos_z'])
        axis_mapping: Maps teleop axis index to robot axis index (default: [0, 1, 2] = identity)
        axis_scales: Scale factor for each teleop axis (default: [1.0, 1.0, 1.0])
        teleop_gripper_key: Key in teleop action for gripper (default: 'gripper')
        robot_gripper_key: Key in robot action for gripper (default: 'gripper.pos')
        clutch_flag_key: Key in teleop action for clutch state (default: 'is_controlling')
        delta_deadband: Ignore deltas smaller than this to prevent drift (default: 1e-4 meters = 0.1mm)
        teleop_rotation_keys: Keys for rotation (for future extension, not used yet)
        robot_rotation_keys: Keys for rotation (for future extension, not used yet)

    Anti-Drift Features:
        1. Deadband filtering: Ignores tiny deltas below threshold (sensor noise)
        2. Position holding: Maintains last commanded position when clutch is released
        3. Smart re-engagement: Uses last commanded position (not observation) as reference

    Example:
        # Identity mapping (Haply X→Robot X, Y→Y, Z→Z)
        processor = HaplyToSlimCrispClutchProcessor()

        # Swap axes (Haply X→Robot Y, Haply Y→Robot X)
        processor = HaplyToSlimCrispClutchProcessor(
            axis_mapping=[1, 0, 2],  # [teleop_X→robot[1], teleop_Y→robot[0], teleop_Z→robot[2]]
        )

        # Scale axes (e.g., amplify Haply motion 2x in X, 0.5x in Z)
        processor = HaplyToSlimCrispClutchProcessor(
            axis_scales=[2.0, 1.0, 0.5],
        )
    """

    # Teleop action keys
    teleop_position_keys: list[str] = field(default_factory=lambda: ["x", "y", "z"])
    teleop_gripper_key: str = "gripper"
    clutch_flag_key: str = "is_controlling"
    teleop_rotation_keys: list[str] | None = None  # For future: ['qw', 'qx', 'qy', 'qz']

    # Robot observation/action keys
    robot_position_keys: list[str] = field(default_factory=lambda: ["ee_pos_x", "ee_pos_y", "ee_pos_z"])
    robot_gripper_key: str = "gripper.pos"
    robot_rotation_keys: list[str] | None = None  # For future: ['ee_quat_w', 'ee_quat_x', ...]

    # Coordinate transformation
    axis_mapping: list[int] = field(
        default_factory=lambda: [0, 1, 2]
    )  # teleop_axis_i → robot_axis[axis_mapping[i]]
    axis_scales: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    # Drift prevention
    delta_deadband: float = 1e-4  # Ignore deltas smaller than this (meters) to prevent drift

    # Internal state (not serialized in config)
    initial_robot_position: dict[str, float] | None = field(
        default=None, init=False, repr=False
    )  # Captured on first call
    haply_position_at_clutch_start: dict[str, float] | None = field(default=None, init=False, repr=False)
    robot_position_at_clutch_start: dict[str, float] | None = field(default=None, init=False, repr=False)
    last_commanded_position: dict[str, float] | None = field(default=None, init=False, repr=False)
    was_clutch_active: bool = field(default=False, init=False, repr=False)
    # was_clutch_active: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        """Validate configuration."""
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

    def action(self, action: RobotAction) -> RobotAction:
        """Convert raw Haply positions to absolute robot positions using cumulative clutching.

        Args:
            action: Teleop action dict with RAW positions and clutch flag

        Returns:
            Robot action dict with absolute positions
        """
        # Get observation from the current transition
        observation = self.transition.get(TransitionKey.OBSERVATION, {})

        # Capture initial robot position on first call (prevents drift before first clutch)
        if self.initial_robot_position is None:
            self.initial_robot_position = {key: observation.get(key, 0.0) for key in self.robot_position_keys}
            print(
                f"🤖 Initial robot position captured: "
                f"[{', '.join(f'{v:.3f}' for v in self.initial_robot_position.values())}]"
            )

        # Check clutch state
        clutch_active = action.get(self.clutch_flag_key, False)

        # Get RAW positions from Haply
        raw_haply_positions = [action.get(key, 0.0) for key in self.teleop_position_keys]

        # === Clutch State Machine ===

        # Detect clutch engagement (transition from inactive to active)
        if clutch_active and not self.was_clutch_active:
            # Capture BOTH Haply position and robot position as references
            self.haply_position_at_clutch_start = {
                key: action.get(key, 0.0) for key in self.teleop_position_keys
            }

            # Use last commanded position if available (more accurate), otherwise use observation
            if self.last_commanded_position is not None:
                self.robot_position_at_clutch_start = self.last_commanded_position.copy()
                print(
                    f"🎮 Clutch engaged! Using last commanded position as reference: "
                    f"[{', '.join(f'{v:.3f}' for v in self.robot_position_at_clutch_start.values())}]"
                )
            else:
                self.robot_position_at_clutch_start = {
                    key: observation.get(key, 0.0) for key in self.robot_position_keys
                }
                print(
                    f"🎮 Clutch engaged! Robot position captured: "
                    f"[{', '.join(f'{v:.3f}' for v in self.robot_position_at_clutch_start.values())}]"
                )

            print(
                f"    Haply reference: [{', '.join(f'{v:.3f}' for v in self.haply_position_at_clutch_start.values())}]"
            )

        # Detect clutch disengagement (transition from active to inactive)
        elif not clutch_active and self.was_clutch_active:
            print("✋ Clutch released! Robot will hold last position (preventing drift).")
            # last_commanded_position is already set from previous active cycle

        self.was_clutch_active = clutch_active

        # === Compute deltas from Haply movement ===

        if clutch_active and self.haply_position_at_clutch_start is not None:
            # Compute deltas: current_haply - haply_at_clutch_start
            teleop_deltas = [
                current - self.haply_position_at_clutch_start[key]
                for current, key in zip(raw_haply_positions, self.teleop_position_keys)
            ]

            # Apply deadband to filter noise and prevent drift
            teleop_deltas = [delta if abs(delta) > self.delta_deadband else 0.0 for delta in teleop_deltas]
        else:
            # Clutch inactive: no deltas
            teleop_deltas = [0.0] * len(self.teleop_position_keys)

        # === Compute Target Position ===

        if clutch_active and self.robot_position_at_clutch_start is not None:
            # Clutch is active: apply scaled and mapped deltas to reference position

            # Transform deltas: scale and map from teleop axes to robot axes
            robot_deltas = [0.0] * len(self.robot_position_keys)
            for _teleop_idx, (delta, scale, robot_idx) in enumerate(
                zip(teleop_deltas, self.axis_scales, self.axis_mapping, strict=True)
            ):
                robot_deltas[robot_idx] = delta * scale

            # Compute absolute target: reference + transformed deltas
            target_positions = {
                robot_key: self.robot_position_at_clutch_start[robot_key] + robot_delta
                for robot_key, robot_delta in zip(self.robot_position_keys, robot_deltas, strict=True)
            }

            # Store this as the last commanded position
            self.last_commanded_position = target_positions.copy()
        else:
            # Clutch inactive: hold last commanded position (prevents drift!)
            if self.last_commanded_position is not None:
                target_positions = self.last_commanded_position.copy()
            else:
                # Before first clutch: hold initial position (prevents drift from observation noise)
                target_positions = self.initial_robot_position.copy()

        # Build output action
        robot_action = target_positions.copy()

        # Pass through gripper (no transformation needed)
        if self.teleop_gripper_key in action:
            robot_action[self.robot_gripper_key] = action[self.teleop_gripper_key]

        # TODO: Handle rotation when implemented
        # if self.teleop_rotation_keys and self.robot_rotation_keys:
        #     for teleop_key, robot_key in zip(self.teleop_rotation_keys, self.robot_rotation_keys):
        #         if teleop_key in action:
        #             robot_action[robot_key] = action[teleop_key]

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
            "delta_deadband": self.delta_deadband,
        }

    def reset(self) -> None:
        """Reset processor state between episodes."""
        self.initial_robot_position = None
        self.haply_position_at_clutch_start = None
        self.robot_position_at_clutch_start = None
        self.last_commanded_position = None
        self.was_clutch_active = False

    def transform_features(self, features: dict) -> dict:
        """Transform feature names from teleop format to robot format.

        This updates the action feature metadata to reflect the output keys.
        """
        transformed = {}
        for key, value in features.items():
            if isinstance(value, dict) and "names" in value:
                # Build new names dict with robot keys
                new_names = {}

                # Map position keys
                for teleop_key, robot_key in zip(
                    self.teleop_position_keys, self.robot_position_keys, strict=True
                ):
                    if teleop_key in value["names"]:
                        # Map through axis_mapping
                        teleop_idx = value["names"][teleop_key]
                        new_names[robot_key] = teleop_idx

                # Map gripper key
                if self.teleop_gripper_key in value["names"]:
                    new_names[self.robot_gripper_key] = value["names"][self.teleop_gripper_key]

                # TODO: Map rotation keys when implemented

                transformed[key] = {**value, "names": new_names}
            else:
                transformed[key] = value

        return transformed
