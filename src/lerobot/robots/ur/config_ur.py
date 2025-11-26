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

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("ur_rtde")
@dataclass
class URRobotConfig(RobotConfig):
    """Configuration for controlling a Universal Robots arm through RTDE."""

    robot_ip: str = "192.168.50.201"

    # Control strategy
    use_admittance: bool = True
    admittance_frequency_hz: float = 1000.0
    admittance_mass: tuple[float, ...] = (20.0, 20.0, 20.0, 2.0, 2.0, 2.0)
    admittance_damping: tuple[float, ...] = (800.0, 800.0, 800.0, 80.0, 80.0, 80.0)
    admittance_stiffness: tuple[float, ...] = (8000.0, 8000.0, 8000.0, 800.0, 800.0, 800.0)

    servo_frequency_hz: float = 500.0
    servo_velocity: float = 0.5
    servo_acceleration: float = 1.5
    servo_lookahead_time: float = 0.1
    servo_gain: float = 400.0

    # Teleoperation mapping
    max_translation_m: float = 0.3
    max_rotation_rad: float = 1.5
    limit_rotation_deltas: bool = True
    command_period_s: float = 1.0 / 60.0
    virtual_gripper_speed: float = 0.5

    force_lowpass_alpha: float = 0.1

    # TODO: set this to your desired 6D joint vector for automatic reset, or keep None to skip.
    home_joint_positions: tuple[float, ...] | None = None

    # Schunk gripper IO configuration
    gripper_ready_channel: int = 0
    gripper_open_channel: int = 1
    gripper_close_channel: int = 2
    gripper_settle_time: float = 0.05
    gripper_pulse_duration: float = 0.2

    monitor_dashboard: bool = True
    dashboard_retry_attempts: int = 3

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
