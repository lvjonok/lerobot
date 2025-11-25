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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("omega3")
@TeleoperatorConfig.register_subclass("omega6")
@dataclass
class ForceDimensionOmegaConfig(TeleoperatorConfig):
    """
    Configuration for the Force Dimension Omega teleoperators (Omega.3, Omega.6, Omega.7, ...).

    Attributes:
        device_index: Select a specific device by index as reported by the SDK.
        serial_number: Select a specific device by serial number (if supported).
        device_type: Name (or enum value) of the DeviceType to open. Defaults depend on ``config.type``.
        translation_scale: Global scale applied to linear deltas (in meters).
        rotation_scale: Global scale applied to rotation deltas (in radians).
        translation_deadband_m: Threshold under which linear motion is ignored.
        rotation_deadband_rad: Threshold under which rotation is ignored.
        enable_button_index: Optional button index that toggles teleoperation.
        recenter_on_enable: When True, latch a new neutral pose on enable rising edge.
        gripper_open_button_index: Optional button index mapped to positive gripper velocity.
        gripper_close_button_index: Optional button index mapped to negative gripper velocity.
        gripper_speed: Magnitude of the gripper velocity command when the button is held.
    """

    device_index: int | None = None
    serial_number: int | None = None
    device_type: str | int | None = None

    translation_scale: float = 1.0
    rotation_scale: float = 1.0
    translation_deadband_m: float = 1e-4
    rotation_deadband_rad: float = 5e-3

    enable_button_index: int | None = 0
    recenter_on_enable: bool = True

    gripper_open_button_index: int | None = None
    gripper_close_button_index: int | None = None
    gripper_speed: float = 1.0

    def __post_init__(self) -> None:
        if self.device_index is not None and self.device_index < 0:
            raise ValueError("device_index must be non-negative when provided.")
        if self.serial_number is not None and self.serial_number < 0:
            raise ValueError("serial_number must be non-negative when provided.")
        if self.device_index is not None and self.serial_number is not None:
            raise ValueError("Only one of device_index or serial_number can be set.")
        if self.device_type is None:
            # Default to the most likely target based on the registered config type.
            if self.type == "omega6":
                self.device_type = "OMEGA6_RIGHT"
            else:
                self.device_type = "OMEGA3"
