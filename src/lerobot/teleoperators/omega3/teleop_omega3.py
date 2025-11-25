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
from typing import Any

import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.rotation import Rotation

from .config_omega3 import Omega3Config

logger = logging.getLogger(__name__)

try:
    from forcedimension_core import dhd
    from forcedimension_core.constants import DeviceType, ErrorNum
except ImportError as import_err:  # pragma: no cover - exercised when dependency missing
    dhd = None
    DeviceType = None
    ErrorNum = None
    _IMPORT_ERROR = import_err
else:
    _IMPORT_ERROR = None


def _ensure_sdk_available() -> None:
    if _IMPORT_ERROR is not None:
        raise ImportError(
            "The Omega3 teleoperator requires the Force Dimension Python bindings. "
            "Install them via `pip install lerobot[omega3]`."
        ) from _IMPORT_ERROR


def _identity_rotation() -> Rotation:
    return Rotation.from_quat(np.array([0.0, 0.0, 0.0, 1.0], dtype=float))


class Omega3(Teleoperator):
    """
    Teleoperator for the Force Dimension Omega.3 haptic device.

    The driver exposes calibrated deltas in translation and rotation, matching the expected
    target action format (`target_*` keys) used by the default teleoperation pipelines.
    """

    config_class = Omega3Config
    name = "omega3"

    def __init__(self, config: Omega3Config):
        _ensure_sdk_available()
        super().__init__(config)
        self.config = config
        self._device_id: int | None = None
        self._zero_pos = np.zeros(3, dtype=float)
        self._zero_rot = _identity_rotation()
        self._has_reference = False
        self._enabled = False
        self._last_button_mask = 0

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "enabled": bool,
            "target_x": float,
            "target_y": float,
            "target_z": float,
            "target_wx": float,
            "target_wy": float,
            "target_wz": float,
            "gripper_vel": float,
            "omega3.button_mask": int,
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        # Haptic feedback has not been implemented yet.
        return {}

    @property
    def is_connected(self) -> bool:
        return self._device_id is not None

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        device_id = self._open_device()
        if device_id < 0:
            self._raise_last_error("Unable to open Omega.3 device")
        self._device_id = device_id

        # Disable force output for passive teleoperation.
        dhd.enableForce(False, self._device_id)
        logger.info("Connected to Omega.3 (id=%s, serial=%s)", self._device_id, self._serial_number())

        if calibrate:
            self.calibrate()

    def _open_device(self) -> int:
        if self.config.serial_number is not None:
            return dhd.openSerial(int(self.config.serial_number))
        if self.config.device_index is not None:
            return dhd.openID(int(self.config.device_index))

        target_type = self._resolve_device_type()
        return dhd.openType(target_type)

    def _resolve_device_type(self) -> DeviceType:
        if DeviceType is None:
            raise RuntimeError("forcedimension_core DeviceType enum is not available.")

        value = self.config.device_type or "OMEGA3"
        if isinstance(value, str):
            key = value.upper()
            if not hasattr(DeviceType, key):
                raise ValueError(f"Unknown DeviceType '{value}'")
            return getattr(DeviceType, key)

        return DeviceType(int(value))

    def _serial_number(self) -> int | None:
        if not self.is_connected:
            return None
        sn = dhd.getSerialNumber(self._device_id)
        return None if sn < 0 else sn

    def calibrate(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        print(
            "Hold the Omega.3 handle in the neutral pose you want to consider as zero "
            "and press ENTER to capture it."
        )
        input("Ready? Press ENTER to capture the reference pose...")

        pose = self._read_pose()
        if pose is None:
            self._raise_last_error("Failed to capture calibration pose")
            return

        self._zero_pos, self._zero_rot = pose
        self._enabled = False
        self._has_reference = True
        logger.info("Captured new Omega.3 neutral pose.")

    @property
    def is_calibrated(self) -> bool:
        # Calibration lives in-memory for this teleoperator.
        return self._has_reference

    def configure(self) -> None:
        # Nothing to configure beyond disabling force output at the moment.
        if self.is_connected:
            dhd.enableForce(False, self._device_id)

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        pose = self._read_pose()
        if pose is None:
            return {}

        pos, rot = pose
        button_mask = self._read_button_mask()
        enabled = self._read_enabled(button_mask)

        if enabled and not self._enabled and self.config.recenter_on_enable:
            # Rising edge -> capture new baseline without user interaction.
            self._zero_pos = pos.copy()
            self._zero_rot = rot
            self._has_reference = True

        delta_pos = self._zero_rot.inv().apply(pos - self._zero_pos)
        delta_rot = self._zero_rot.inv() * rot
        rotvec = delta_rot.as_rotvec()

        delta_pos = self._apply_deadband(delta_pos, self.config.translation_deadband_m)
        rotvec = self._apply_deadband(rotvec, self.config.rotation_deadband_rad)

        scaled_pos = delta_pos * float(self.config.translation_scale)
        scaled_rot = rotvec * float(self.config.rotation_scale)

        if not enabled:
            scaled_pos[:] = 0.0
            scaled_rot[:] = 0.0

        gripper_vel = self._compute_gripper_velocity(button_mask)

        action = {
            "enabled": enabled,
            "target_x": float(scaled_pos[0]),
            "target_y": float(scaled_pos[1]),
            "target_z": float(scaled_pos[2]),
            "target_wx": float(scaled_rot[0]),
            "target_wy": float(scaled_rot[1]),
            "target_wz": float(scaled_rot[2]),
            "gripper_vel": gripper_vel,
            "omega3.button_mask": button_mask,
        }

        self._enabled = enabled
        self._last_button_mask = button_mask
        return action

    def _apply_deadband(self, values: np.ndarray, threshold: float) -> np.ndarray:
        if threshold <= 0:
            return values
        mask = np.abs(values) < threshold
        values = values.copy()
        values[mask] = 0.0
        return values

    def _read_enabled(self, button_mask: int) -> bool:
        button_idx = self.config.enable_button_index
        if button_idx is None:
            return True
        if button_idx < 0 or button_idx >= 32:
            raise ValueError("enable_button_index must be between 0 and 31.")
        return bool(button_mask & (1 << button_idx))

    def _compute_gripper_velocity(self, button_mask: int) -> float:
        speed = float(self.config.gripper_speed)
        vel = 0.0
        open_idx = self.config.gripper_open_button_index
        close_idx = self.config.gripper_close_button_index

        if open_idx is not None and open_idx >= 0 and (button_mask & (1 << open_idx)):
            vel += speed
        if close_idx is not None and close_idx >= 0 and (button_mask & (1 << close_idx)):
            vel -= speed

        return vel

    def _read_button_mask(self) -> int:
        if not self.is_connected:
            return 0
        mask = dhd.getButtonMask(self._device_id)
        if mask < 0:
            self._log_last_error("Failed to read Omega.3 button mask")
            return self._last_button_mask
        return mask

    def _read_pose(self) -> tuple[np.ndarray, Rotation] | None:
        if not self.is_connected:
            return None

        pos_buf = [0.0, 0.0, 0.0]
        rot_buf = [[0.0, 0.0, 0.0] for _ in range(3)]
        status = dhd.getPositionAndOrientationFrame(pos_buf, rot_buf, self._device_id)

        if status < 0:
            if ErrorNum is not None and dhd.errorGetLast() == ErrorNum.NOT_AVAILABLE:
                # Fallback to position-only devices.
                err = dhd.getPosition(pos_buf, self._device_id)
                if err < 0:
                    self._log_last_error("Failed to read Omega.3 position")
                    return None
                return np.array(pos_buf, dtype=float), self._zero_rot

            self._log_last_error("Failed to read Omega.3 pose")
            return None

        pos = np.array(pos_buf, dtype=float)
        rot_matrix = np.array(rot_buf, dtype=float)
        return pos, Rotation.from_matrix(rot_matrix)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        raise NotImplementedError("Omega.3 force feedback is not implemented yet.")

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        dhd.close(self._device_id)
        logger.info("Disconnected Omega.3 (id=%s)", self._device_id)
        self._device_id = None
        self._enabled = False

    def _raise_last_error(self, message: str) -> None:
        err = self._format_last_error()
        raise RuntimeError(f"{message}: {err}")

    def _log_last_error(self, message: str) -> None:
        err = self._format_last_error()
        logger.warning("%s: %s", message, err)

    def _format_last_error(self) -> str:
        if dhd is None:
            return "Force Dimension SDK unavailable"
        try:
            error_name = dhd.errorGetLastStr()
        except Exception:  # pragma: no cover - defensive
            return "Unknown SDK error"
        return error_name
