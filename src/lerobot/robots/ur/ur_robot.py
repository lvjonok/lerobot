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

import logging
import time
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.rotation import Rotation

from ..robot import Robot
from .admittance import (
    CartesianAdmittanceConfig,
    HighFrequencyAdmittanceLoop,
    matrix_to_pose_quat,
    tcp_pose_to_matrix,
)
from .config_ur import URRobotConfig

logger = logging.getLogger(__name__)


class URRobot(Robot):
    """
    Universal Robots arm controller using the RTDE interface.

    The robot exposes an SE(3) target interface (`target_*` keys) that matches the Force Dimension Omega
    teleoperator output, enabling intuitive pose control.
    """

    config_class = URRobotConfig
    name = "ur_rtde"

    def __init__(self, config: URRobotConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._rtde_control = None
        self._rtde_receive = None
        self._rtde_modules: tuple[Any, Any] | None = None
        self._connected = False
        self._admittance_loop: HighFrequencyAdmittanceLoop | None = None
        self._reference_pose: np.ndarray | None = None
        self._last_pose_command: np.ndarray | None = None
        self._last_enabled = False
        self._force_lowpass = np.zeros(6, dtype=float)
        self._gripper_closed = True

    @property
    def observation_features(self) -> dict:
        features = {
            "observation.joint_positions": {"dtype": "float32", "shape": (6,), "names": None},
            "observation.joint_velocities": {"dtype": "float32", "shape": (6,), "names": None},
            "observation.joint_currents": {"dtype": "float32", "shape": (6,), "names": None},
            "observation.target_joint_positions": {"dtype": "float32", "shape": (6,), "names": None},
            "observation.ee_position": {"dtype": "float32", "shape": (3,), "names": None},
            "observation.ee_quaternion_xyzw": {"dtype": "float32", "shape": (4,), "names": None},
            "observation.ee_velocity": {"dtype": "float32", "shape": (6,), "names": None},
            "observation.ee_force": {"dtype": "float32", "shape": (6,), "names": None},
            "observation.gripper_position": {"dtype": "float32", "shape": (1,), "names": None},
            "observation.timestamp": {"dtype": "float32", "shape": (1,), "names": None},
        }

        for cam_name, cam_cfg in self.config.cameras.items():
            features[f"observation.camera.{cam_name}"] = {
                "dtype": "uint8",
                "shape": (cam_cfg.height, cam_cfg.width, 3),
                "names": None,
            }

        return features

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
        }

    @property
    def is_connected(self) -> bool:
        return self._connected and self._rtde_control is not None and self._rtde_receive is not None

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        rtde_control_mod, rtde_receive_mod = self._import_rtde_modules()
        try:
            self._rtde_control = rtde_control_mod.RTDEControlInterface(self.config.robot_ip)
            self._rtde_receive = rtde_receive_mod.RTDEReceiveInterface(self.config.robot_ip)
        except Exception as exc:
            raise RuntimeError(f"Failed to connect to UR robot at {self.config.robot_ip}") from exc

        for cam in self.cameras.values():
            cam.connect()

        self._connected = True
        if calibrate:
            self.calibrate()
        self.configure()
        self._close_gripper()
        self._reset_joints_if_requested()
        logger.info("Connected to UR robot at %s", self.config.robot_ip)

    @property
    def is_calibrated(self) -> bool:
        # UR RTDE provides calibrated joint encoders; we only zero the FT sensor.
        return True

    def calibrate(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if hasattr(self._rtde_control, "zeroFtSensor"):
            self._rtde_control.zeroFtSensor()
        self._reference_pose = self._current_pose_quat()

    def configure(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if hasattr(self._rtde_control, "endFreedriveMode"):
            self._rtde_control.endFreedriveMode()

        if self.config.use_admittance:
            self._admittance_loop = self._make_admittance_loop()
        elif self._admittance_loop is not None:
            self._admittance_loop.stop()
            self._admittance_loop = None

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs: dict[str, Any] = {}
        recv = self._rtde_receive
        assert recv is not None

        joints = np.asarray(recv.getActualQ(), dtype=float)
        obs["observation.joint_positions"] = joints.tolist()

        try:
            obs["observation.joint_velocities"] = np.asarray(recv.getActualQd(), dtype=float).tolist()
        except Exception:
            obs["observation.joint_velocities"] = np.zeros_like(joints).tolist()

        try:
            obs["observation.joint_currents"] = np.asarray(recv.getActualCurrent(), dtype=float).tolist()
        except Exception:
            obs["observation.joint_currents"] = np.zeros_like(joints).tolist()

        try:
            obs["observation.target_joint_positions"] = np.asarray(recv.getTargetQ(), dtype=float).tolist()
        except Exception:
            obs["observation.target_joint_positions"] = obs["observation.joint_positions"]

        tcp_pose = np.asarray(recv.getActualTCPPose(), dtype=float)
        obs["observation.ee_position"] = tcp_pose[:3].tolist()
        ee_quat = Rotation.from_rotvec(tcp_pose[3:]).as_quat()
        obs["observation.ee_quaternion_xyzw"] = ee_quat.tolist()

        obs["observation.ee_velocity"] = np.asarray(recv.getActualTCPSpeed(), dtype=float).tolist()

        force = np.asarray(recv.getActualTCPForce(), dtype=float)
        alpha = float(self.config.force_lowpass_alpha)
        alpha = min(max(alpha, 0.0), 1.0)
        self._force_lowpass = (1.0 - alpha) * self._force_lowpass + alpha * force
        obs["observation.ee_force"] = self._force_lowpass.tolist()

        obs["observation.gripper_position"] = [0.0 if self._gripper_closed else 1.0]

        try:
            timestamp = float(recv.getTimestamp())
        except Exception:
            timestamp = time.time()
        obs["observation.timestamp"] = [timestamp]

        for name, cam in self.cameras.items():
            obs[f"observation.camera.{name}"] = cam.async_read()

        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        enabled = bool(action.get("enabled", True))
        delta_pos = np.array(
            [
                float(action.get("target_x", 0.0)),
                float(action.get("target_y", 0.0)),
                float(action.get("target_z", 0.0)),
            ],
            dtype=float,
        )
        delta_rot = np.array(
            [
                float(action.get("target_wx", 0.0)),
                float(action.get("target_wy", 0.0)),
                float(action.get("target_wz", 0.0)),
            ],
            dtype=float,
        )

        delta_pos = self._clamp_vector(delta_pos, self.config.max_translation_m)
        if self.config.limit_rotation_deltas:
            delta_rot = self._clamp_vector(delta_rot, self.config.max_rotation_rad)

        if not enabled:
            if self._admittance_loop is not None:
                self._admittance_loop.stop()
            self._reference_pose = None
            self._last_pose_command = None
            self._last_enabled = False
            return {
                "enabled": False,
                "target_x": float(delta_pos[0]),
                "target_y": float(delta_pos[1]),
                "target_z": float(delta_pos[2]),
                "target_wx": float(delta_rot[0]),
                "target_wy": float(delta_rot[1]),
                "target_wz": float(delta_rot[2]),
                "gripper_vel": float(action.get("gripper_vel", 0.0)),
            }

        if not self._last_enabled or self._reference_pose is None:
            self._reference_pose = self._current_pose_quat()

        assert self._reference_pose is not None
        ref_pos = self._reference_pose[:3]
        ref_rot = Rotation.from_quat(self._reference_pose[3:])
        target_rot = ref_rot * Rotation.from_rotvec(delta_rot)
        target_pose = np.concatenate([ref_pos + delta_pos, target_rot.as_quat()])
        self._last_pose_command = target_pose
        self._last_enabled = True

        self._command_pose(target_pose)

        return {
            "enabled": True,
            "target_x": float(delta_pos[0]),
            "target_y": float(delta_pos[1]),
            "target_z": float(delta_pos[2]),
            "target_wx": float(delta_rot[0]),
            "target_wy": float(delta_rot[1]),
            "target_wz": float(delta_rot[2]),
            "gripper_vel": float(action.get("gripper_vel", 0.0)),
        }

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self._admittance_loop is not None:
            self._admittance_loop.stop()
            self._admittance_loop = None

        for cam in self.cameras.values():
            cam.disconnect()

        self._shutdown_rtde()
        self._connected = False
        self._reference_pose = None
        self._last_pose_command = None
        self._last_enabled = False
        logger.info("Disconnected UR robot at %s", self.config.robot_ip)

    def _close_gripper(self) -> None:
        """Best-effort gripper close. TODO: wire actual gripper driver & teleop input."""

        self._gripper_closed = True
        logger.info("Assuming UR gripper is closed (placeholder implementation).")

    def _reset_joints_if_requested(self) -> None:
        if self.config.home_joint_positions is None:
            return

        joints = np.asarray(self.config.home_joint_positions, dtype=float).flatten()
        if joints.shape[0] != 6:
            raise ValueError("home_joint_positions must contain 6 joint values.")

        if self._rtde_control is None:
            raise DeviceNotConnectedError("Cannot reset joints before RTDE control is available.")

        logger.info("Resetting UR joints to configured home pose.")
        try:
            self._rtde_control.moveJ(
                joints.tolist(),
                float(self.config.servo_velocity),
                float(self.config.servo_acceleration),
            )
        except Exception as exc:  # pragma: no cover - hardware failure
            logger.warning("Failed to move to home joint positions: %s", exc)

    def _import_rtde_modules(self):
        if self._rtde_modules is not None:
            return self._rtde_modules

        try:
            import rtde_control  # type: ignore
            import rtde_receive  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The UR RTDE robot requires the `ur-rtde` package. Install it via `pip install lerobot[ur]`."
            ) from exc

        self._rtde_modules = (rtde_control, rtde_receive)
        return self._rtde_modules

    def _make_admittance_loop(self) -> HighFrequencyAdmittanceLoop:
        assert self._rtde_control is not None and self._rtde_receive is not None
        cfg = CartesianAdmittanceConfig(
            mass=np.asarray(self.config.admittance_mass, dtype=float),
            damping=np.asarray(self.config.admittance_damping, dtype=float),
            stiffness=np.asarray(self.config.admittance_stiffness, dtype=float),
            frequency=float(self.config.admittance_frequency_hz),
        )
        return HighFrequencyAdmittanceLoop(
            self._rtde_control,
            self._rtde_receive,
            cfg,
            velocity=self.config.servo_velocity,
            acceleration=self.config.servo_acceleration,
            lookahead_time=self.config.servo_lookahead_time,
            gain=self.config.servo_gain,
        )

    def _current_pose_quat(self) -> np.ndarray:
        assert self._rtde_receive is not None
        tcp_pose = np.asarray(self._rtde_receive.getActualTCPPose(), dtype=float)
        transform = tcp_pose_to_matrix(tcp_pose)
        return matrix_to_pose_quat(transform)

    def _command_pose(self, pose: np.ndarray) -> None:
        if self.config.use_admittance:
            if self._admittance_loop is None:
                self._admittance_loop = self._make_admittance_loop()
            self._admittance_loop.set_target_pose(pose)
            if not self._admittance_loop.is_running():
                self._admittance_loop.start()
            return

        self._servo_direct(pose)

    def _servo_direct(self, pose: np.ndarray) -> None:
        assert self._rtde_control is not None
        dt = 1.0 / max(float(self.config.servo_frequency_hz), 1.0)
        rotvec = Rotation.from_quat(pose[3:]).as_rotvec()
        tcp_pose = np.concatenate([pose[:3], rotvec])
        try:
            start_period = self._rtde_control.initPeriod()
            self._rtde_control.servoL(
                tcp_pose.tolist(),
                float(self.config.servo_velocity),
                float(self.config.servo_acceleration),
                dt,
                float(self.config.servo_lookahead_time),
                float(self.config.servo_gain),
            )
            self._rtde_control.waitPeriod(start_period)
        except Exception as exc:  # pragma: no cover - hardware failure
            logger.warning("Failed to send servo command: %s", exc)

    def _clamp_vector(self, values: np.ndarray, limit: float) -> np.ndarray:
        if limit <= 0:
            return values
        norm = float(np.linalg.norm(values))
        if norm <= limit or norm == 0.0:
            return values
        return values / norm * limit

    def _shutdown_rtde(self) -> None:
        if self._rtde_control is not None:
            for method in ("stopScript", "forceModeStop", "endFreedriveMode", "disconnect"):
                func = getattr(self._rtde_control, method, None)
                if callable(func):
                    try:
                        func()
                    except Exception:
                        pass
            self._rtde_control = None

        if self._rtde_receive is not None:
            disconnect = getattr(self._rtde_receive, "disconnect", None)
            if callable(disconnect):
                try:
                    disconnect()
                except Exception:
                    pass
            self._rtde_receive = None

    def __del__(self):
        try:
            if self.is_connected:
                self.disconnect()
        except Exception:
            # Avoid raising during interpreter shutdown.
            pass
