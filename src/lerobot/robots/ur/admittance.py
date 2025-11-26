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

"""Cartesian admittance utilities for Universal Robots arms."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from lerobot.utils.rotation import Rotation

logger = logging.getLogger(__name__)


def _ensure_matrix(name: str, values: np.ndarray) -> np.ndarray:
    """Validate and reshape controller parameter matrices."""

    arr = np.asarray(values, dtype=float)
    if arr.size == 6:
        arr = np.diag(arr)
    arr = arr.reshape(6, 6)
    if np.linalg.matrix_rank(arr) < 6:
        raise ValueError(f"{name} matrix must be full rank.")
    return arr


def pose_quat_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert a position + quaternion pose into a homogeneous transform."""

    pose = np.asarray(pose, dtype=float).flatten()
    if pose.shape[0] != 7:
        raise ValueError("Pose must contain 7 values: [x, y, z, qx, qy, qz, qw].")

    translation = pose[:3]
    rotation = Rotation.from_quat(pose[3:])
    transform = np.eye(4)
    transform[:3, :3] = rotation.as_matrix()
    transform[:3, 3] = translation
    return transform


def tcp_pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert a UR TCP pose [x, y, z, rx, ry, rz] to a matrix."""

    pose = np.asarray(pose, dtype=float).flatten()
    if pose.shape[0] != 6:
        raise ValueError("TCP pose must contain 6 values: [x, y, z, rx, ry, rz].")

    translation = pose[:3]
    rotvec = pose[3:]
    rotation = Rotation.from_rotvec(rotvec)
    transform = np.eye(4)
    transform[:3, :3] = rotation.as_matrix()
    transform[:3, 3] = translation
    return transform


def matrix_to_tcp_pose(transform: np.ndarray) -> np.ndarray:
    """Convert a homogeneous transform into UR TCP pose representation."""

    if transform.shape != (4, 4):
        raise ValueError("Transform must be a 4x4 matrix.")

    translation = transform[:3, 3]
    rotation = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
    return np.concatenate([translation, rotation])


def matrix_to_pose_quat(transform: np.ndarray) -> np.ndarray:
    """Convert a homogeneous transform to position + quaternion representation."""

    if transform.shape != (4, 4):
        raise ValueError("Transform must be a 4x4 matrix.")

    translation = transform[:3, 3]
    quat = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return np.concatenate([translation, quat])


@dataclass
class CartesianAdmittanceConfig:
    """Configuration for the Cartesian admittance controller."""

    mass: np.ndarray
    damping: np.ndarray
    stiffness: np.ndarray
    frequency: float

    def __post_init__(self) -> None:
        if self.frequency <= 0.0:
            raise ValueError("Frequency must be positive.")
        self.mass = _ensure_matrix("Mass", self.mass)
        self.damping = _ensure_matrix("Damping", self.damping)
        self.stiffness = _ensure_matrix("Stiffness", self.stiffness)


class CartesianAdmittance:
    """Discrete-time Cartesian admittance controller."""

    def __init__(self, config: CartesianAdmittanceConfig) -> None:
        self._config = config
        self._dt = 1.0 / config.frequency
        self._mass_inv = np.linalg.inv(config.mass)
        self._velocity = np.zeros(6, dtype=float)
        self._pose = np.eye(4)
        self._initialised = False

    def reset(self, pose: np.ndarray) -> None:
        """Reset the virtual mass state to the provided pose."""

        self._pose = pose.copy()
        self._velocity[:] = 0.0
        self._initialised = True

    @staticmethod
    def _pose_error(pose: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Compute 6D pose error between the current and target transforms."""

        pos_error = pose[:3, 3] - target[:3, 3]
        rot_error_matrix = target[:3, :3].T @ pose[:3, :3]
        rot_error = Rotation.from_matrix(rot_error_matrix).as_rotvec()
        return np.concatenate([pos_error, rot_error])

    def compute(self, current_pose: np.ndarray, target_pose: np.ndarray, wrench: np.ndarray) -> np.ndarray:
        """Update virtual pose using the measured wrench."""

        if not self._initialised:
            self.reset(current_pose)

        wrench = np.asarray(wrench, dtype=float).flatten()
        if wrench.shape[0] != 6:
            raise ValueError("Wrench must contain six elements.")

        error = self._pose_error(self._pose, target_pose)
        acceleration = self._mass_inv @ (
            wrench - self._config.damping @ self._velocity - self._config.stiffness @ error
        )

        self._velocity += acceleration * self._dt
        # Limit maximum velocities for stability.
        self._velocity[:3] = np.clip(self._velocity[:3], -0.75, 0.75)
        self._velocity[3:] = np.clip(self._velocity[3:], -2.5, 2.5)

        # Integrate translational and rotational components separately.
        self._pose[:3, 3] += self._velocity[:3] * self._dt

        delta_rotvec = self._velocity[3:] * self._dt
        if np.linalg.norm(delta_rotvec) > 1e-12:
            delta_rotation = Rotation.from_rotvec(delta_rotvec).as_matrix()
            self._pose[:3, :3] = self._pose[:3, :3] @ delta_rotation

        return self._pose.copy()


class HighFrequencyAdmittanceLoop:
    """Background loop that applies Cartesian admittance at high frequency."""

    def __init__(
        self,
        rtde_control_iface,
        rtde_receive_iface,
        config: CartesianAdmittanceConfig,
        *,
        velocity: float = 0.25,
        acceleration: float = 1.5,
        lookahead_time: float = 0.1,
        gain: float = 400.0,
    ) -> None:
        self._control = rtde_control_iface
        self._receive = rtde_receive_iface
        self._config = config
        self._controller = CartesianAdmittance(config)
        self._velocity = float(velocity)
        self._acceleration = float(acceleration)
        self._lookahead = float(lookahead_time)
        self._gain = float(gain)
        self._dt = 1.0 / config.frequency

        self._target_lock = threading.Lock()
        self._target_pose: Optional[np.ndarray] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_target_pose(self, pose: np.ndarray) -> None:
        """Set the desired pose as position + quaternion."""

        target = pose_quat_to_matrix(pose)
        with self._target_lock:
            self._target_pose = target

    def start(self) -> None:
        if self.is_running():
            return
        self._stop_event.clear()
        # Only reset initialization if we're actually starting fresh
        if self._thread is None or not self._thread.is_alive():
            self._controller._initialised = False  # Reset on next compute
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.is_running():
            return
        self._stop_event.set()
        assert self._thread is not None
        self._thread.join(timeout=1.0)
        self._thread = None
        self._controller._initialised = False

    def _current_target(self, fallback: np.ndarray) -> np.ndarray:
        with self._target_lock:
            if self._target_pose is None:
                return fallback
            return self._target_pose.copy()

    def _run(self) -> None:
        dt = self._dt
        while not self._stop_event.is_set():
            loop_start = time.perf_counter()

            try:
                tcp_pose = np.asarray(self._receive.getActualTCPPose(), dtype=float)
                current_transform = tcp_pose_to_matrix(tcp_pose)
                wrench = np.asarray(self._receive.getActualTCPForce(), dtype=float)
            except Exception as exc:  # pragma: no cover - hardware read failure
                logger.warning("UR admittance loop read failed: %s", exc)
                time.sleep(0.01)
                continue

            target_transform = self._current_target(current_transform)

            desired_transform = self._controller.compute(current_transform, target_transform, wrench)
            desired_pose = matrix_to_tcp_pose(desired_transform)

            try:
                start_period = self._control.initPeriod()
                self._control.servoL(
                    desired_pose.tolist(),
                    self._velocity,
                    self._acceleration,
                    dt,
                    self._lookahead,
                    self._gain,
                )
                self._control.waitPeriod(start_period)
            except Exception as exc:  # pragma: no cover - hardware write failure
                logger.warning("UR admittance loop write failed: %s", exc)
                time.sleep(0.01)

            elapsed = time.perf_counter() - loop_start
            remainder = dt - elapsed
            if remainder > 0:
                time.sleep(remainder)
