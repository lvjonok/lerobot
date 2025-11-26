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

"""Schunk gripper control via Universal Robots RTDE IO interfaces."""

from __future__ import annotations

import time
from typing import Any, Optional, cast

__all__ = ["SchunkGripperController"]


class SchunkGripperController:
    """Control helper for the discrete Schunk pneumatic gripper protocol."""

    def __init__(
        self,
        robot_ip: str,
        *,
        ready_channel: int = 0,
        open_channel: int = 1,
        close_channel: int = 2,
        settle_time: float = 0.05,
        pulse_duration: float = 0.5,
        io_interface: Optional[Any] = None,
        receive_interface: Optional[Any] = None,
        init_ready: bool = True,
    ) -> None:
        if io_interface is None:
            try:
                import rtde_io  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "RTDE IO interface not available. Install lerobot[ur] to control the Schunk gripper."
                ) from exc

            io_interface = rtde_io.RTDEIOInterface(robot_ip)

        if receive_interface is None:
            try:
                import rtde_receive  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "RTDE receive interface not available. Install lerobot[ur] to control the Schunk gripper."
                ) from exc

            receive_interface = rtde_receive.RTDEReceiveInterface(robot_ip)

        self._io = cast(Any, io_interface)
        self._receive = cast(Any, receive_interface)
        self._ready_channel = ready_channel
        self._open_channel = open_channel
        self._close_channel = close_channel
        self._settle_time = settle_time
        self._pulse_duration = pulse_duration
        self._last_command_open: Optional[bool] = None

        if init_ready:
            self._io.setStandardDigitalOut(self._ready_channel, True)
            if self._settle_time > 0:
                time.sleep(self._settle_time)

    def open(self, *, wait: Optional[float] = None) -> None:
        self._command(open_gripper=True, wait=wait)

    def close(self, *, wait: Optional[float] = None) -> None:
        self._command(open_gripper=False, wait=wait)

    def toggle(self, *, wait: Optional[float] = None) -> bool:
        current = self.is_open()
        if current is None:
            target_open = (
                not self._last_command_open if self._last_command_open is not None else True
            )
        else:
            target_open = not current
        self._command(open_gripper=target_open, wait=wait)
        return bool(target_open)

    def is_open(self) -> Optional[bool]:
        open_state = self._receive.getDigitalOutState(self._open_channel)
        if open_state is not None and bool(open_state):
            return True

        close_state = self._receive.getDigitalOutState(self._close_channel)
        if close_state is not None and bool(close_state):
            return False

        return self._last_command_open

    def set_analog_current(self, channel: int, ratio: float) -> None:
        self._io.setAnalogOutputCurrent(channel, ratio)

    def _command(self, *, open_gripper: bool, wait: Optional[float]) -> None:
        wait = self._settle_time if wait is None else wait

        channel = self._open_channel if open_gripper else self._close_channel
        second_channel = self._close_channel if open_gripper else self._open_channel
        self._io.setStandardDigitalOut(channel, True)
        self._io.setStandardDigitalOut(second_channel, False)

        if self._pulse_duration > 0:
            time.sleep(self._pulse_duration)
            self._io.setStandardDigitalOut(channel, False)

        if wait > 0:
            time.sleep(wait)

        self._last_command_open = open_gripper
