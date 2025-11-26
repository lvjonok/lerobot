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
from typing import Callable

logger = logging.getLogger(__name__)


class URDashboardHelper:
    """Thin wrapper around rtde_dashboard to detect robot stops and prompt operators."""

    _READY_MODES = {"running", "play", "power_on"}

    def __init__(self, robot_ip: str, prompt_fn: Callable[[str], str] | None = None) -> None:
        try:
            from rtde_dashboard import DashboardClient  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "rtde_dashboard is unavailable. Install lerobot[ur] to enable dashboard monitoring."
            ) from exc

        self._client = DashboardClient(robot_ip)
        self._client.connect()
        self._prompt = prompt_fn if prompt_fn is not None else input
        logger.info("Connected to UR dashboard at %s", robot_ip)

    def disconnect(self) -> None:
        try:
            self._client.disconnect()
        except Exception:  # pragma: no cover - best effort cleanup
            pass

    def ensure_running(self, *, on_pause: Callable[[dict[str, str]], None] | None = None) -> None:
        """Block until the dashboard reports the robot back in a running state."""

        robot_mode = self._robot_mode()
        if self._is_ready(robot_mode):
            return

        safety_mode = self._safety_mode()
        info = {"robot_mode": robot_mode, "safety_mode": safety_mode}
        if on_pause is not None:
            on_pause(info)

        logger.warning(
            "UR robot paused (robot mode=%s, safety mode=%s). Waiting for operator confirmation.",
            robot_mode,
            safety_mode,
        )

        while True:
            response = self._prompt(
                "Robot is paused/protective stopped. Resolve it on the teach pendant, then type 'y' to continue "
                "(or 'q' to abort): "
            ).strip()
            if response.lower().startswith("q"):
                raise RuntimeError("Operator aborted teleoperation after dashboard stop.")
            if response.lower().startswith("y"):
                robot_mode = self._robot_mode()
                if self._is_ready(robot_mode):
                    logger.info("UR dashboard reports robot back in %s mode. Resuming.", robot_mode)
                    return
                logger.warning("Robot still not running (mode=%s). Resolve and try again.", robot_mode)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _robot_mode(self) -> str:
        try:
            response = self._client.robotmode()
        except Exception as exc:  # pragma: no cover - dashboard network failure
            logger.warning("Failed to query robot mode: %s", exc)
            return "unknown"
        return self._extract_status(response)

    def _safety_mode(self) -> str:
        try:
            response = self._client.safetymode()
        except Exception as exc:  # pragma: no cover - dashboard network failure
            logger.warning("Failed to query safety mode: %s", exc)
            return "unknown"
        return self._extract_status(response)

    def _is_ready(self, mode: str) -> bool:
        mode_lower = mode.lower()
        return any(token in mode_lower for token in self._READY_MODES)

    @staticmethod
    def _extract_status(raw: str) -> str:
        if raw is None:
            return "unknown"
        text = str(raw).strip()
        if ":" in text:
            text = text.split(":", 1)[1]
        if "Robotmode" in text:
            text = text.replace("Robotmode", "")
        if "Safety mode" in text:
            text = text.replace("Safety mode", "")
        return text.strip() or "unknown"
