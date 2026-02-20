"""Twist division utility for smooth robot control at higher frequency than dataset FPS.

When datasets are recorded/downsampled at low FPS (e.g. 10Hz), but the robot needs smooth
control at higher frequency (e.g. 30Hz), the twist (velocity) action is divided by N so that
each sub-step covers 1/N of the original displacement. Combined with fresh robot observations
per sub-step, this produces smooth trajectories that adapt to the robot's actual position.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def divide_twist(
    action: dict[str, Any],
    n_substeps: int,
    velocity_keys: tuple[str, ...] = ("linear_vel", "angular_vel"),
) -> dict[str, Any]:
    """Divide twist velocity components by n_substeps, keeping other keys unchanged.

    Velocity keys (position/rotation displacements) are divided so each sub-step
    covers 1/N of the total displacement. Other keys (e.g. gripper.pos) are absolute
    targets and are passed through unchanged.

    Args:
        action: Twist-format action dict (e.g. linear_vel, angular_vel, gripper.pos).
        n_substeps: Number of sub-steps to divide into.
        velocity_keys: Keys whose values are displacements and should be divided.

    Returns:
        New action dict with velocity components divided by n_substeps.
    """
    divided = dict(action)
    for key in velocity_keys:
        if key in divided:
            divided[key] = np.asarray(divided[key], dtype=np.float32) / n_substeps
    return divided
