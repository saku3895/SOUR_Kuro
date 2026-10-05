"""Shared skeleton and anatomical range-of-motion configuration."""

from __future__ import annotations

from typing import Final


JOINT_NAMES: Final[tuple[str, ...]] = (
    "hips",
    "neck",
    "head",
    "L_shoulder",
    "L_elbow",
    "L_wrist",
    "R_shoulder",
    "R_elbow",
    "R_wrist",
    "L_hip",
    "L_knee",
    "L_ankle",
    "R_hip",
    "R_knee",
    "R_ankle",
)

PARENTS: Final[dict[str, str | None]] = {
    "hips": None,
    "neck": "hips",
    "head": "neck",
    "L_shoulder": "neck",
    "L_elbow": "L_shoulder",
    "L_wrist": "L_elbow",
    "R_shoulder": "neck",
    "R_elbow": "R_shoulder",
    "R_wrist": "R_elbow",
    "L_hip": "hips",
    "L_knee": "L_hip",
    "L_ankle": "L_knee",
    "R_hip": "hips",
    "R_knee": "R_hip",
    "R_ankle": "R_knee",
}

JOINT_INDEX: Final[dict[str, int]] = {
    name: index for index, name in enumerate(JOINT_NAMES)
}

# Entries are (joint index, Euler axis index, range name, levels, minimum, maximum).
# Euler axes are [Z, X, Y], i.e. [Roll, Pitch, Yaw] for this project.
AXIS_MAPPING: Final[dict[str, tuple[int, int, str, int, float, float]]] = {
    "A": (0, 0, "range", 10, -50, 50),
    "B": (0, 1, "range", 10, -30, 45),
    "C": (0, 2, "range", 10, -40, 40),
    "D": (1, 0, "range", 10, -50, 50),
    "E": (1, 1, "range", 10, -50, 60),
    "F": (1, 2, "range", 10, -60, 60),
    "G": (2, 0, "range", 10, -90, 90),
    "H": (2, 1, "range", 10, -30, 135),
    "I": (2, 2, "range", 10, -70, 90),
    "J": (3, 1, "range", 10, -5, 145),
    "K": (4, 0, "range", 10, -90, 90),
    "L": (4, 1, "range", 10, -30, 135),
    "M": (4, 2, "range", 10, -70, 90),
    "N": (5, 1, "range", 10, -5, 145),
    "O": (6, 0, "range", 10, -20, 45),
    "P": (6, 1, "range", 10, -15, 125),
    "Q": (6, 2, "range", 10, -45, 45),
    "R": (7, 1, "range", 10, 0, 130),
    "S": (8, 1, "range", 10, -45, 20),
    "T": (8, 0, "range", 10, -30, 20),
}

AXIS_IDS: Final[tuple[str, ...]] = tuple(AXIS_MAPPING)

__all__ = ["AXIS_IDS", "AXIS_MAPPING", "JOINT_INDEX", "JOINT_NAMES", "PARENTS"]