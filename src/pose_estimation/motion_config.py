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

# Absolute parent-relative angles produced by the pose-estimation coordinate
# convention for the reference T-pose. Motion-language angles are relative to
# these offsets, so the T-pose is always the zero state.
T_POSE_OFFSETS: Final[tuple[tuple[float, float, float], ...]] = (
    (0.0, 0.0, 0.0),       # hips
    (0.0, 0.0, 0.0),       # neck
    (0.0, 0.0, 0.0),       # head
    (90.0, 0.0, 0.0),      # L_shoulder
    (0.0, 0.0, 0.0),       # L_elbow
    (0.0, 0.0, 0.0),       # L_wrist
    (90.0, 0.0, 180.0),    # R_shoulder
    (0.0, 0.0, 0.0),       # R_elbow
    (0.0, 0.0, 0.0),       # R_wrist
    (180.0, 0.0, 180.0),   # L_hip
    (0.0, 0.0, 0.0),       # L_knee
    (0.0, 0.0, 0.0),       # L_ankle
    (180.0, 0.0, 180.0),   # R_hip
    (0.0, 0.0, 0.0),       # R_knee
    (0.0, 0.0, 0.0),       # R_ankle
)

# Signs convert the pose-estimation frame convention to the common
# anatomical motion convention.  The right shoulder's Pitch and Yaw axes
# mirror the left shoulder; Roll already has the desired orientation.
JOINT_ANGLE_SIGNS: Final[dict[str, tuple[float, float, float]]] = {
    "R_shoulder": (1.0, -1.0, -1.0),
    "R_elbow": (1.0, -1.0, 1.0),
}

# Entries are (joint index, Euler axis index, range name, levels, minimum, maximum).
# Euler axes are [Z, X, Y], i.e. [Roll, Pitch, Yaw]. The limits below are
# relative motion-language limits around T-pose, not absolute frame angles.
# Each quantization level is approximately 15 degrees wide.
# Elbow joints expose Pitch and Yaw, while knees expose Pitch only.
AXIS_MAPPING: Final[dict[str, tuple[int, int, str, int, float, float]]] = {
    "A": (0, 0, "range", 7, -50, 50),
    "B": (0, 1, "range", 7, -50, 50),
    "C": (0, 2, "range", 7, -50, 50),
    "D": (1, 0, "range", 7, -50, 50),
    "E": (1, 1, "range", 7, -50, 50),
    "F": (1, 2, "range", 7, -50, 50),
    "G": (3, 0, "range", 17, -90, 165),
    "H": (3, 1, "range", 19, -45, 230),
    "I": (3, 2, "range", 11, -30, 135),
    "J": (4, 1, "range", 10, -10, 140),  # 左肘のPitch
    "K": (4, 2, "range", 10, -75, 75),   # 左肘のYaw
    "L": (6, 0, "range", 17, -90, 165),
    "M": (6, 1, "range", 19, -45, 230),
    "N": (6, 2, "range", 11, -30, 135),
    "O": (7, 1, "range", 10, -10, 140),  # 右肘のPitch
    "P": (7, 2, "range", 10, -75, 75),   # 右肘のYaw
    "Q": (9, 0, "range", 4, -30, 30),
    "R": (9, 1, "range", 10, -30, 120),
    "S": (9, 2, "range", 6, -45, 45),
    "T": (10, 1, "range", 10, -10, 140),
    "U": (12, 0, "range", 4, -30, 30),
    "V": (12, 1, "range", 10, -30, 120),
    "W": (12, 2, "range", 6, -45, 45),
    "X": (13, 1, "range", 10, -10, 140),
}

AXIS_IDS: Final[tuple[str, ...]] = tuple(AXIS_MAPPING)

__all__ = [
    "AXIS_IDS",
    "AXIS_MAPPING",
    "JOINT_INDEX",
    "JOINT_ANGLE_SIGNS",
    "JOINT_NAMES",
    "PARENTS",
    "T_POSE_OFFSETS",
]