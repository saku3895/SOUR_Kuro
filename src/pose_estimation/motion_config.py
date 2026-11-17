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

# Entries are (joint index, Euler axis index, range name, levels, minimum, maximum).
# Euler axes are [Z, X, Y], i.e. [Roll, Pitch, Yaw]. The limits below are
# relative motion-language limits around T-pose, not absolute frame angles.
# Elbow joints expose Pitch and Yaw, while knees expose Pitch only.
AXIS_MAPPING: Final[dict[str, tuple[int, int, str, int, float, float]]] = {
    "A": (0, 0, "range", 11, -50, 50),   # 11分割で 0° がレベル5の中央値(0°)に
    "B": (0, 1, "range", 11, -50, 50),
    "C": (0, 2, "range", 11, -50, 50),
    "D": (1, 0, "range", 11, -50, 50),
    "E": (1, 1, "range", 11, -50, 50),
    "F": (1, 2, "range", 11, -50, 50),
    "G": (3, 0, "range", 13, -90, 90),
    "H": (3, 1, "range", 11, -45, 135),
    "I": (3, 2, "range", 11, -75, 75),
    "J": (4, 1, "range", 15, -10, 140),  # 左肘のPitch
    "K": (4, 2, "range", 11, -75, 75),   # 左肘のYaw
    "L": (6, 0, "range", 13, -90, 90),
    "M": (6, 1, "range", 11, -45, 135),
    "N": (6, 2, "range", 11, -75, 75),
    "O": (7, 1, "range", 15, -10, 140),  # 右肘のPitch
    "P": (7, 2, "range", 11, -75, 75),   # 右肘のYaw
    "Q": (9, 0, "range", 9, -30, 30),
    "R": (9, 1, "range", 11, -30, 120),
    "S": (9, 2, "range", 9, -45, 45),
    "T": (10, 1, "range", 15, -10, 140), # 膝: 0° がレベル1の中央値に
    "U": (12, 0, "range", 9, -30, 30),
    "V": (12, 1, "range", 11, -30, 120),
    "W": (12, 2, "range", 9, -45, 45),
    "X": (13, 1, "range", 15, -10, 140), # 膝: 0° がレベル1の中央値に
}

AXIS_IDS: Final[tuple[str, ...]] = tuple(AXIS_MAPPING)

__all__ = [
    "AXIS_IDS",
    "AXIS_MAPPING",
    "JOINT_INDEX",
    "JOINT_NAMES",
    "PARENTS",
    "T_POSE_OFFSETS",
]