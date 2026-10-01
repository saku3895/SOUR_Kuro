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

# Entries are (joint index, Euler axis index, minimum, maximum, levels).
# Euler axes are [Z, X, Y], i.e. [Roll, Pitch, Yaw] for this project.
_ROM: Final[dict[str, tuple[float, float]]] = {
    "hips": (-45.0, 45.0),
    "neck": (-60.0, 60.0),
    "head": (-60.0, 60.0),
    "shoulder_roll": (-60.0, 60.0),
    "shoulder_pitch": (-60.0, 180.0),
    "shoulder_yaw": (-90.0, 90.0),
    "elbow_roll": (-10.0, 10.0),
    "elbow_pitch": (0.0, 150.0),
    "elbow_yaw": (-10.0, 10.0),
    "wrist": (-60.0, 60.0),
    "hip_roll": (-45.0, 45.0),
    "hip_pitch": (-30.0, 120.0),
    "hip_yaw": (-45.0, 45.0),
    "knee": (0.0, 150.0),
    "ankle": (-45.0, 30.0),
}


def _axis(joint: str, axis: int, rom_name: str, levels: int = 10) -> tuple[int, int, float, float, int]:
    minimum, maximum = _ROM[rom_name]
    return JOINT_INDEX[joint], axis, minimum, maximum, levels


AXIS_MAPPING: Final[dict[str, tuple[int, int, float, float, int]]] = {
    "A": _axis("hips", 0, "hips"),
    "B": _axis("hips", 1, "hips"),
    "C": _axis("hips", 2, "hips"),
    "D": _axis("neck", 0, "neck"),
    "E": _axis("neck", 1, "neck"),
    "F": _axis("neck", 2, "neck"),
    "G": _axis("L_shoulder", 0, "shoulder_roll"),
    "H": _axis("L_shoulder", 1, "shoulder_pitch"),
    "I": _axis("L_shoulder", 2, "shoulder_yaw"),
    "J": _axis("L_elbow", 0, "elbow_roll"),
    "K": _axis("L_elbow", 1, "elbow_pitch"),
    "L": _axis("L_elbow", 2, "elbow_yaw"),
    "M": _axis("L_wrist", 0, "wrist"),
    "N": _axis("L_wrist", 1, "wrist"),
    "O": _axis("L_wrist", 2, "wrist"),
    "P": _axis("R_shoulder", 0, "shoulder_roll"),
    "Q": _axis("R_shoulder", 1, "shoulder_pitch"),
    "R": _axis("R_shoulder", 2, "shoulder_yaw"),
    "S": _axis("R_elbow", 0, "elbow_roll"),
    "T": _axis("R_elbow", 1, "elbow_pitch"),
    "U": _axis("R_elbow", 2, "elbow_yaw"),
    "V": _axis("R_wrist", 0, "wrist"),
    "W": _axis("R_wrist", 1, "wrist"),
    "X": _axis("R_wrist", 2, "wrist"),
    "Y": _axis("L_hip", 0, "hip_roll"),
    "Z": _axis("L_hip", 1, "hip_pitch"),
    "AA": _axis("L_hip", 2, "hip_yaw"),
    "AB": _axis("L_knee", 0, "elbow_roll"),
    "AC": _axis("L_knee", 1, "knee"),
    "AD": _axis("L_knee", 2, "elbow_roll"),
    "AE": _axis("L_ankle", 0, "ankle"),
    "AF": _axis("L_ankle", 1, "ankle"),
    "AG": _axis("L_ankle", 2, "ankle"),
    "AH": _axis("R_hip", 0, "hip_roll"),
    "AI": _axis("R_hip", 1, "hip_pitch"),
    "AJ": _axis("R_hip", 2, "hip_yaw"),
    "AK": _axis("R_knee", 0, "elbow_roll"),
    "AL": _axis("R_knee", 1, "knee"),
    "AM": _axis("R_knee", 2, "elbow_roll"),
    "AN": _axis("R_ankle", 0, "ankle"),
    "AO": _axis("R_ankle", 1, "ankle"),
    "AP": _axis("R_ankle", 2, "ankle"),
}

AXIS_IDS: Final[tuple[str, ...]] = tuple(AXIS_MAPPING)

__all__ = ["AXIS_IDS", "AXIS_MAPPING", "JOINT_INDEX", "JOINT_NAMES", "PARENTS"]