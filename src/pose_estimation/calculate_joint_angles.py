"""Convert COCO 17-point 3-D poses to parent-relative Z-X-Y Euler angles."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

try:
    from .motion_config import JOINT_NAMES, PARENTS
except ImportError:
    from motion_config import JOINT_NAMES, PARENTS


COCO_INDEX = {
    "nose": 0, "lefteye": 1, "righteye": 2, "leftear": 3, "rightear": 4,
    "leftshoulder": 5, "rightshoulder": 6, "leftelbow": 7, "rightelbow": 8,
    "leftwrist": 9, "rightwrist": 10, "lefthip": 11, "righthip": 12,
    "leftknee": 13, "rightknee": 14, "leftankle": 15, "rightankle": 16,
}

_CHILDREN = {
    "neck": "head", "head": None, "L_shoulder": "L_elbow", "L_elbow": "L_wrist",
    "L_wrist": None, "R_shoulder": "R_elbow", "R_elbow": "R_wrist", "R_wrist": None,
    "L_hip": "L_knee", "L_knee": "L_ankle", "L_ankle": None,
    "R_hip": "R_knee", "R_knee": "R_ankle", "R_ankle": None,
}


def _unit(vector: np.ndarray, label: str) -> np.ndarray:
    length = np.linalg.norm(vector)
    if length <= np.finfo(float).eps:
        raise ValueError(f"zero-length vector for {label}")
    return vector / length


def _frame_from_direction(direction: np.ndarray, reference: np.ndarray, label: str) -> np.ndarray:
    x_axis = _unit(direction, label)
    y_axis = reference - np.dot(reference, x_axis) * x_axis
    if np.linalg.norm(y_axis) <= np.finfo(float).eps:
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(fallback, x_axis)) > 0.9:
            fallback = np.array([0.0, 0.0, 1.0])
        y_axis = fallback - np.dot(fallback, x_axis) * x_axis
    y_axis = _unit(y_axis, f"{label} reference")
    z_axis = _unit(np.cross(x_axis, y_axis), f"{label} cross product")
    y_axis = _unit(np.cross(z_axis, x_axis), f"{label} orthogonal axis")
    return np.column_stack((x_axis, y_axis, z_axis))


def _named_points(frame: np.ndarray) -> dict[str, np.ndarray]:
    points = {name: frame[index] for name, index in COCO_INDEX.items()}
    points["hips"] = (points["lefthip"] + points["righthip"]) / 2.0
    points["neck"] = (points["leftshoulder"] + points["rightshoulder"]) / 2.0
    points["head"] = np.mean(
        [points["nose"], points["lefteye"], points["righteye"], points["leftear"], points["rightear"]],
        axis=0,
    )
    for side in ("L", "R"):
        source = "left" if side == "L" else "right"
        for joint in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
            points[f"{side}_{joint}"] = points[f"{source}{joint}"]
    return points


def _joint_frames(points: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    body_up = points["neck"] - points["hips"]
    body_frame = _frame_from_direction(
        points["rightshoulder"] - points["leftshoulder"], body_up, "body",
    )
    frames = {"hips": body_frame}
    for joint in JOINT_NAMES[1:]:
        parent = PARENTS[joint]
        if joint == "head":
            direction = points["nose"] - points["head"]
            reference = points["rightear"] - points["leftear"]
        else:
            child = _CHILDREN[joint]
            direction = points[child] - points[joint] if child else points[joint] - points[parent]
            reference = body_frame[:, 1]
        frames[joint] = _frame_from_direction(direction, reference, joint)
    return frames


def calculate_joint_angles(keypoints_3d: np.ndarray) -> np.ndarray:
    """Return parent-relative ``(frames, 15, 3)`` Z-X-Y angles in degrees."""
    if not isinstance(keypoints_3d, np.ndarray):
        raise TypeError("keypoints_3d must be a NumPy array")
    if keypoints_3d.ndim != 3 or keypoints_3d.shape[1:] != (17, 3):
        raise ValueError("keypoints_3d must have shape (frames, 17, 3)")
    if keypoints_3d.shape[0] == 0 or not np.issubdtype(keypoints_3d.dtype, np.number):
        raise ValueError("keypoints_3d must contain numeric frames")
    if not np.all(np.isfinite(keypoints_3d)):
        raise ValueError("keypoints_3d must contain only finite values")

    angles = np.empty((keypoints_3d.shape[0], len(JOINT_NAMES), 3), dtype=float)
    for frame_index, frame in enumerate(keypoints_3d.astype(float, copy=False)):
        frames = _joint_frames(_named_points(frame))
        for joint_index, joint in enumerate(JOINT_NAMES):
            parent = PARENTS[joint]
            relative = frames[joint] if parent is None else frames[parent].T @ frames[joint]
            angles[frame_index, joint_index] = Rotation.from_matrix(relative).as_euler("zxy", degrees=True)
    return angles


__all__ = ["calculate_joint_angles"]