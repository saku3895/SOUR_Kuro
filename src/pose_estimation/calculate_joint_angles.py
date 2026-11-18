"""Convert COCO 17-point 3-D poses to parent-relative Z-X-Y Euler angles."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

try:
    from .motion_config import JOINT_ANGLE_SIGNS, JOINT_NAMES, PARENTS, T_POSE_OFFSETS
except ImportError:
    from motion_config import JOINT_ANGLE_SIGNS, JOINT_NAMES, PARENTS, T_POSE_OFFSETS


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

_CAMERA_NEUTRAL_ROTATION = np.diag([1.0, -1.0, -1.0])


def _unit(vector: np.ndarray, label: str = "vector") -> np.ndarray:
    length = np.linalg.norm(vector)
    if length <= np.finfo(float).eps:
        raise ValueError(f"zero-length vector for {label}")
    return vector / length


def _build_frame(
    y_vec: np.ndarray,
    ref_vec: np.ndarray,
    kind: str,
    label: str,
) -> np.ndarray:
    """Build a right-handed local frame with an explicit arm/leg convention."""
    y_axis = _unit(y_vec, f"{label} bone Y")
    reference = _unit(ref_vec, f"{label} reference")
    if kind not in {"arm", "leg"}:
        raise ValueError(f"unknown frame kind: {kind}")

    # Arm: body-up x lateral-bone, Leg: body-right x vertical-bone.
    # In both cases this explicitly makes Z the front/back axis.
    z_axis = np.cross(reference, y_axis)
    if np.linalg.norm(z_axis) <= np.finfo(float).eps * 10.0:
        fallback = np.array([0.0, 0.0, 1.0])
        z_axis = fallback - np.dot(fallback, y_axis) * y_axis
    z_axis = _unit(z_axis, f"{label} lateral/front axis")
    x_axis = _unit(np.cross(y_axis, z_axis), f"{label} X")
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
    """
    腕と脚で計算ロジックおよび参照ベクトル（ref_vec）を明確に分けてローカル座標系を構築する。
    - 腕: 参照ベクトル=body_up -> X軸は「上下方向」
    - 脚: 参照ベクトル=body_lateral -> X軸は「左右方向」
    """
    # 体幹の上下ベクトル (hips -> neck)
    body_up = _unit(points["neck"] - points["hips"], "body_up")
    # 体幹の左右ベクトル (left_hip -> right_hip)
    body_lateral = _unit(points["righthip"] - points["lefthip"], "body_lateral")

    frames: dict[str, np.ndarray] = {}
    for joint in JOINT_NAMES:
        # ボーンベクトル（Y軸）の取得
        if joint == "hips":
            y_vec = points["neck"] - points["hips"]
        elif joint == "head":
            y_vec = points["head"] - points["neck"]
            if np.linalg.norm(y_vec) <= np.finfo(float).eps:
                y_vec = points["nose"] - points["head"]
        else:
            child = _CHILDREN[joint]
            parent = PARENTS[joint]
            y_vec = (
                points[child] - points[joint]
                if child
                else points[joint] - points[parent]
            )

        # 腕系関節と脚・体幹系関節で計算を分岐
        if joint in ("L_shoulder", "R_shoulder"):
            side = "L" if joint.startswith("L_") else "R"
            elbow = points[f"{side}_elbow"]
            wrist = points[f"{side}_wrist"]
            upper_arm = elbow - points[joint]
            forearm = wrist - elbow
            shoulder_frame = _build_arm_shoulder_frame(
                upper_arm,
                forearm,
                body_up,
                body_lateral,
                joint,
            )
            elbow_frame, _ = _build_arm_elbow_frame(
                shoulder_frame,
                forearm,
                joint,
            )
            frames[joint] = shoulder_frame
            frames[f"{side}_elbow"] = elbow_frame
            frames[f"{side}_wrist"] = elbow_frame
            continue
        if joint in ("L_hip", "R_hip"):
            side = "L" if joint.startswith("L_") else "R"
            knee = points[f"{side}_knee"]
            ankle = points[f"{side}_ankle"]
            thigh = knee - points[joint]
            calf = ankle - knee
            hip_frame = _build_leg_hip_frame(
                thigh,
                calf,
                body_lateral,
                joint,
            )
            knee_frame, _ = _build_hinge_child_frame(
                hip_frame,
                calf,
                joint,
            )
            frames[joint] = hip_frame
            frames[f"{side}_knee"] = knee_frame
            frames[f"{side}_ankle"] = knee_frame
            continue
        if joint in (
            "L_elbow", "L_wrist", "R_elbow", "R_wrist",
            "L_knee", "L_ankle", "R_knee", "R_ankle",
        ):
            continue
        frames[joint] = _build_frame(y_vec, body_lateral, "leg", joint)

    return frames


def _build_arm_shoulder_frame(
    upper_arm: np.ndarray,
    forearm: np.ndarray,
    body_up: np.ndarray,
    body_lateral: np.ndarray,
    label: str,
) -> np.ndarray:
    """Build a shoulder frame whose X axis is the elbow hinge axis."""
    y_axis = _unit(upper_arm, f"{label} upper-arm Y")
    reference_x = body_up - np.dot(body_up, y_axis) * y_axis
    if np.linalg.norm(reference_x) <= np.finfo(float).eps * 10.0:
        reference_x = body_lateral - np.dot(body_lateral, y_axis) * y_axis
    reference_x = _unit(reference_x, f"{label} shoulder X reference")

    hinge_x = np.cross(upper_arm, forearm)
    if np.linalg.norm(hinge_x) <= np.finfo(float).eps * 10.0:
        x_axis = reference_x
    else:
        x_axis = _unit(hinge_x, f"{label} elbow hinge X")
        if np.dot(x_axis, reference_x) < 0.0:
            x_axis = -x_axis

    z_axis = _unit(np.cross(x_axis, y_axis), f"{label} shoulder Z")
    return np.column_stack((x_axis, y_axis, z_axis))


def _build_arm_elbow_frame(
    shoulder_frame: np.ndarray,
    forearm: np.ndarray,
    label: str,
) -> tuple[np.ndarray, float]:
    """Build the elbow frame using only an X-axis hinge rotation."""
    return _build_hinge_child_frame(shoulder_frame, forearm, label)


def _build_leg_hip_frame(
    thigh: np.ndarray,
    calf: np.ndarray,
    body_lateral: np.ndarray,
    label: str,
) -> np.ndarray:
    """Build a hip frame whose X axis is the knee hinge axis."""
    y_axis = _unit(thigh, f"{label} thigh Y")
    reference_x = body_lateral - np.dot(body_lateral, y_axis) * y_axis
    reference_x = _unit(reference_x, f"{label} hip X reference")

    hinge_x = np.cross(thigh, calf)
    if np.linalg.norm(hinge_x) <= np.finfo(float).eps * 10.0:
        x_axis = reference_x
    else:
        x_axis = _unit(hinge_x, f"{label} knee hinge X")
        if np.dot(x_axis, reference_x) < 0.0:
            x_axis = -x_axis

    z_axis = _unit(np.cross(x_axis, y_axis), f"{label} hip Z")
    return np.column_stack((x_axis, y_axis, z_axis))


def _build_hinge_child_frame(
    parent_frame: np.ndarray,
    child_segment: np.ndarray,
    label: str,
) -> tuple[np.ndarray, float]:
    """Build a child frame from a parent Y/Z plane using only X-axis rotation."""
    child_axis = _unit(child_segment, f"{label} child Y")
    y_axis = parent_frame[:, 1]
    z_axis = parent_frame[:, 2]
    angle = float(np.arctan2(np.dot(child_axis, z_axis), np.dot(child_axis, y_axis)))
    rotation = Rotation.from_euler("x", angle).as_matrix()
    return parent_frame @ rotation, angle


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
            relative = (
                _CAMERA_NEUTRAL_ROTATION.T @ frames[joint]
                if parent is None
                else frames[parent].T @ frames[joint]
            )
            joint_angles = Rotation.from_matrix(relative).as_euler("zxy", degrees=True)
            if joint in JOINT_ANGLE_SIGNS:
                offsets = np.asarray(T_POSE_OFFSETS[joint_index], dtype=float)
                relative = (joint_angles - offsets + 180.0) % 360.0 - 180.0
                joint_angles = relative * JOINT_ANGLE_SIGNS[joint] + offsets
            angles[frame_index, joint_index] = joint_angles
    return angles


__all__ = ["calculate_joint_angles"]
