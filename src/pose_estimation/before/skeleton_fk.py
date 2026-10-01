"""Forward-kinematics reconstruction for decoded upper-body angles."""

import numpy as np

from .skeleton_definition import BONE_VECTORS, JOINT_INDEX, JOINT_NAMES


def _rotation_matrix(angles):
    """Build Rz @ Rx @ Ry from [Roll(Z), Pitch(X), Yaw(Y)] degrees."""
    roll, pitch, yaw = np.radians(angles)
    cz, sz = np.cos(roll), np.sin(roll)
    cx, sx = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    return rz @ rx @ ry


def angles_to_positions(angle_frames):
    """Reconstruct ``(frames, 6, 3)`` positions from ``(frames, 6, 3)`` angles."""
    angles = np.asarray(angle_frames, dtype=float)
    if angles.ndim != 3 or angles.shape[1:] != (len(JOINT_NAMES), 3):
        raise ValueError("angle_frames must have shape (frames, 6, 3)")
    if angles.shape[0] == 0 or not np.all(np.isfinite(angles)):
        raise ValueError("angle_frames must contain finite frames")

    positions = np.zeros((angles.shape[0], len(JOINT_NAMES), 3), dtype=float)
    for frame_index, frame_angles in enumerate(angles):
        global_rotations = {"hips": _rotation_matrix(frame_angles[JOINT_INDEX["hips"]])}
        for parent_name, child_name in BONE_VECTORS:
            parent_index = JOINT_INDEX[parent_name]
            child_index = JOINT_INDEX[child_name]
            parent_rotation = global_rotations[parent_name]
            positions[frame_index, child_index] = (
                positions[frame_index, parent_index]
                + parent_rotation @ BONE_VECTORS[(parent_name, child_name)]
            )
            global_rotations[child_name] = (
                parent_rotation @ _rotation_matrix(frame_angles[child_index])
            )
    return positions


__all__ = ["angles_to_positions"]
