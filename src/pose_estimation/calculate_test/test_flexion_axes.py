from __future__ import annotations

import numpy as np

from pose_estimation.calculate_joint_angles import _joint_frames, _named_points
from pose_estimation.keypoint_json import load_keypoints_json


def test_elbow_flexion_axis_is_normal_to_bone_plane() -> None:
    _, _, keypoints = load_keypoints_json("data_logs/posture/squat.json")
    frames = _joint_frames(_named_points(keypoints))
    points = _named_points(keypoints)

    for joint, parent, child in (
        ("L_elbow", "L_shoulder", "L_wrist"),
        ("R_elbow", "R_shoulder", "R_wrist"),
        ("L_knee", "L_hip", "L_ankle"),
        ("R_knee", "R_hip", "R_ankle"),
    ):
        outgoing = points[child] - points[joint]
        if joint.endswith(("elbow", "wrist")):
            incoming = points[parent] - points[joint]
            expected = np.cross(incoming, outgoing)
        else:
            reference = points["righthip"] - points["lefthip"]
            expected = np.cross(reference, outgoing)
        expected /= np.linalg.norm(expected)
        axis_index = 0 if joint.endswith(("elbow", "wrist")) else 2
        np.testing.assert_allclose(
            abs(float(np.dot(frames[joint][:, axis_index], expected))),
            1.0,
            atol=1e-10,
        )
