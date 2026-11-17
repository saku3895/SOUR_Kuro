import numpy as np
import pytest
from pathlib import Path

from pose_estimation.calculate_joint_angles import COCO_INDEX
from pose_estimation.keypoint_json import load_keypoints_json, parse_keypoints_json


def _keypoints():
    return {
        name: [float(index), 0.0, 0.0]
        for name, index in COCO_INDEX.items()
    }


def test_parse_keypoints_json_applies_scale():
    pose_name, scale, points = parse_keypoints_json({
        "pose_name": "sample",
        "scale": 2,
        "keypoints": _keypoints(),
    })

    assert pose_name == "sample"
    assert scale == 2.0
    np.testing.assert_allclose(points[COCO_INDEX["nose"]], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(points[COCO_INDEX["leftshoulder"]], [10.0, 0.0, 0.0])


def test_parse_keypoints_json_rejects_missing_point():
    keypoints = _keypoints()
    del keypoints["nose"]

    with pytest.raises(ValueError, match="不足"):
        parse_keypoints_json({"keypoints": keypoints})


def test_parse_keypoints_json_rejects_invalid_coordinate():
    keypoints = _keypoints()
    keypoints["nose"] = [0.0, 1.0]

    with pytest.raises(ValueError, match="3要素"):
        parse_keypoints_json({"keypoints": keypoints})


def test_load_camera_keypoints_json():
    path = (
        Path(__file__).resolve().parents[3]
        / "data_logs"
        / "keypoints"
        / "t_pose-real.json"
    )
    pose_name, scale, points = load_keypoints_json(path)

    assert pose_name == "名前なし"
    assert scale == 1.0
    assert points.shape == (17, 3)
    assert np.isfinite(points).all()
