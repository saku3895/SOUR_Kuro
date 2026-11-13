"""
フェーズ2：ローカル座標系および関節角度の計算検証 ユニットテスト。

検証内容:
1. 各関節のローカル座標系（X, Y, Z基底）の完全な直交性（内積が 1e-10 以下でゼロ）
2. ローカル座標系が正規直交基底（各軸のノルム = 1.0）かつ右手系（det = 1.0）であること
3. 各関節のローカルX軸と実際のボーン方向ベクトルとのコサイン類似度（完全一致 = 1.0）
4. 回転行列からオイラー角（Z-X-Y順）への正確な変換（T字ポーズ屈曲関節 = 0.0° 等）
5. ジンバルロック特異点や極端な姿勢における数値安定性（NaN/Infの非発生）
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pose_estimation.calculate_joint_angles import (
    _CHILDREN,
    _joint_frames,
    _named_points,
    calculate_joint_angles,
)
from pose_estimation.motion_config import JOINT_INDEX, JOINT_NAMES, PARENTS


# ---------------------------------------------------------------------------
# 1. ローカル座標系の直交性・正規性・右手系検証
# ---------------------------------------------------------------------------

class TestLocalFrameOrthonormality:
    """各関節のローカル座標系（回転行列）が正規直交基底を形成しているかの検証。"""

    @pytest.mark.parametrize("fixture_name", ["tpose_coco", "squat_coco", "raised_left_arm_coco"])
    def test_local_axes_are_strictly_orthogonal(self, request, fixture_name):
        """
        すべての関節のローカル基底ベクトル (X, Y, Z) の内積が
        機械精度（1e-10 以下）でゼロであること（完全な直交性）。
        """
        coco_frame = request.getfixturevalue(fixture_name)
        points = _named_points(coco_frame)
        frames = _joint_frames(points)

        for joint_name, frame in frames.items():
            x_axis = frame[:, 0]
            y_axis = frame[:, 1]
            z_axis = frame[:, 2]

            dot_xy = float(np.dot(x_axis, y_axis))
            dot_yz = float(np.dot(y_axis, z_axis))
            dot_zx = float(np.dot(z_axis, x_axis))

            assert abs(dot_xy) < 1e-10, (
                f"関節 [{joint_name}] の X軸とY軸が直交していません (dot={dot_xy:.2e})"
            )
            assert abs(dot_yz) < 1e-10, (
                f"関節 [{joint_name}] の Y軸とZ軸が直交していません (dot={dot_yz:.2e})"
            )
            assert abs(dot_zx) < 1e-10, (
                f"関節 [{joint_name}] の Z軸とX軸が直交していません (dot={dot_zx:.2e})"
            )

    @pytest.mark.parametrize("fixture_name", ["tpose_coco", "squat_coco", "raised_left_arm_coco"])
    def test_local_axes_are_unit_vectors(self, request, fixture_name):
        """各基底ベクトルのノルム（長さ）が厳密に 1.0 であること。"""
        coco_frame = request.getfixturevalue(fixture_name)
        points = _named_points(coco_frame)
        frames = _joint_frames(points)

        for joint_name, frame in frames.items():
            for axis_idx, axis_name in enumerate(["X", "Y", "Z"]):
                norm_val = float(np.linalg.norm(frame[:, axis_idx]))
                np.testing.assert_allclose(
                    norm_val,
                    1.0,
                    atol=1e-10,
                    err_msg=f"関節 [{joint_name}] の {axis_name}軸ベクトルのノルムが 1.0 ではありません。",
                )

    @pytest.mark.parametrize("fixture_name", ["tpose_coco", "squat_coco", "raised_left_arm_coco"])
    def test_local_frame_is_right_handed(self, request, fixture_name):
        """
        回転行列の行列式が +1.0 であること。
        外積の掛ける順序ミスによる鏡像反転（左手系バグ）がないことを保証。
        """
        coco_frame = request.getfixturevalue(fixture_name)
        points = _named_points(coco_frame)
        frames = _joint_frames(points)

        for joint_name, frame in frames.items():
            det = float(np.linalg.det(frame))
            np.testing.assert_allclose(
                det,
                1.0,
                atol=1e-10,
                err_msg=f"関節 [{joint_name}] の回転行列の行列式が +1.0 ではありません (det={det:.4f})。鏡像反転の疑いがあります。",
            )


# ---------------------------------------------------------------------------
# 2. ボーン方向ベクトルとローカル軸のコサイン類似度検証
# ---------------------------------------------------------------------------

class TestBoneDirectionCosineSimilarity:
    """各関節のローカル軸がボーンの幾何学的方向と完全に一致しているかの検証。"""

    @pytest.mark.parametrize("fixture_name", ["tpose_coco", "squat_coco", "raised_left_arm_coco"])
    def test_local_x_axis_matches_bone_direction_cosine_similarity(self, request, fixture_name):
        """
        各関節のローカルX軸が、定義されたボーン方向ベクトルとコサイン類似度 1.0
        （空間上で完全に同じ方向）を向いていることを検証。
        """
        coco_frame = request.getfixturevalue(fixture_name)
        points = _named_points(coco_frame)
        frames = _joint_frames(points)

        for joint in JOINT_NAMES[1:]:
            parent = PARENTS[joint]
            if joint == "head":
                expected_direction = points["nose"] - points["head"]
            else:
                child = _CHILDREN[joint]
                expected_direction = (
                    points[child] - points[joint] if child else points[joint] - points[parent]
                )

            # 期待される単位方向ベクトル
            expected_unit = expected_direction / np.linalg.norm(expected_direction)
            # 関節フレームの X 軸（1列目）
            actual_x_axis = frames[joint][:, 0]

            # コサイン類似度（内積）
            cosine_sim = float(np.dot(actual_x_axis, expected_unit))

            np.testing.assert_allclose(
                cosine_sim,
                1.0,
                atol=1e-10,
                err_msg=f"関節 [{joint}] のローカルX軸とボーン方向ベクトルのコサイン類似度が 1.0 ではありません (sim={cosine_sim:.6f})",
            )

    def test_body_frame_x_axis_matches_shoulder_span_cosine_similarity(self, tpose_coco):
        """
        腰（hips / body_frame）のローカルX軸が、
        左肩から右肩への方向ベクトルとコサイン類似度 1.0 であること。
        """
        points = _named_points(tpose_coco)
        frames = _joint_frames(points)

        shoulder_span = points["rightshoulder"] - points["leftshoulder"]
        shoulder_unit = shoulder_span / np.linalg.norm(shoulder_span)
        hips_x_axis = frames["hips"][:, 0]

        cosine_sim = float(np.dot(hips_x_axis, shoulder_unit))
        np.testing.assert_allclose(
            cosine_sim,
            1.0,
            atol=1e-10,
            err_msg=f"hips の X軸と肩スパン方向のコサイン類似度が 1.0 ではありません (sim={cosine_sim:.6f})",
        )


# ---------------------------------------------------------------------------
# 3. 回転行列からオイラー角への変換検証
# ---------------------------------------------------------------------------

class TestEulerAngleConversion:
    """回転行列から Z-X-Y オイラー角への変換正確性の検証。"""

    def test_tpose_flexion_joint_angles_are_zero(self, tpose_coco):
        """
        標準 T 字ポーズにおいて、屈曲関節（肘・膝・手首・足首など）の
        オイラー角が 0.0° であること。
        """
        angles = calculate_joint_angles(tpose_coco[np.newaxis, ...])[0]

        # 屈曲関節インデックス: 肘、手首、膝、足首
        flexion_joint_names = [
            "L_elbow", "R_elbow",
            "L_wrist", "R_wrist",
            "L_knee", "R_knee",
            "L_ankle", "R_ankle",
        ]

        for name in flexion_joint_names:
            idx = JOINT_INDEX[name]
            joint_angles = angles[idx]
            np.testing.assert_allclose(
                joint_angles,
                0.0,
                atol=1e-5,
                err_msg=f"T字ポーズにおいて関節 [{name}] の角度が 0.0° になっていません: {joint_angles}",
            )

    def test_rotation_matrix_roundtrip_zxy(self, tpose_coco):
        """
        算出されたオイラー角から Z-X-Y 順序で回転行列を再構成した際、
        親相対回転行列（relative matrix）と機械精度で一致すること。
        """
        points = _named_points(tpose_coco)
        frames = _joint_frames(points)
        angles = calculate_joint_angles(tpose_coco[np.newaxis, ...])[0]

        for joint_index, joint in enumerate(JOINT_NAMES):
            parent = PARENTS[joint]
            expected_relative = frames[joint] if parent is None else frames[parent].T @ frames[joint]

            # 算出された角度から回転行列を再構成（Z-X-Y順、度数法）
            reconstructed_rot = Rotation.from_euler("zxy", angles[joint_index], degrees=True).as_matrix()

            np.testing.assert_allclose(
                reconstructed_rot,
                expected_relative,
                atol=1e-7,
                err_msg=f"関節 [{joint}] のオイラー角再構成回転行列が親相対回転行列と一致しません。",
            )

    def test_squat_pose_detects_lower_body_motion(self, squat_coco, tpose_coco):
        """スクワット姿勢において下半身関節（膝・股関節）に屈曲角度が算出されること。"""
        squat_angles = calculate_joint_angles(squat_coco[np.newaxis, ...])[0]
        tpose_angles = calculate_joint_angles(tpose_coco[np.newaxis, ...])[0]

        for knee_name in ["L_knee", "R_knee"]:
            idx = JOINT_INDEX[knee_name]
            # 直立（Tポーズ）時と明確な角度差が生じていること
            angle_diff = np.linalg.norm(squat_angles[idx] - tpose_angles[idx])
            assert angle_diff > 10.0, (
                f"スクワット姿勢において [{knee_name}] に十分な屈曲角が検出されていません (diff={angle_diff:.2f}°)"
            )


# ---------------------------------------------------------------------------
# 4. ジンバルロックおよび特異点のエッジケース検証
# ---------------------------------------------------------------------------

class TestGimbalLockAndSingularities:
    """特異点付近の姿勢に対するオイラー角計算の安定性検証。"""

    @pytest.mark.parametrize("fixture_name", ["tpose_coco", "squat_coco", "raised_left_arm_coco"])
    def test_all_mock_postures_produce_finite_angles(self, request, fixture_name):
        """すべてのモック姿勢入力に対して、NaN や Inf が一切発生しないこと。"""
        coco_frame = request.getfixturevalue(fixture_name)
        angles = calculate_joint_angles(coco_frame[np.newaxis, ...])

        assert not np.isnan(angles).any(), f"{fixture_name} の計算結果に NaN が含まれています。"
        assert not np.isinf(angles).any(), f"{fixture_name} の計算結果に Inf が含まれています。"
        assert angles.shape == (1, len(JOINT_NAMES), 3)

    def test_gimbal_lock_pitch_singularity_safety(self, tpose_coco):
        """
        腕を真上（ピッチ90度付近）に振り上げたような特異点・ジンバルロック姿勢でも、
        オイラー角計算が破綻せず、有限な数値（finite）として安全に算出されること。
        """
        gimbal_coco = tpose_coco.copy()
        # 左肩、肘、手首を垂直上向きに配置
        left_shoulder = gimbal_coco[5]  # leftshoulder
        gimbal_coco[7] = left_shoulder + np.array([0.0, 0.27, 0.0])  # leftelbow
        gimbal_coco[9] = left_shoulder + np.array([0.0, 0.50, 0.0])  # leftwrist

        # calculate_joint_angles を実行
        angles = calculate_joint_angles(gimbal_coco[np.newaxis, ...])

        # クラッシュせず、すべての角度が有限値であること
        assert np.all(np.isfinite(angles)), "ジンバルロック姿勢において NaN または Inf が発生しました。"
