"""
フェーズ3：逆運動学（FK/IK）と座標再構築の整合性検証 ユニットテスト。

検証内容:
1. 順運動学（FK）モデルによるオイラー角からの 15 関節 3D 座標の再計算
2. 「正規化された元の 3D 座標」と「オイラー角から再構築された 3D 座標」の
   PA-MPJPE による姿勢形状完全一致検証（機械精度 1e-10 以下）
3. ルート（腰: hips）が再構築後も厳密に原点 (0, 0, 0) に固定されていることの確認
4. 特異点（ジンバルロック付近）姿勢に対する安全かつ正確な往復復元（Round-trip）
5. ロボット固定ボーン長モデルを用いた再構築における幾何学的妥当性の検証
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pose_estimation.calculate_joint_angles import (
    _named_points,
    calculate_joint_angles,
)
from pose_estimation.motion_config import JOINT_INDEX, JOINT_NAMES, PARENTS
from pose_estimation.calculate_test.conftest import pa_mpjpe, make_tpose_coco


# ---------------------------------------------------------------------------
# 順運動学（FK）ヘルパー関数とボーン定義
# ---------------------------------------------------------------------------

# ロボット標準の固定ボーン長（メートル単位）
DEFAULT_ROBOT_BONE_LENGTHS: dict[str, float] = {
    "neck": 0.30,
    "head": 0.18,
    "shoulder_span_half": 0.18,
    "hip_span_half": 0.11,
    "upper_arm": 0.27,
    "forearm": 0.23,
    "thigh": 0.40,
    "calf": 0.40,
}


def extract_bone_lengths_from_points(named_points: dict[str, np.ndarray]) -> dict[str, float]:
    """15 関節の座標辞書から各セグメントのボーン長（ユークリッド距離）を算出する。"""
    return {
        "neck": float(np.linalg.norm(named_points["neck"] - named_points["hips"])),
        "head": float(np.linalg.norm(named_points["head"] - named_points["neck"])),
        "shoulder_span_half": float(
            np.linalg.norm(named_points["rightshoulder"] - named_points["leftshoulder"]) / 2.0
        ),
        "hip_span_half": float(
            np.linalg.norm(named_points["righthip"] - named_points["lefthip"]) / 2.0
        ),
        "upper_arm_L": float(np.linalg.norm(named_points["L_elbow"] - named_points["L_shoulder"])),
        "forearm_L": float(np.linalg.norm(named_points["L_wrist"] - named_points["L_elbow"])),
        "upper_arm_R": float(np.linalg.norm(named_points["R_elbow"] - named_points["R_shoulder"])),
        "forearm_R": float(np.linalg.norm(named_points["R_wrist"] - named_points["R_elbow"])),
        "thigh_L": float(np.linalg.norm(named_points["L_knee"] - named_points["L_hip"])),
        "calf_L": float(np.linalg.norm(named_points["L_ankle"] - named_points["L_knee"])),
        "thigh_R": float(np.linalg.norm(named_points["R_knee"] - named_points["R_hip"])),
        "calf_R": float(np.linalg.norm(named_points["R_ankle"] - named_points["R_knee"])),
    }


def forward_kinematics(
    angles_deg: np.ndarray,
    bone_lengths: dict[str, float],
) -> np.ndarray:
    """
    各関節の親相対オイラー角 (15, 3) [Z, X, Y]（度数法）とボーン長から、
    順運動学（FK）により hips 原点 (0, 0, 0) の 15 関節 3D 座標 (15, 3) を再構築する。
    """
    if angles_deg.shape != (len(JOINT_NAMES), 3):
        raise ValueError(f"angles_deg must have shape ({len(JOINT_NAMES)}, 3)")

    # 1. 各関節の相対回転行列を Z-X-Y 順序で復元
    rel_rot = [
        Rotation.from_euler("zxy", angles_deg[i], degrees=True).as_matrix()
        for i in range(len(JOINT_NAMES))
    ]

    # 2. 親から累積してグローバル回転行列（局所座標系の向き）を算出
    global_frames: dict[str, np.ndarray] = {}
    for i, joint in enumerate(JOINT_NAMES):
        parent = PARENTS[joint]
        if parent is None:
            global_frames[joint] = rel_rot[i]
        else:
            global_frames[joint] = global_frames[parent] @ rel_rot[i]

    # 3. 各関節の空間座標を hips 原点から累積計算
    positions = np.zeros((len(JOINT_NAMES), 3), dtype=float)
    body_frame = global_frames["hips"]

    # hips（腰）は常に原点 (0, 0, 0)
    positions[JOINT_INDEX["hips"]] = np.array([0.0, 0.0, 0.0])

    # neck: body_frame の Y 軸（上向き）に沿って配置
    positions[JOINT_INDEX["neck"]] = (
        positions[JOINT_INDEX["hips"]] + body_frame[:, 1] * bone_lengths["neck"]
    )

    # head: neck の局所 X 軸方向に配置
    positions[JOINT_INDEX["head"]] = (
        positions[JOINT_INDEX["neck"]] + global_frames["neck"][:, 0] * bone_lengths["head"]
    )

    # 肩: neck から body_frame の X 軸方向（左右）に配置
    positions[JOINT_INDEX["L_shoulder"]] = (
        positions[JOINT_INDEX["neck"]] - body_frame[:, 0] * bone_lengths["shoulder_span_half"]
    )
    positions[JOINT_INDEX["R_shoulder"]] = (
        positions[JOINT_INDEX["neck"]] + body_frame[:, 0] * bone_lengths["shoulder_span_half"]
    )

    # 腕関節: 各関節の局所 X 軸方向に沿って子関節を配置
    upper_l = bone_lengths.get("upper_arm_L", bone_lengths.get("upper_arm", 0.27))
    fore_l = bone_lengths.get("forearm_L", bone_lengths.get("forearm", 0.23))
    upper_r = bone_lengths.get("upper_arm_R", bone_lengths.get("upper_arm", 0.27))
    fore_r = bone_lengths.get("forearm_R", bone_lengths.get("forearm", 0.23))

    positions[JOINT_INDEX["L_elbow"]] = (
        positions[JOINT_INDEX["L_shoulder"]] + global_frames["L_shoulder"][:, 0] * upper_l
    )
    positions[JOINT_INDEX["L_wrist"]] = (
        positions[JOINT_INDEX["L_elbow"]] + global_frames["L_elbow"][:, 0] * fore_l
    )
    positions[JOINT_INDEX["R_elbow"]] = (
        positions[JOINT_INDEX["R_shoulder"]] + global_frames["R_shoulder"][:, 0] * upper_r
    )
    positions[JOINT_INDEX["R_wrist"]] = (
        positions[JOINT_INDEX["R_elbow"]] + global_frames["R_elbow"][:, 0] * fore_r
    )

    # 股関節: hips から body_frame の X 軸方向（左右）に配置
    positions[JOINT_INDEX["L_hip"]] = (
        positions[JOINT_INDEX["hips"]] - body_frame[:, 0] * bone_lengths["hip_span_half"]
    )
    positions[JOINT_INDEX["R_hip"]] = (
        positions[JOINT_INDEX["hips"]] + body_frame[:, 0] * bone_lengths["hip_span_half"]
    )

    # 脚関節: 各関節の局所 X 軸方向に沿って子関節を配置
    thigh_l = bone_lengths.get("thigh_L", bone_lengths.get("thigh", 0.40))
    calf_l = bone_lengths.get("calf_L", bone_lengths.get("calf", 0.40))
    thigh_r = bone_lengths.get("thigh_R", bone_lengths.get("thigh", 0.40))
    calf_r = bone_lengths.get("calf_R", bone_lengths.get("calf", 0.40))

    positions[JOINT_INDEX["L_knee"]] = (
        positions[JOINT_INDEX["L_hip"]] + global_frames["L_hip"][:, 0] * thigh_l
    )
    positions[JOINT_INDEX["L_ankle"]] = (
        positions[JOINT_INDEX["L_knee"]] + global_frames["L_knee"][:, 0] * calf_l
    )
    positions[JOINT_INDEX["R_knee"]] = (
        positions[JOINT_INDEX["R_hip"]] + global_frames["R_hip"][:, 0] * thigh_r
    )
    positions[JOINT_INDEX["R_ankle"]] = (
        positions[JOINT_INDEX["R_knee"]] + global_frames["R_knee"][:, 0] * calf_r
    )

    return positions


# ---------------------------------------------------------------------------
# 1. 順運動学（FK）による再構築と PA-MPJPE 完全一致検証
# ---------------------------------------------------------------------------

class TestForwardKinematicsRoundtrip:
    """「正規化座標 → オイラー角 → 順運動学座標」の Round-trip 整合性検証。"""

    @pytest.mark.parametrize("fixture_name", ["tpose_coco", "squat_coco", "raised_left_arm_coco"])
    def test_roundtrip_pa_mpjpe_is_strictly_zero(self, request, fixture_name):
        """
        モック姿勢データにおいて、逆運動学（calculate_joint_angles）で求めた
        オイラー角から順運動学（FK）で再構築した 15 関節座標と、
        正規化された元の 15 関節座標との PA-MPJPE が、
        機械精度（1e-10 以下）でゼロ（完全一致）となることを検証。
        """
        coco_frame = request.getfixturevalue(fixture_name)
        named_pts = _named_points(coco_frame)

        # フェーズ1で定義された正規化 15 関節座標（腰原点）
        original_15j = np.array([named_pts[j] - named_pts["hips"] for j in JOINT_NAMES])
        bone_lengths = extract_bone_lengths_from_points(named_pts)

        # IK: 座標 → オイラー角
        angles = calculate_joint_angles(coco_frame[np.newaxis, ...])[0]

        # FK: オイラー角 → 3D 座標再構築
        reconstructed_15j = forward_kinematics(angles, bone_lengths)

        # PA-MPJPE を計算
        error = pa_mpjpe(reconstructed_15j, original_15j)

        assert error < 1e-10, (
            f"姿勢 [{fixture_name}] の Round-trip PA-MPJPE ({error:.2e} m) が許容限界 (1e-10) を超えました。"
        )

    def test_scale_invariance_in_roundtrip(self, tpose_coco):
        """
        同一姿勢でスケール（体格）が 0.5 倍および 2.0 倍に変化しても、
        PA-MPJPE による形状誤差が機械精度でゼロを保つこと。
        """
        for scale in [0.5, 1.5, 2.0]:
            scaled_coco = make_tpose_coco(scale=scale)
            named_pts = _named_points(scaled_coco)
            original_15j = np.array([named_pts[j] - named_pts["hips"] for j in JOINT_NAMES])
            bone_lengths = extract_bone_lengths_from_points(named_pts)

            angles = calculate_joint_angles(scaled_coco[np.newaxis, ...])[0]
            reconstructed_15j = forward_kinematics(angles, bone_lengths)

            error = pa_mpjpe(reconstructed_15j, original_15j)
            assert error < 1e-10, (
                f"スケール {scale} 倍における Round-trip PA-MPJPE ({error:.2e}) が 0 になりませんでした。"
            )


# ---------------------------------------------------------------------------
# 2. ルート（腰）座標系の原点固定確認
# ---------------------------------------------------------------------------

class TestHipsRootOriginConstraint:
    """再構築後の姿勢において腰（hips）が厳密に原点 (0, 0, 0) に固定されていることの検証。"""

    @pytest.mark.parametrize("fixture_name", ["tpose_coco", "squat_coco", "raised_left_arm_coco"])
    def test_reconstructed_hips_is_strictly_at_origin(self, request, fixture_name):
        """任意のモック姿勢から再構築された 3D 座標において、腰（hips）が厳密に原点にあること。"""
        coco_frame = request.getfixturevalue(fixture_name)
        named_pts = _named_points(coco_frame)
        bone_lengths = extract_bone_lengths_from_points(named_pts)

        angles = calculate_joint_angles(coco_frame[np.newaxis, ...])[0]
        reconstructed_15j = forward_kinematics(angles, bone_lengths)

        hips_pos = reconstructed_15j[JOINT_INDEX["hips"]]
        np.testing.assert_allclose(
            hips_pos,
            np.zeros(3),
            atol=1e-12,
            err_msg=f"再構築された腰の座標が原点 (0, 0, 0) ではありません: {hips_pos}",
        )


# ---------------------------------------------------------------------------
# 3. 特異点（ジンバルロック付近）での姿勢復元検証
# ---------------------------------------------------------------------------

class TestSingularityAndGimbalLockRoundtrip:
    """ジンバルロック（特異点）を含む姿勢に対する往復復元（Round-trip）の堅牢性検証。"""

    def test_gimbal_lock_posture_reconstruction(self, tpose_coco):
        """
        腕を垂直真上に振り上げた特異点（ピッチ90°付近）の姿勢において、
        オイラー角計算が破綻せず、再構築後の座標が元の姿勢形状と PA-MPJPE (1e-10 以下)
        で完全に一致すること。
        """
        gimbal_coco = tpose_coco.copy()
        left_shoulder = gimbal_coco[5]  # leftshoulder
        gimbal_coco[7] = left_shoulder + np.array([0.0, 0.27, 0.0])  # leftelbow
        gimbal_coco[9] = left_shoulder + np.array([0.0, 0.50, 0.0])  # leftwrist

        named_pts = _named_points(gimbal_coco)
        original_15j = np.array([named_pts[j] - named_pts["hips"] for j in JOINT_NAMES])
        bone_lengths = extract_bone_lengths_from_points(named_pts)

        # IK
        angles = calculate_joint_angles(gimbal_coco[np.newaxis, ...])[0]
        assert np.all(np.isfinite(angles)), "ジンバルロック姿勢で角度に NaN または Inf が発生しました。"

        # FK
        reconstructed_15j = forward_kinematics(angles, bone_lengths)
        assert np.all(np.isfinite(reconstructed_15j)), "再構築座標に NaN または Inf が発生しました。"

        # PA-MPJPE による完全一致アサーション
        error = pa_mpjpe(reconstructed_15j, original_15j)
        assert error < 1e-10, (
            f"ジンバルロック姿勢の Round-trip PA-MPJPE ({error:.2e}) が許容限界 (1e-10) を超えました。"
        )


# ---------------------------------------------------------------------------
# 4. ロボット固定ボーン長モデルによる姿勢再構築の幾何学的整合性
# ---------------------------------------------------------------------------

class TestRobotFixedBoneLengthsReconstruction:
    """ロボット固定ボーン長を用いた再構築における各部位の方向ベクトル検証。"""

    @pytest.mark.parametrize("fixture_name", ["tpose_coco", "squat_coco", "raised_left_arm_coco"])
    def test_robot_fixed_bone_limb_directions_match_original(self, request, fixture_name):
        """
        ロボット固定ボーン長を用いて再構築した際、スケールやプロポーションが異なっていても、
        各ボーン（上腕、前腕、大腿、下腿）の方向ベクトルが元の姿勢と
        コサイン類似度 1.0（完全一致）であることを検証。
        """
        coco_frame = request.getfixturevalue(fixture_name)
        named_pts = _named_points(coco_frame)
        angles = calculate_joint_angles(coco_frame[np.newaxis, ...])[0]

        # ロボット固定ボーン長で再構築
        robot_coords = forward_kinematics(angles, DEFAULT_ROBOT_BONE_LENGTHS)

        # 検証対象のボーンペア（親関節名, 子関節名）
        bone_pairs = [
            ("L_shoulder", "L_elbow"),
            ("L_elbow", "L_wrist"),
            ("R_shoulder", "R_elbow"),
            ("R_elbow", "R_wrist"),
            ("L_hip", "L_knee"),
            ("L_knee", "L_ankle"),
            ("R_hip", "R_knee"),
            ("R_knee", "R_ankle"),
        ]

        for parent_name, child_name in bone_pairs:
            p_idx = JOINT_INDEX[parent_name]
            c_idx = JOINT_INDEX[child_name]

            # 元のボーン方向ベクトル
            orig_vec = named_pts[child_name] - named_pts[parent_name]
            orig_unit = orig_vec / np.linalg.norm(orig_vec)

            # 再構築後のボーン方向ベクトル
            recon_vec = robot_coords[c_idx] - robot_coords[p_idx]
            recon_unit = recon_vec / np.linalg.norm(recon_vec)

            # コサイン類似度
            cosine_sim = float(np.dot(orig_unit, recon_unit))
            np.testing.assert_allclose(
                cosine_sim,
                1.0,
                atol=1e-10,
                err_msg=f"ボーン [{parent_name} -> {child_name}] の再構築方向が元の方向と一致しません (sim={cosine_sim:.6f})",
            )
