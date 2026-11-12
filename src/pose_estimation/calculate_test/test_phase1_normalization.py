"""
フェーズ1：座標正規化と基準点の検証 ユニットテスト。

検証内容:
1. 基準点算出（首・腰の中点、頭部重心）の幾何学的正確性
2. 腰を原点とする相対座標系への平行移動の正確性（腰が (0, 0, 0) に厳密に配置されること）
3. PA-MPJPE (Procrustes Aligned MPJPE) による純粋な姿勢形状維持の検証
   （骨の長さの違い・スケール差に引きずられないこと）
4. 並進不変性（任意の位置に配置されたCOCO座標が同じ腰原点系に収束すること）
5. スケール不変性（体格の違いによるスケール差があっても姿勢形状が維持されること）
6. 異常系・エッジケース（ゼロベクトル、NaN/Inf、形状不正）のハンドリング
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow this test module to be executed directly from the repository root.
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import pytest

from pose_estimation.calculate_joint_angles import (
    COCO_INDEX,
    _named_points,
    calculate_joint_angles,
)
from pose_estimation.motion_config import JOINT_INDEX, JOINT_NAMES
from pose_estimation.visualize_motion import reconstruct_3d_positions

from pose_estimation.calculate_test.conftest import (
    make_tpose_coco,
    pa_mpjpe,
    procrustes_align,
)


# ---------------------------------------------------------------------------
# 1. 基準点（首・腰・頭部）の計算検証
# ---------------------------------------------------------------------------

class TestReferencePointCalculation:
    """_named_points による基準点（腰中点、首中点、頭部重心）の算出ロジック検証。"""

    def test_hips_is_exact_midpoint_of_left_and_right_hip(self, tpose_coco):
        """腰（hips）が左右の股関節（lefthip, righthip）の厳密な中点であること。"""
        points = _named_points(tpose_coco)
        expected_hips = (
            tpose_coco[COCO_INDEX["lefthip"]] + tpose_coco[COCO_INDEX["righthip"]]
        ) / 2.0
        np.testing.assert_allclose(
            points["hips"],
            expected_hips,
            atol=1e-12,
            err_msg="腰中点（hips）の計算が lefthip と righthip の中点と一致しません。",
        )

    def test_neck_is_exact_midpoint_of_left_and_right_shoulder(self, tpose_coco):
        """首（neck）が左右の肩（leftshoulder, rightshoulder）の厳密な中点であること。"""
        points = _named_points(tpose_coco)
        expected_neck = (
            tpose_coco[COCO_INDEX["leftshoulder"]] + tpose_coco[COCO_INDEX["rightshoulder"]]
        ) / 2.0
        np.testing.assert_allclose(
            points["neck"],
            expected_neck,
            atol=1e-12,
            err_msg="首中点（neck）の計算が leftshoulder と rightshoulder の中点と一致しません。",
        )

    def test_head_is_centroid_of_five_facial_landmarks(self, tpose_coco):
        """頭部（head）が顔の5点（鼻・両目・両耳）の厳密な重心であること。"""
        points = _named_points(tpose_coco)
        facial_indices = [
            COCO_INDEX["nose"],
            COCO_INDEX["lefteye"],
            COCO_INDEX["righteye"],
            COCO_INDEX["leftear"],
            COCO_INDEX["rightear"],
        ]
        expected_head = np.mean(tpose_coco[facial_indices], axis=0)
        np.testing.assert_allclose(
            points["head"],
            expected_head,
            atol=1e-12,
            err_msg="頭部（head）の計算が顔5点の重心と一致しません。",
        )

    def test_named_points_preserves_limb_coordinates(self, tpose_coco):
        """手足の四肢関節が COCO 座標の値をそのまま保持していること。"""
        points = _named_points(tpose_coco)
        for side in ("L", "R"):
            source = "left" if side == "L" else "right"
            for joint in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
                original = tpose_coco[COCO_INDEX[f"{source}{joint}"]]
                np.testing.assert_allclose(
                    points[f"{side}_{joint}"],
                    original,
                    atol=1e-12,
                    err_msg=f"{side}_{joint} の座標が元の {source}{joint} と一致しません。",
                )


# ---------------------------------------------------------------------------
# 2. 腰原点化と相対座標系の検証
# ---------------------------------------------------------------------------

class TestHipsOriginNormalization:
    """腰を原点とした相対座標系への変換と、基準点の原点配置の検証。"""

    def test_reconstructed_hips_is_strictly_at_origin(self, tpose_coco_batch):
        """
        パイプライン（calculate_joint_angles → reconstruct_3d_positions）通過後、
        腰（hips）が厳密に原点 (0, 0, 0) に配置されていること。
        """
        angles = calculate_joint_angles(tpose_coco_batch)
        positions = reconstruct_3d_positions(angles)

        hips_pos = positions[0, JOINT_INDEX["hips"]]
        np.testing.assert_allclose(
            hips_pos,
            np.zeros(3),
            atol=1e-12,
            err_msg=f"腰関節が厳密な原点 (0, 0, 0) に配置されていません: {hips_pos}",
        )

    def test_arbitrary_global_translation_invariance(self, tpose_coco):
        """
        入力 COCO 座標に任意の大域的オフセットを加えても、
        出力される関節角度および相対関節座標が完全に一致すること（並進不変性）。
        """
        base_batch = tpose_coco[np.newaxis, ...]
        base_angles = calculate_joint_angles(base_batch)
        base_positions = reconstruct_3d_positions(base_angles)

        # 任意の大規模オフセット（カメラから遠く離れた位置を模倣）
        offsets = [
            np.array([100.0, 200.0, 300.0]),
            np.array([-50.0, -50.0, -50.0]),
            np.array([0.001, -0.002, 0.005]),
        ]
        for offset in offsets:
            translated_coco = base_batch + offset
            translated_angles = calculate_joint_angles(translated_coco)
            translated_positions = reconstruct_3d_positions(translated_angles)

            # 角度が一致すること
            np.testing.assert_allclose(
                translated_angles,
                base_angles,
                atol=1e-10,
                err_msg=f"オフセット {offset} を加えた際に角度が変化しました。",
            )
            # 再構成座標（腰原点系）が一致すること
            np.testing.assert_allclose(
                translated_positions,
                base_positions,
                atol=1e-10,
                err_msg=f"オフセット {offset} を加えた際に再構成座標が変化しました。",
            )


# ---------------------------------------------------------------------------
# 3. PA-MPJPE による姿勢形状維持の検証（プロクラステス解析）
# ---------------------------------------------------------------------------

class TestProcrustesAlignedMPJPE:
    """
    PA-MPJPE を用いて、骨の長さの違い（スケール差）に引きずられることなく
    純粋な「姿勢の形」が正しく変換できているかを幾何学的に検証する。

    注意: 入力（人間のCOCO骨格）と出力（ロボット骨格）はボーン長が異なるため、
    単純な関節位置の引き算は絶対に行わず、必ず Procrustes 解析で
    「位置」「スケール」「向き」を揃えた上で誤差を評価する。
    """

    @staticmethod
    def _extract_comparable_points(
        coco_frame: np.ndarray,
    ) -> np.ndarray:
        """
        COCO 17 点から、ロボット 15 関節に対応する 15 関節座標を抽出する。
        （_named_points のロジックを用いて hips, neck, head を補完）
        """
        points = _named_points(coco_frame)
        comparable = np.empty((len(JOINT_NAMES), 3), dtype=float)
        for idx, name in enumerate(JOINT_NAMES):
            comparable[idx] = points[name]
        return comparable

    def test_coco_normalization_pipeline_pa_mpjpe(self, tpose_coco):
        """
        カメラから取得した生COCO座標（大域オフセット＋スケール変動あり）を
        腰原点相対座標系へ正規化するパイプラインにおいて、
        Procrustes解析による PA-MPJPE が機械ゼロ（1e-12 以下）となり、
        姿勢形状が完全に維持されること。
        """
        # 生の入力データ（カメラ座標系: カメラから離れた位置 + スケール変動）
        camera_coco = make_tpose_coco(scale=1.75) + np.array([120.0, -50.0, 300.0])

        # 処理パイプライン: 基準点算出 & 腰原点への相対化
        raw_pts = _named_points(camera_coco)
        normalized_15j = np.array([raw_pts[name] - raw_pts["hips"] for name in JOINT_NAMES])

        # 基準となる理想姿勢（原点・標準スケール）
        ref_pts = _named_points(tpose_coco)
        reference_15j = np.array([ref_pts[name] - ref_pts["hips"] for name in JOINT_NAMES])

        # プロクラステス解析でスケール・向き・位置のアライメント誤差を評価
        error = pa_mpjpe(normalized_15j, reference_15j)
        assert error < 1e-12, (
            f"正規化パイプライン後の PA-MPJPE ({error:.2e}) が 0 になりませんでした。"
        )

    def test_scale_invariance_of_pa_mpjpe(self, tpose_coco):
        """
        人間の体格（骨の長さ）が 0.5 倍（小柄）または 2.0 倍（大柄）に
        変動しても、プロクラステス解析によりスケール差が完全に除去され、
        相対的なPA-MPJPEが完全に不変であること。
        """
        scales = [0.5, 0.75, 1.0, 1.5, 2.0]
        # 腰原点化した15点
        pts_base = _named_points(tpose_coco)
        human_base_15j = np.array([pts_base[name] - pts_base["hips"] for name in JOINT_NAMES])

        for s in scales:
            scaled_coco = make_tpose_coco(scale=s)
            pts_s = _named_points(scaled_coco)
            human_s_15j = np.array([pts_s[name] - pts_s["hips"] for name in JOINT_NAMES])

            # スケールが違っても、PA-MPJPE（同一形状）は機械ゼロ (1e-12 以下)
            error_identity = pa_mpjpe(human_s_15j, human_base_15j)
            assert error_identity < 1e-12, (
                f"スケール {s} 倍の姿勢に対する PA-MPJPE ({error_identity:.2e}) が 0 になりませんでした。"
            )

    def test_zero_length_vector_limb_raises_value_error(self, tpose_coco):
        """
        関節間（例: 肩と肘）の距離が0（ゼロ長ベクトル）の場合、
        ゼロ除算を防ぐため ValueError が送出されること。
        """
        corrupted = tpose_coco.copy()
        # 左肘を左肩と同一点にする（L_shoulder -> L_elbow のベクトル長 = 0）
        corrupted[COCO_INDEX["leftelbow"]] = corrupted[COCO_INDEX["leftshoulder"]]

        with pytest.raises(ValueError, match="zero-length vector"):
            calculate_joint_angles(corrupted[np.newaxis, ...])

    def test_procrustes_alignment_mathematical_properties(self):
        """
        Procrustes 解析自体の数学的性質の検証:
        同一形状に対してスケール・回転・平行移動を加えた場合、
        整列後の残差が機械精度 (1e-10) で 0 になること。
        """
        source = np.random.RandomState(42).randn(15, 3)
        # ランダムな回転
        theta = np.radians(45.0)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        # スケール + 回転 + 平行移動
        target = (source @ R.T) * 2.5 + np.array([10.0, -20.0, 30.0])

        error = pa_mpjpe(source, target)
        assert error < 1e-10, (
            f"数学的に相似な形状に対する PA-MPJPE ({error:.2e}) が 0 になりませんでした。"
        )


# ---------------------------------------------------------------------------
# 4. エッジケース・異常系のハンドリング検証
# ---------------------------------------------------------------------------

class TestNormalizationEdgeCases:
    """異常な入力やゼロ長ベクトル等に対する堅牢性・エラーハンドリング検証。"""

    def test_impossible_17_point_data_raises_value_error(self):
        """
        17点すべてが同一点にある、人間の姿勢として不可能な入力を拒否する。

        配列の形状自体は正しいが、左右肩の距離が0のため体幹座標系を
        構築できず、zero-length vector エラーになる。
        """
        impossible_pose = np.full((1, 17, 3), 100.0, dtype=float)

        with pytest.raises(ValueError, match="zero-length vector for body"):
            calculate_joint_angles(impossible_pose)

    def test_zero_length_vector_shoulder_raises_value_error(self, tpose_coco):
        """
        左右の肩が完全一致（距離0）している場合、
        ゼロ除算を防ぐため ValueError が送出されること。
        """
        corrupted = tpose_coco.copy()
        # 左肩と右肩を同一点にする（体幹フレーム構築時にゼロ長ベクトル）
        corrupted[COCO_INDEX["rightshoulder"]] = corrupted[COCO_INDEX["leftshoulder"]]

        with pytest.raises(ValueError, match="zero-length vector"):
            calculate_joint_angles(corrupted[np.newaxis, ...])

    def test_nan_coordinates_raise_value_error(self, tpose_coco):
        """座標に NaN が含まれる場合、明確な ValueError が送出されること。"""
        corrupted = tpose_coco.copy()
        corrupted[0, 0] = np.nan

        with pytest.raises(ValueError, match="finite values"):
            calculate_joint_angles(corrupted[np.newaxis, ...])

    def test_inf_coordinates_raise_value_error(self, tpose_coco):
        """座標に Inf が含まれる場合、明確な ValueError が送出されること。"""
        corrupted = tpose_coco.copy()
        corrupted[5, 1] = np.inf

        with pytest.raises(ValueError, match="finite values"):
            calculate_joint_angles(corrupted[np.newaxis, ...])

    def test_invalid_shape_raises_value_error(self):
        """(frames, 17, 3) 以外の不正な形状の配列は ValueError となること。"""
        with pytest.raises(ValueError, match="shape"):
            calculate_joint_angles(np.zeros((1, 16, 3)))
        with pytest.raises(ValueError, match="shape"):
            calculate_joint_angles(np.zeros((1, 17, 2)))
        with pytest.raises(ValueError, match="shape"):
            calculate_joint_angles(np.zeros((17, 3)))  # 2D（frames次元欠落）

    def test_empty_frames_raises_value_error(self):
        """フレーム数 0 の入力に対して ValueError が送出されること。"""
        with pytest.raises(ValueError, match="numeric frames"):
            calculate_joint_angles(np.zeros((0, 17, 3)))

    def test_non_array_input_raises_type_error(self):
        """リスト等の非 ndarray 入力に対して TypeError が送出されること。"""
        with pytest.raises(TypeError, match="NumPy array"):
            calculate_joint_angles([[0.0] * 3] * 17)  # type: ignore
