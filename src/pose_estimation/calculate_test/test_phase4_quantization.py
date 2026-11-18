"""
フェーズ4：動作言語化パイプライン（量子化とエンコード/デコード）の検証 ユニットテスト。

検証内容:
1. 量子化ロジックの正確性（線形スケーリング、境界値、クリッピング処理）
2. エンコード仕様の検証（初回フレーム全軸出力、次フレーム以降の差分エンコード、無変化時の空タグ `<>`）
3. デコードと状態保持の検証（復元値がレベル中央値になること、差分更新時の直前状態保持）
4. Round-trip（角度 → 言語 → 角度）の量子化誤差検証（各軸の誤差が理論最大誤差 Δ/2 以内であること）
5. MPJLE（知覚的許容誤差）テスト（デコード後FK再構築姿勢のPA-MPJPEがペナルティ閾値内に収まること）
6. 不正フォーマット入力に対する堅牢性・エラーハンドリング（不正な軸記号、レベル外、括弧欠損など）
"""

from __future__ import annotations

import numpy as np
import pytest

from pose_estimation.calculate_joint_angles import (
    _named_points,
    calculate_joint_angles,
)
from pose_estimation.motion_config import (
    AXIS_IDS,
    AXIS_MAPPING,
    JOINT_NAMES,
    T_POSE_OFFSETS,
)
from pose_estimation.motion_language import (
    MotionLanguageDecoder,
    MotionLanguageEncoder,
    from_motion_angles,
    to_motion_angles,
)
from pose_estimation.calculate_test.conftest import (
    make_tpose_coco,
    make_squat_coco,
    make_raised_left_arm_coco,
    pa_mpjpe,
)
from pose_estimation.calculate_test.test_phase3_kinematics import (
    extract_bone_lengths_from_points,
    forward_kinematics,
)


# ---------------------------------------------------------------------------
# 1. 量子化ロジックの正確性検証
# ---------------------------------------------------------------------------

class TestQuantizationLogic:
    """各軸の角度の離散化（スケーリング、境界値、クリッピング）の検証。"""

    def test_all_axis_levels_are_approximately_fifteen_degrees(self):
        """全軸の1レベル幅が15度以内に収まることを確認する。"""
        for axis_id, (_, _, _, levels, min_v, max_v) in AXIS_MAPPING.items():
            width = (max_v - min_v) / levels
            assert 0 < width <= 15.0, (
                f"軸 {axis_id} の1レベル幅が15度を超えています: {width:.2f}度"
            )

    def test_quantize_midpoints_and_endpoints(self):
        """設定された [min, max] の各レベル境界および中間値が正しい整数レベルに丸められること。"""
        encoder = MotionLanguageEncoder()

        for axis_id, (_, _, _, levels, min_v, max_v) in AXIS_MAPPING.items():
            step = (max_v - min_v) / levels

            # 最小値 -> レベル 0
            assert encoder._quantize(min_v, min_v, max_v, levels) == 0
            # 最大値境界 -> レベル levels - 1
            assert encoder._quantize(max_v, min_v, max_v, levels) == levels - 1

            # 各レベルの中央値 -> 当該レベル index
            for lvl in range(levels):
                mid_val = min_v + step * (lvl + 0.5)
                quantized = encoder._quantize(mid_val, min_v, max_v, levels)
                assert quantized == lvl, (
                    f"軸 [{axis_id}] の中央値 {mid_val:.2f}° がレベル {lvl} ではなく {quantized} になりました。"
                )

    def test_quantize_out_of_bounds_clipping(self):
        """可動域 [min, max] を超える異常値が入力された際、正しく境界レベル（0 または N-1）にクリッピングされること。"""
        encoder = MotionLanguageEncoder()

        for axis_id, (_, _, _, levels, min_v, max_v) in AXIS_MAPPING.items():
            # 最小値を大きく下回る値 -> レベル 0
            assert encoder._quantize(min_v - 100.0, min_v, max_v, levels) == 0
            assert encoder._quantize(-9999.0, min_v, max_v, levels) == 0

            # 最大値を大きく上回る値 -> レベル levels - 1
            assert encoder._quantize(max_v + 100.0, min_v, max_v, levels) == levels - 1
            assert encoder._quantize(9999.0, min_v, max_v, levels) == levels - 1


# ---------------------------------------------------------------------------
# 2. エンコード仕様の検証
# ---------------------------------------------------------------------------

class TestEncodingSpecification:
    """エンコーダーの初回全軸出力、差分出力、無変化時の空タグ出力の検証。"""

    def test_first_frame_encodes_all_axes(self):
        """初回フレームでは全24軸のトークンが欠損なく1つのタグ内に生成されること。"""
        encoder = MotionLanguageEncoder()
        test_angles = np.zeros((1, len(JOINT_NAMES), 3), dtype=float)

        tokens = encoder.encode(test_angles)
        assert len(tokens) == 1

        first_token = tokens[0]
        assert first_token.startswith("<") and first_token.endswith(">")

        body = first_token[1:-1]
        for axis_id in AXIS_IDS:
            assert axis_id in body, f"初回トークンに軸 [{axis_id}] が含まれていません: {first_token}"

    def test_differential_encoding_and_empty_tag_on_unchanged(self):
        """
        連続するフレームにおいて:
        - 変化のないフレームは空タグ `<>` となること
        - 特定の軸のみ変化した場合はその軸のみが含まれること
        """
        encoder = MotionLanguageEncoder()
        frames = np.zeros((4, len(JOINT_NAMES), 3), dtype=float)

        # frame 0: 全軸 0°
        # frame 1: frame 0 と同一（変化なし）
        # frame 2: J軸（左肘Pitch: joint 4, axis 1）のみ 60° に変化
        frames[2, 4, 1] = 60.0
        # frame 3: frame 2 と同一（変化なし）
        frames[3, 4, 1] = 60.0

        tokens = encoder.encode(frames)
        assert len(tokens) == 4

        # frame 0: 全軸
        for axis_id in AXIS_IDS:
            assert axis_id in tokens[0]

        # frame 1: 変化なし -> 空タグ
        assert tokens[1] == "<>"

        # frame 2: J軸のみ変化
        assert "J" in tokens[2]
        # J軸以外の軸は含まれないこと
        for other_axis in AXIS_IDS:
            if other_axis != "J":
                assert other_axis not in tokens[2]

        # frame 3: 変化なし -> 空タグ
        assert tokens[3] == "<>"


# ---------------------------------------------------------------------------
# 3. デコードと状態保持の検証
# ---------------------------------------------------------------------------

class TestDecodingAndStateCarrying:
    """デコーダーの復元値精度（レベル中央値）および状態保持（差分受信時の直前値維持）の検証。"""

    def test_dequantize_returns_level_midpoint(self):
        """量子化レベルから角度への復元値が、設定された刻み幅の厳密な中央値であること。"""
        decoder = MotionLanguageDecoder()

        for axis_id, (_, _, _, levels, min_v, max_v) in AXIS_MAPPING.items():
            step = (max_v - min_v) / levels
            for lvl in range(levels):
                expected_val = min_v + step * (lvl + 0.5)
                restored_val = decoder._dequantize(axis_id, lvl)
                np.testing.assert_allclose(
                    restored_val,
                    expected_val,
                    atol=1e-10,
                    err_msg=f"軸 [{axis_id}] レベル {lvl} の復元値が中央値と一致しません。",
                )

    def test_state_carrying_across_differential_frames(self):
        """差分トークンや空トークンを受信した際、未更新の軸が直前フレームの角度を正しく保持し続けること。"""
        decoder = MotionLanguageDecoder()

        # フレーム列:
        # frame 0: A0, J7 を設定
        # frame 1: 空タグ <> (A0, J7 が保持されるべき)
        # frame 2: J2 のみ更新 (A0 は保持され、J は J2 の値になるべき)
        token_lines = ["<A0J7>", "<>", "<J2>"]
        decoded = decoder.decode(token_lines)
        assert decoded.shape == (3, len(JOINT_NAMES), 3)

        val_A0 = decoder._dequantize("A", 0)
        val_J7 = decoder._dequantize("J", 7)
        val_J2 = decoder._dequantize("J", 2)

        j_A, e_A = AXIS_MAPPING["A"][0], AXIS_MAPPING["A"][1]
        j_J, e_J = AXIS_MAPPING["J"][0], AXIS_MAPPING["J"][1]

        # frame 0
        assert np.isclose(decoded[0, j_A, e_A], val_A0)
        assert np.isclose(decoded[0, j_J, e_J], val_J7)

        # frame 1 (空タグで保持)
        assert np.isclose(decoded[1, j_A, e_A], val_A0)
        assert np.isclose(decoded[1, j_J, e_J], val_J7)

        # frame 2 (Aは保持、Jは更新)
        assert np.isclose(decoded[2, j_A, e_A], val_A0)
        assert np.isclose(decoded[2, j_J, e_J], val_J2)


# ---------------------------------------------------------------------------
# 4. Round-trip（角度 → 言語 → 角度）の量子化誤差検証
# ---------------------------------------------------------------------------

class TestRoundtripQuantizationErrorBounds:
    """エンコード後デコードした際の各軸の角度誤差が理論最大誤差（Δ/2）以内であることの検証。"""

    @pytest.mark.parametrize("fixture_name", ["tpose_coco", "squat_coco", "raised_left_arm_coco"])
    def test_motion_angles_wrap_all_axes_to_principal_range(self, request, fixture_name):
        """
        Tポーズとの差分角度がEuler角の周期性を無視して±360°へ逸脱せず、
        全関節・全軸で主値範囲 [-180°, 180°) に正規化されることを確認する。
        """
        coco_frame = request.getfixturevalue(fixture_name)
        absolute_angles = calculate_joint_angles(coco_frame[np.newaxis, ...])
        expected = (absolute_angles - np.asarray(T_POSE_OFFSETS)[np.newaxis, ...] + 180.0) % 360.0 - 180.0

        motion_angles = to_motion_angles(absolute_angles)

        np.testing.assert_allclose(motion_angles, expected, atol=1e-10)
        decoded_motion = MotionLanguageDecoder().decode(
            MotionLanguageEncoder().encode(motion_angles)
        )
        for axis_id, (joint, axis, _, levels, minimum, maximum) in AXIS_MAPPING.items():
            raw_value = absolute_angles[0, joint, axis] - T_POSE_OFFSETS[joint][axis]
            wrapped_value = motion_angles[0, joint, axis]
            if (
                not np.isclose(raw_value, wrapped_value)
                and minimum <= wrapped_value <= maximum
            ):
                step = (maximum - minimum) / levels
                error = abs(decoded_motion[0, joint, axis] - wrapped_value)
                assert error <= step / 2.0 + 1e-5, (
                    f"{fixture_name} の軸 {axis_id} ({JOINT_NAMES[joint]} 軸{axis})で"
                    "角度の折り返しに失敗: "
                    f"期待値={wrapped_value:.3f}°, 復元値={decoded_motion[0, joint, axis]:.3f}°"
                )

    def test_random_angles_within_bounds_roundtrip_error(self):
        """可動域内のランダムな連続角度に対して、復元誤差がステップ幅の半分 (step / 2.0) 以下に収まること。"""
        encoder = MotionLanguageEncoder()
        decoder = MotionLanguageDecoder()

        rng = np.random.default_rng(42)
        n_frames = 20
        test_angles = np.zeros((n_frames, len(JOINT_NAMES), 3), dtype=float)

        for axis_id, (j_idx, e_idx, _, _, min_v, max_v) in AXIS_MAPPING.items():
            test_angles[:, j_idx, e_idx] = rng.uniform(min_v, max_v, size=n_frames)

        tokens = encoder.encode(test_angles)
        decoded_angles = decoder.decode(tokens)

        for axis_id, (j_idx, e_idx, _, levels, min_v, max_v) in AXIS_MAPPING.items():
            step = (max_v - min_v) / levels
            max_allowed_error = step / 2.0 + 1e-5

            input_vals = test_angles[:, j_idx, e_idx]
            decoded_vals = decoded_angles[:, j_idx, e_idx]
            abs_errors = np.abs(input_vals - decoded_vals)

            max_err = float(np.max(abs_errors))
            assert max_err <= max_allowed_error, (
                f"軸 [{axis_id}] の最大復元誤差 {max_err:.3f}° が理論限界 {max_allowed_error:.3f}° を超過しました。"
            )

    @pytest.mark.parametrize("fixture_name", ["tpose_coco", "squat_coco", "raised_left_arm_coco"])
    def test_mock_postures_quantized_angle_errors(self, request, fixture_name):
        """各モック姿勢（T-pose、スクワット、腕挙げ）から得られた実角度の復元誤差が理論最大値以内であること。"""
        coco_frame = request.getfixturevalue(fixture_name)
        angles = calculate_joint_angles(coco_frame[np.newaxis, ...])
        motion_angles = to_motion_angles(angles)

        encoder = MotionLanguageEncoder()
        decoder = MotionLanguageDecoder()

        tokens = encoder.encode(motion_angles)
        decoded_motion = decoder.decode(tokens)

        for axis_id, (j_idx, e_idx, _, levels, min_v, max_v) in AXIS_MAPPING.items():
            step = (max_v - min_v) / levels
            max_allowed_error = step / 2.0 + 1e-5

            # クリッピング後の入力値と比較
            expected_input = np.clip(motion_angles[0, j_idx, e_idx], min_v, max_v)
            decoded_val = decoded_motion[0, j_idx, e_idx]
            error = abs(expected_input - decoded_val)

            assert error <= max_allowed_error, (
                f"姿勢 [{fixture_name}] 軸 [{axis_id}] の誤差 {error:.2f}° が許容限界 {max_allowed_error:.2f}° を超えました。"
            )


# ---------------------------------------------------------------------------
# 5. MPJLE（知覚的許容誤差）テスト
# ---------------------------------------------------------------------------

class TestPerceptualLocalizationErrorMPJLE:
    """量子化・復元された姿勢をFKで3D再構築した際の位置誤差（PA-MPJPE）がペナルティ閾値内に収まるかの検証。"""

    @pytest.mark.parametrize(
        ("fixture_name", "threshold_m"),
        [
            ("tpose_coco", 0.04),            # T-pose: 40mm (4cm) 以内
            ("raised_left_arm_coco", 0.04),   # 腕挙げ: 40mm (4cm) 以内
            ("squat_coco", 0.10),             # スクワット（屈曲姿勢）: 100mm (10cm) 以内
        ],
    )
    def test_reconstructed_3d_pose_mpjle_below_threshold(self, request, fixture_name, threshold_m):
        """
        モック姿勢を動作言語へエンコード・デコードし、順運動学（FK）で再構築した3D座標が、
        量子化前の元の座標に対してあらかじめ定めた許容閾値内に収まり、姿勢形状が破綻なく再現されること。
        """
        coco_frame = request.getfixturevalue(fixture_name)
        named_pts = _named_points(coco_frame)
        bone_lengths = extract_bone_lengths_from_points(named_pts)

        # 1. 姿勢推定オイラー角の算出と動作言語角度への変換
        orig_angles = calculate_joint_angles(coco_frame[np.newaxis, ...])
        motion_angles = to_motion_angles(orig_angles)

        # 2. 動作言語エンコード -> デコード
        encoder = MotionLanguageEncoder()
        decoder = MotionLanguageDecoder()
        tokens = encoder.encode(motion_angles)
        decoded_motion = decoder.decode(tokens)
        decoded_angles = from_motion_angles(decoded_motion)

        # 3. 順運動学（FK）による 3D 座標の再構築
        orig_3d = forward_kinematics(orig_angles[0], bone_lengths)
        quantized_3d = forward_kinematics(decoded_angles[0], bone_lengths)

        # 4. PA-MPJPE による姿勢局所化誤差（MPJLE）の評価
        error_m = pa_mpjpe(quantized_3d, orig_3d)

        assert error_m <= threshold_m, (
            f"姿勢 [{fixture_name}] の量子化後 3D 位置誤差 MPJLE ({error_m * 1000.0:.2f} mm) が "
            f"許容ペナルティ閾値 ({threshold_m * 1000.0:.2f} mm) を超過しました。"
        )


# ---------------------------------------------------------------------------
# 6. 不正フォーマット入力に対する堅牢性・エラーハンドリング
# ---------------------------------------------------------------------------

class TestInvalidFormatErrorHandling:
    """異常系・不正なフォーマットが入力された際の例外ハンドリング検証。"""

    def test_decode_missing_brackets_raises_value_error(self):
        """山括弧 `<>` で囲まれていない不正な行に対して ValueError が送出されること。"""
        decoder = MotionLanguageDecoder()
        with pytest.raises(ValueError, match="expected <...>"):
            decoder.decode(["A5B5"])
        with pytest.raises(ValueError, match="expected <...>"):
            decoder.decode(["<A5B5"])
        with pytest.raises(ValueError, match="expected <...>"):
            decoder.decode(["A5B5>"])

    def test_decode_unknown_axis_symbol_raises_value_error(self):
        """存在しない軸記号（例: `Z`）を含むトークンに対して ValueError が送出されること。"""
        decoder = MotionLanguageDecoder()
        with pytest.raises(ValueError, match="invalid token"):
            decoder.decode(["<Z5>"])
        with pytest.raises(ValueError, match="invalid token"):
            decoder.decode(["<A5Y0>"])

    def test_decode_out_of_range_level_raises_value_error(self):
        """定義された最大レベル数を超える不正なレベル指定に対して ValueError が送出されること。"""
        decoder = MotionLanguageDecoder()
        # A軸は levels=11 (0..10 が有効、99 は不正)
        with pytest.raises(ValueError, match="outside"):
            decoder.decode(["<A99>"])

    def test_decode_duplicate_axis_raises_value_error(self):
        """同一フレーム内で同一の軸IDが重複指定されている場合に ValueError が送出されること。"""
        decoder = MotionLanguageDecoder()
        with pytest.raises(ValueError, match="duplicate axis ID"):
            decoder.decode(["<A5A6>"])

    def test_decode_empty_lines_input_raises_value_error(self):
        """有効なフレームが存在しない空行のみの入力に対して ValueError が送出されること。"""
        decoder = MotionLanguageDecoder()
        with pytest.raises(ValueError, match="no frames"):
            decoder.decode(["   ", ""])

    def test_encode_invalid_shape_and_non_finite_raises_value_error(self):
        """エンコーダーに入力された配列が不正な形状や NaN/Inf を含む場合に ValueError が送出されること。"""
        encoder = MotionLanguageEncoder()
        with pytest.raises(ValueError, match="shape"):
            encoder.encode(np.zeros((1, 14, 3)))
        with pytest.raises(ValueError, match="shape"):
            encoder.encode(np.zeros((15, 3)))
        with pytest.raises(ValueError, match="finite"):
            bad_angles = np.zeros((1, 15, 3))
            bad_angles[0, 0, 0] = np.nan
            encoder.encode(bad_angles)
