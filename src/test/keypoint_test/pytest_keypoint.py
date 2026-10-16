# src/test/keypoint_test/pytest_keypoint.py

import pytest
import numpy as np
from pose_estimation import calculate_joint_angles as calc_angles
from pose_estimation import motion_language as m_lang
from pose_estimation import visualize_motion as vis_motion
from pose_estimation import motion_config as m_config


@pytest.fixture
def dummy_coco_coords():
    """テスト用の 1フレーム COCO 17点 座標 (1, 17, 3)"""
    coords = np.zeros((1, 17, 3), dtype=float)
    
    # 0: nose, 1: lefteye, 2: righteye, 3: leftear, 4: rightear
    coords[0, 0] = [0.0, 0.85, 0.0]
    coords[0, 1] = [-0.02, 0.87, 0.0]
    coords[0, 2] = [0.02, 0.87, 0.0]
    coords[0, 3] = [-0.05, 0.86, 0.0]
    coords[0, 4] = [0.05, 0.86, 0.0]
    
    # 5: leftshoulder, 6: rightshoulder
    coords[0, 5] = [-0.2, 0.7, 0.0]
    coords[0, 6] = [0.2, 0.7, 0.0]
    
    # 7: leftelbow, 8: rightelbow
    coords[0, 7] = [-0.4, 0.7, 0.0]
    coords[0, 8] = [0.4, 0.7, 0.0]
    
    # 9: leftwrist, 10: rightwrist
    coords[0, 9] = [-0.6, 0.7, 0.0]
    coords[0, 10] = [0.6, 0.7, 0.0]
    
    # 11: lefthip, 12: righthip
    coords[0, 11] = [-0.1, 0.0, 0.0]
    coords[0, 12] = [0.1, 0.0, 0.0]
    
    # 13: leftknee, 14: rightknee
    coords[0, 13] = [-0.1, -0.4, 0.0]
    coords[0, 14] = [0.1, -0.4, 0.0]
    
    # 15: leftankle, 16: rightankle
    coords[0, 15] = [-0.1, -0.8, 0.0]
    coords[0, 16] = [0.1, -0.8, 0.0]
    
    return coords


# ===================================================================
# 1. パイプライン疎通・型チェック
# ===================================================================

def test_1_calculate_joint_angles(dummy_coco_coords):
    """1. 座標 -> 関節角度 (Euler角) の計算検証"""
    angles = calc_angles.calculate_joint_angles(dummy_coco_coords)
    assert angles.shape == (1, len(m_config.JOINT_NAMES), 3)
    assert not np.isnan(angles).any()
    assert not np.isinf(angles).any()


def test_2_quantization_and_encode():
    """2. 角度 -> 動作言語エンコードの検証"""
    encoder = m_lang.MotionLanguageEncoder()
    test_angles = np.zeros((2, len(m_config.JOINT_NAMES), 3), dtype=float)
    tokens = encoder.encode(test_angles)
    assert len(tokens) == 2
    assert tokens[0].startswith("<") and tokens[0].endswith(">")
    assert tokens[1] == "<>"


def test_3_decode_and_reconstruction():
    """3. 動作言語 -> デコード -> 3D座標再構成の検証"""
    decoder = m_lang.MotionLanguageDecoder()
    test_tokens = ["<A0B4C6D9>", "<>"]
    decoded_angles = decoder.decode_lines(test_tokens)
    assert decoded_angles.shape == (2, len(m_config.JOINT_NAMES), 3)
    reconstructed_coords = vis_motion.reconstruct_3d_positions(decoded_angles)
    assert reconstructed_coords.shape == (2, len(m_config.JOINT_NAMES), 3)


def test_4_decode_invalid_tokens():
    """4. デコード時の不正トークンに対する異常系テスト"""
    decoder = m_lang.MotionLanguageDecoder()
    with pytest.raises(ValueError):
        decoder.decode_lines(["<A99>"])
    with pytest.raises(ValueError):
        decoder.decode_lines(["<Z9>"])


# ===================================================================
# 2. 数値精度・量子化限界テスト
# ===================================================================

def test_5_coco_to_euler_tpose_precision(dummy_coco_coords):
    """5. COCO座標（Tポーズ）からの角度算出精度検証"""
    angles = calc_angles.calculate_joint_angles(dummy_coco_coords)
    flexion_joint_indices = [4, 5, 7, 8, 10, 11, 13, 14]  # 肘・手首・膝・足首
    assert np.allclose(angles[0, flexion_joint_indices], 0.0, atol=1e-5), (
        f"屈曲関節の角度に不自然なズレがあります:\n{angles[0, flexion_joint_indices]}"
    )


def test_6_quantization_roundtrip_error_bounds():
    """6. 各軸の可動域内における量子化復元誤差が理論最大値（Δ/2）以内か検証"""
    encoder = m_lang.MotionLanguageEncoder()
    decoder = m_lang.MotionLanguageDecoder()

    np.random.seed(42)
    test_angles = np.zeros((10, len(m_config.JOINT_NAMES), 3), dtype=float)
    for axis_id, (j_idx, e_idx, _, _, min_v, max_v) in m_config.AXIS_MAPPING.items():
        test_angles[:, j_idx, e_idx] = np.random.uniform(min_v, max_v, size=10)

    tokens = encoder.encode(test_angles)
    decoded_angles = decoder.decode_lines(tokens)

    for axis_id, (joint_idx, euler_idx, _, levels, min_val, max_val) in m_config.AXIS_MAPPING.items():
        step = (max_val - min_val) / levels
        max_allowed_error = step / 2.0 + 1e-5

        input_val = test_angles[:, joint_idx, euler_idx]
        decoded_val = decoded_angles[:, joint_idx, euler_idx]
        error = np.abs(input_val - decoded_val)

        assert np.all(error <= max_allowed_error), (
            f"軸 [{axis_id}] の最大誤差 {np.max(error):.2f}° が許容限界 {max_allowed_error:.2f}° を超えました。"
        )


def test_7_zero_angle_config_exactness():
    """7. 全軸の設定において 0.0° が量子化レベルの中央値に存在するか検証"""
    for axis_id, (_, _, _, levels, min_val, max_val) in m_config.AXIS_MAPPING.items():
        step = (max_val - min_val) / levels
        midpoints = [min_val + step * (q + 0.5) for q in range(levels)]
        has_zero = any(abs(m) < 1e-5 for m in midpoints)
        assert has_zero, (
            f"軸 [{axis_id}] (範囲:[{min_val}, {max_val}], 分割:{levels}) に 0.0° のレベルが存在しません。"
        )


def test_8_tpose_zero_angle_reconstruction_integrity():
    """8. 全関節 0° 入力時、エンコード➔デコード後も完全 0.0° が保持されるか検証"""
    encoder = m_lang.MotionLanguageEncoder()
    decoder = m_lang.MotionLanguageDecoder()

    zero_angles = np.zeros((1, len(m_config.JOINT_NAMES), 3), dtype=float)
    tokens = encoder.encode(zero_angles)
    decoded_angles = decoder.decode_lines(tokens)

    assert np.allclose(decoded_angles, 0.0, atol=1e-5), (
        f"0° (Tポーズ) の復元角度に誤差が発生しています:\n{decoded_angles}"
    )


# ===================================================================
# 3. 3D空間座標差分計測（完全一致アサーションは行わず数値計測）
# ===================================================================

def test_9_tpose_3d_coordinate_difference():
    """9. Tポーズ (0°) 時における直計算座標とデコード後座標のズレ（cm単位）を算出・表示"""
    zero_angles = np.zeros((1, len(m_config.JOINT_NAMES), 3), dtype=float)
    
    # 直計算の 3D 座標
    coords_direct = vis_motion.reconstruct_3d_positions(zero_angles)[0]

    # デコード経由の 3D 座標
    encoder = m_lang.MotionLanguageEncoder()
    decoder = m_lang.MotionLanguageDecoder()
    decoded_angles = decoder.decode_lines(encoder.encode(zero_angles))
    coords_decoded = vis_motion.reconstruct_3d_positions(decoded_angles)[0]

    # メートル -> センチメートル換算
    diff_xyz_cm = (coords_decoded - coords_direct) * 100.0
    dist_cm = np.linalg.norm(diff_xyz_cm, axis=-1)

    print("\n================ Tポーズ (0°) 復元時の 3D 座標ズレ一覧 ================")
    print(f"{'関節名':<12} | {'直線距離 (cm)':<12} | {'dx (cm)':<8} | {'dy (cm)':<8} | {'dz (cm)':<8}")
    print("-" * 65)
    for idx, name in enumerate(m_config.JOINT_NAMES):
        dx, dy, dz = diff_xyz_cm[idx]
        d_cm = dist_cm[idx]
        print(f"{name:<12} | {d_cm:12.4f} | {dx:8.4f} | {dy:8.4f} | {dz:8.4f}")
    print("=======================================================================")


def test_10_coordinate_quantization_error_bounds():
    """10. 量子化による 3D 空間位置の平均誤差が許容値（5cm以内）か検証"""
    encoder = m_lang.MotionLanguageEncoder()
    decoder = m_lang.MotionLanguageDecoder()

    np.random.seed(42)
    sample_angles = np.zeros((50, len(m_config.JOINT_NAMES), 3), dtype=float)
    for axis_id, (j_idx, e_idx, _, _, min_v, max_v) in m_config.AXIS_MAPPING.items():
        sample_angles[:, j_idx, e_idx] = np.random.uniform(min_v, max_v, size=50)

    coords_direct = vis_motion.reconstruct_3d_positions(sample_angles)
    tokens = encoder.encode(sample_angles)
    decoded_angles = decoder.decode_lines(tokens)
    coords_decoded = vis_motion.reconstruct_3d_positions(decoded_angles)

    pos_diff_cm = np.linalg.norm(coords_direct - coords_decoded, axis=-1) * 100.0
    mean_overall_error = np.mean(pos_diff_cm)

    assert mean_overall_error < 5.0, (
        f"3D 空間の平均位置誤差 ({mean_overall_error:.2f} cm) が許容限界 (5.0 cm) を超えました。"
    )