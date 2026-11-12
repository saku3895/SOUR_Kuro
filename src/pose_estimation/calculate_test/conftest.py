"""
共通フィクスチャと合成データ生成ユーティリティ。

pytest は conftest.py を自動検出するため、同ディレクトリ配下のテストで
フィクスチャを import なしで参照できる。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# src/ ディレクトリをパスに追加（既存テスト pytest_keypoint.py と同様の方式）
_SRC_DIR = Path(__file__).resolve().parents[2]  # …/src/
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from pose_estimation.keypoint_json import load_keypoints_json


_POSTURE_DIR = Path(__file__).resolve().parents[3] / "data_logs" / "posture"


# ---------------------------------------------------------------------------
# 合成 T 字ポーズ（T-pose）生成
# ---------------------------------------------------------------------------

def make_tpose_coco(scale: float = 1.0) -> np.ndarray:
    _, _, keypoints = load_keypoints_json(_POSTURE_DIR / "t_pose.json")
    return keypoints * scale


def make_raised_left_arm_coco() -> np.ndarray:
    """左腕を真上に挙上したポーズ（T-pose の左腕変形版）。"""
    _, _, keypoints = load_keypoints_json(_POSTURE_DIR / "raised_left_arm.json")
    return keypoints


def make_squat_coco() -> np.ndarray:
    """膝を 90° 程度屈曲させたスクワットポーズ。"""
    _, _, keypoints = load_keypoints_json(_POSTURE_DIR / "squat.json")
    return keypoints


# ---------------------------------------------------------------------------
# PA-MPJPE（Procrustes Aligned Mean Per Joint Position Error）
# ---------------------------------------------------------------------------

def procrustes_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Procrustes 解析により source を target に整列（平行移動・スケール・回転）する。

    スケール差（人間骨格 vs ロボット骨格）を除去した純粋な姿勢誤差の評価に使用。

    Args:
        source: (N, 3) 整列元座標
        target: (N, 3) 整列先座標
    Returns:
        source_aligned: (N, 3) target のスケール・位置へ整列後の source
    """
    # 1. 重心を引く
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    s = source - mu_s
    t = target - mu_t

    # 2. スケール正規化（Frobenius ノルム）
    ss = np.sqrt((s ** 2).sum(axis=1).mean())
    st = np.sqrt((t ** 2).sum(axis=1).mean())
    if ss < 1e-12:
        return source.copy()
    s = s / ss
    t_norm = t / st if st > 1e-12 else t

    # 3. 最適回転（SVD: H = s^T t → R = V U^T）
    H = s.T @ t_norm
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # 鏡像補正（det(R) = -1 の場合）
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 4. 整列して target スケール・位置へ戻す
    return (s @ R.T) * st + mu_t


def pa_mpjpe(pred: np.ndarray, target: np.ndarray) -> float:
    """
    PA-MPJPE を計算する。

    スケール・回転・平行移動を Procrustes 解析で合わせた後の
    関節位置平均誤差（target と同じ単位）。

    Args:
        pred:   (N, 3) 予測関節座標
        target: (N, 3) 参照関節座標
    Returns:
        float: 平均関節位置誤差（target 座標のスケール単位）
    """
    pred_aligned = procrustes_align(pred, target)
    return float(np.mean(np.linalg.norm(pred_aligned - target, axis=1)))


# ---------------------------------------------------------------------------
# pytest フィクスチャ
# ---------------------------------------------------------------------------

@pytest.fixture
def tpose_coco() -> np.ndarray:
    """T 字ポーズ COCO 座標 (17, 3)。"""
    return make_tpose_coco()


@pytest.fixture
def tpose_coco_batch() -> np.ndarray:
    """T 字ポーズ COCO 座標のバッチ形式 (1, 17, 3)。"""
    return make_tpose_coco()[np.newaxis, ...]


@pytest.fixture
def raised_left_arm_coco() -> np.ndarray:
    """左腕挙上ポーズ COCO 座標 (17, 3)。"""
    return make_raised_left_arm_coco()


@pytest.fixture
def squat_coco() -> np.ndarray:
    """スクワットポーズ COCO 座標 (17, 3)。"""
    return make_squat_coco()
