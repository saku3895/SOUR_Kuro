"""Load and validate single-pose COCO 17-point JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .calculate_joint_angles import COCO_INDEX


def load_keypoints_json(path: str | Path) -> tuple[str, float, np.ndarray]:
    """Return pose name, scale, and keypoints shaped ``(17, 3)``."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON形式が不正です: {error}") from error
    except OSError:
        raise
    return parse_keypoints_json(data)


def parse_keypoints_json(data: Any) -> tuple[str, float, np.ndarray]:
    """Validate parsed JSON data and return pose name, scale, and keypoints."""
    if not isinstance(data, dict):
        raise ValueError("JSONのルートはオブジェクトである必要があります")
    keypoints = data.get("keypoints")
    if not isinstance(keypoints, dict):
        raise ValueError("keypoints objectが必要です")

    missing = [name for name in COCO_INDEX if name not in keypoints]
    if missing:
        raise ValueError(f"keypointsが不足しています: {', '.join(missing)}")

    try:
        scale = float(data.get("scale", 1.0))
    except (TypeError, ValueError) as error:
        raise ValueError("scaleは数値である必要があります") from error
    if not np.isfinite(scale):
        raise ValueError("scaleは有限値である必要があります")

    points = np.empty((len(COCO_INDEX), 3), dtype=float)
    for name, index in COCO_INDEX.items():
        value = keypoints[name]
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"{name}の座標は3要素の配列である必要があります")
        try:
            points[index] = [float(component) for component in value]
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name}の座標は数値である必要があります") from error

    if not np.all(np.isfinite(points)):
        raise ValueError("keypointsには有限値のみ指定してください")
    return str(data.get("pose_name", "名前なし")), scale, points * scale


__all__ = ["load_keypoints_json", "parse_keypoints_json"]
