"""Calculate anatomical joint angles from 17-point YOLO pose data.

The calculation API accepts NumPy arrays.  The CSV functions at the bottom
are an offline input/output adapter for recorded pose data.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np


JOINTS = {
    "nose": "j00_nose",
    "left_eye": "j01_L_eye",
    "right_eye": "j02_R_eye",
    "left_ear": "j03_L_ear",
    "right_ear": "j04_R_ear",
    "left_shoulder": "j05_L_shoulder",
    "right_shoulder": "j06_R_shoulder",
    "left_elbow": "j07_L_elbow",
    "right_elbow": "j08_R_elbow",
    "left_wrist": "j09_L_wrist",
    "right_wrist": "j10_R_wrist",
    "left_hip": "j11_L_hip",
    "right_hip": "j12_R_hip",
    "left_knee": "j13_L_knee",
    "right_knee": "j14_R_knee",
    "left_ankle": "j15_L_ankle",
    "right_ankle": "j16_R_ankle",
}

ANGLE_TRIPLETS = {
    "neck_angle_deg": ("hip_center", "neck", "nose"),
    "left_shoulder_angle_deg": ("neck", "left_shoulder", "left_elbow"),
    "right_shoulder_angle_deg": ("neck", "right_shoulder", "right_elbow"),
    "left_elbow_angle_deg": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow_angle_deg": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_hip_angle_deg": ("neck", "left_hip", "left_knee"),
    "right_hip_angle_deg": ("neck", "right_hip", "right_knee"),
    "left_knee_angle_deg": ("left_hip", "left_knee", "left_ankle"),
    "right_knee_angle_deg": ("right_hip", "right_knee", "right_ankle"),
}

JOINT_INDEX = {name: index for index, name in enumerate(JOINTS)}


def _unit(vector: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length < 1e-8:
        return None
    return vector / length


def _angle(first: np.ndarray, vertex: np.ndarray, last: np.ndarray) -> float:
    first_axis = _unit(first - vertex)
    last_axis = _unit(last - vertex)
    if first_axis is None or last_axis is None:
        return math.nan
    cosine = float(np.clip(np.dot(first_axis, last_axis), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize the last axis and preserve invalid vectors as NaN."""
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = vectors / lengths
    return np.where(np.isfinite(normalized) & (lengths >= 1e-8), normalized, np.nan)


def _vector_angles(first: np.ndarray, vertex: np.ndarray,
                   last: np.ndarray) -> np.ndarray:
    first_axis = _normalize_vectors(first - vertex)
    last_axis = _normalize_vectors(last - vertex)
    cosine = np.sum(first_axis * last_axis, axis=-1)
    with np.errstate(invalid="ignore"):
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _median_filter(points: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return points
    if window % 2 == 0:
        raise ValueError("median window must be an odd positive integer")
    radius = window // 2
    filtered = np.full_like(points, np.nan)
    for row in range(points.shape[0]):
        start = max(0, row - radius)
        stop = min(points.shape[0], row + radius + 1)
        with np.errstate(all="ignore"):
            filtered[row] = np.nanmedian(points[start:stop], axis=0)
    return filtered


def read_pose_csv(path: str | Path, max_abs_coordinate: float = 3.0) -> tuple[list[dict[str, str]], dict[str, np.ndarray]]:
    """Read recorder metadata and points, rejecting non-finite/outlier points."""
    metadata: list[dict[str, str]] = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        while True:
            position = stream.tell()
            line = stream.readline()
            if not line:
                raise ValueError("CSV header was not found")
            if not line.startswith("#"):
                stream.seek(position)
                break
            key, _, value = line[1:].partition(":")
            metadata.append({"key": key.strip(), "value": value.strip()})

        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("CSV header was not found")
        required = [f"{prefix}_{axis}" for prefix in JOINTS.values() for axis in ("x", "y", "z")]
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"missing keypoint columns: {', '.join(missing[:3])}")

        rows = list(reader)
    points = np.full((len(rows), len(JOINTS), 3), np.nan, dtype=float)
    for row_index, row in enumerate(rows):
        for joint_index, prefix in enumerate(JOINTS.values()):
            try:
                point = np.array([float(row[f"{prefix}_{axis}"]) for axis in ("x", "y", "z")])
            except (TypeError, ValueError):
                continue
            if np.all(np.isfinite(point)) and np.all(np.abs(point) <= max_abs_coordinate):
                points[row_index, joint_index] = point
    return metadata, {name: points[:, index] for index, name in enumerate(JOINTS)}


def _derived_points(points: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    derived = dict(points)
    derived["hip_center"] = (points["left_hip"] + points["right_hip"]) / 2
    derived["neck"] = (points["left_shoulder"] + points["right_shoulder"]) / 2
    return derived


def _points_to_named_array(points: np.ndarray) -> dict[str, np.ndarray]:
    """Return named views for a ``(frames, 17, 3)`` pose array."""
    if points.ndim == 2:
        points = points[np.newaxis, ...]
    if points.ndim != 3 or points.shape[1:] != (len(JOINTS), 3):
        raise ValueError("points must have shape (frames, 17, 3) or (17, 3)")
    return {name: points[:, index] for name, index in JOINT_INDEX.items()}


def calculate_pose_angles(points: np.ndarray, median_window: int = 1) -> dict[str, np.ndarray]:
    """Calculate pose angles directly from a ``(frames, 17, 3)`` array.

    The returned arrays have one value per frame and are expressed in degrees.
    Invalid or incomplete poses produce NaN.  The head orientation is reported
    as pitch, yaw and roll relative to the torso coordinate frame.
    """
    named_points = _points_to_named_array(np.asarray(points, dtype=float))
    if median_window > 1:
        named_points = {
            name: _median_filter(value, median_window)
            for name, value in named_points.items()
        }
    joints = _derived_points(named_points)
    result = {
        name: _vector_angles(joints[first], joints[vertex], joints[last])
        for name, (first, vertex, last) in ANGLE_TRIPLETS.items()
    }
    result.update(_head_orientation(joints))
    return result


def calculate_angles(points: dict[str, np.ndarray], median_window: int = 3) -> dict[str, np.ndarray]:
    """Return one angle series per anatomical joint, in degrees."""
    joints = _derived_points({
        name: _median_filter(value, median_window) for name, value in points.items()
    })
    return {
        name: _vector_angles(joints[first], joints[vertex], joints[last])
        for name, (first, vertex, last) in ANGLE_TRIPLETS.items()
    }


def _head_orientation(points: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Estimate head yaw, pitch and roll relative to the torso frame."""
    right = _normalize_vectors(points["right_shoulder"] - points["left_shoulder"])
    up = _normalize_vectors(points["neck"] - points["hip_center"])
    face = _normalize_vectors(
        points["nose"] - (points["left_eye"] + points["right_eye"]) / 2
    )
    eyes = _normalize_vectors(points["right_eye"] - points["left_eye"])
    forward = _normalize_vectors(np.cross(right, up))

    face_right = np.sum(face * right, axis=-1)
    face_forward = np.sum(face * forward, axis=-1)
    face_up = np.sum(face * up, axis=-1)
    eyes_up = np.sum(eyes * up, axis=-1)
    eyes_right = np.sum(eyes * right, axis=-1)
    horizontal = np.hypot(face_right, face_forward)
    with np.errstate(invalid="ignore"):
        return {
            "head_yaw_deg": np.degrees(np.arctan2(face_right, face_forward)),
            "head_pitch_deg": np.degrees(np.arctan2(face_up, horizontal)),
            "head_roll_deg": np.degrees(np.arctan2(eyes_up, eyes_right)),
        }


def write_angle_csv(input_path: str | Path, output_path: str | Path,
                    median_window: int = 3, max_abs_coordinate: float = 3.0) -> None:
    metadata, points = read_pose_csv(input_path, max_abs_coordinate)
    pose_array = np.stack([points[name] for name in JOINTS], axis=1)
    angles = calculate_pose_angles(pose_array, median_window)
    with Path(input_path).open(newline="", encoding="utf-8") as stream:
        while stream.readline().startswith("#"):
            pass
        stream.seek(0)
        source_reader = csv.DictReader(line for line in stream if not line.startswith("#"))
        source_rows = list(source_reader)
    base_columns = [column for column in ("timestamp", "frame_id", "fps", "depth_m")
                    if source_reader.fieldnames and column in source_reader.fieldnames]
    fieldnames = base_columns + list(angles)
    with Path(output_path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, source in enumerate(source_rows):
            row = {column: source.get(column, "") for column in base_columns}
            row.update({name: "" if math.isnan(values[index]) else f"{values[index]:.3f}"
                        for name, values in angles.items()})
            writer.writerow(row)


def write_angle_csv_from_array(points: np.ndarray, output_path: str | Path,
                               median_window: int = 3) -> None:
    """Write calculated angles from camera pose data to a CSV file."""
    angles = calculate_pose_angles(points, median_window)
    fieldnames = list(angles)
    with Path(output_path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(points)):
            writer.writerow({
                name: "" if math.isnan(values[index]) else f"{values[index]:.3f}"
                for name, values in angles.items()
            })


def main(arguments: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Calculate joint angles from a 17-point pose CSV")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="output angle CSV (default: <input>_angles.csv)")
    parser.add_argument("--median-window", type=int, default=3)
    parser.add_argument("--max-abs-coordinate", type=float, default=3.0)
    args = parser.parse_args(arguments)
    output = args.output or args.input_csv.with_name(f"{args.input_csv.stem}_angles.csv")
    write_angle_csv(args.input_csv, output, args.median_window, args.max_abs_coordinate)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()