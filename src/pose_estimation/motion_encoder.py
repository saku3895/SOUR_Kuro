"""Encode joint-angle sequences as human motion language."""

from collections.abc import Mapping

import numpy as np


DEFAULT_JOINT_INDICES = {
    "A": 0,  # hips
    "B": 1,  # neck
    "C": 3,  # L_shoulder
    "D": 4,  # L_elbow
    "E": 6,  # R_shoulder
    "F": 7,  # R_elbow
    # 下半身対応を再開するときに、以下を有効化する。
    # "G": 9,  # L_hip
    # "H": 10,  # L_knee
    # "I": 12,  # R_hip
    # "J": 13,  # R_knee
}

UPPER_BODY_JOINTS = ("A", "B", "C", "D", "E", "F")
# 下半身対応を再開するときは、G〜Jを追加してこの集合を切り替える。
# ALL_BODY_JOINTS = UPPER_BODY_JOINTS + ("G", "H", "I", "J")
_THREE_AXIS_JOINTS = frozenset("ABCE")
# 下半身対応時はG/Iを3軸関節へ追加する。
# _THREE_AXIS_JOINTS = frozenset("ABCEGI")
_PITCH_AXIS = 1


class MotionLanguageEncoder:
    """Convert ``(frames, 15, 3)`` Euler-angle arrays to ``<...>`` lines.

    The input angle order is ``[Roll, Pitch, Yaw]``. Three-axis joints use
    all three values in that order. Elbows and knees use the absolute Pitch
    value as their single flexion/extension value.
    """

    def __init__(self, joint_indices=None):
        """Create an encoder with an optional A-F to input-index mapping."""
        if joint_indices is None:
            joint_indices = DEFAULT_JOINT_INDICES
        if not isinstance(joint_indices, Mapping):
            raise TypeError("joint_indices must be a mapping")
        if set(joint_indices) != set(UPPER_BODY_JOINTS):
            raise ValueError("joint_indices must contain exactly A through F")
        if any(
            not isinstance(index, (int, np.integer)) or not 0 <= index < 15
            for index in joint_indices.values()
        ):
            raise ValueError("joint indices must be integers from 0 through 14")
        self.joint_indices = {
            label: int(joint_indices[label]) for label in UPPER_BODY_JOINTS
        }

    @staticmethod
    def _validate_frame(frame_angles):
        if not isinstance(frame_angles, np.ndarray):
            raise TypeError("frame angles must be a NumPy array")
        frame = np.asarray(frame_angles, dtype=float)
        if frame.shape != (15, 3):
            raise ValueError("frame angles must have shape (15, 3)")
        if not np.all(np.isfinite(frame)):
            raise ValueError("frame angles must contain only finite values")
        return frame

    @staticmethod
    def _quantize_three_axis(values):
        clipped = np.clip(values, -180.0, 180.0)
        quantized = np.floor((clipped + 180.0) / 360.0 * 10.0)
        return np.clip(quantized, 0, 9).astype(int)

    @staticmethod
    def _quantize_one_axis(value):
        clipped = np.clip(abs(value), 0.0, 180.0)
        quantized = np.floor(clipped / 180.0 * 10.0)
        return int(np.clip(quantized, 0, 9))

    def _encode_joint(self, label, frame_angles):
        values = frame_angles[self.joint_indices[label]]
        if label in _THREE_AXIS_JOINTS:
            return f"{label}{''.join(map(str, self._quantize_three_axis(values)))}"
        return f"{label}{self._quantize_one_axis(values[_PITCH_AXIS])}"

    def encode_frame(self, current_frame_angles, prev_frame_angles=None):
        """Encode one frame, emitting only changed quantized joints when set."""
        current = self._validate_frame(current_frame_angles)
        previous = None
        if prev_frame_angles is not None:
            previous = self._validate_frame(prev_frame_angles)

        current_tokens = {
            label: self._encode_joint(label, current)
            for label in UPPER_BODY_JOINTS
        }
        if previous is None:
            tokens = current_tokens.values()
        else:
            previous_tokens = {
                label: self._encode_joint(label, previous)
                for label in UPPER_BODY_JOINTS
            }
            tokens = (
                current_tokens[label]
                for label in UPPER_BODY_JOINTS
                if current_tokens[label] != previous_tokens[label]
            )
        return f"<{''.join(tokens)}>"

    def encode_sequence(self, raw_angles_array):
        """Encode all frames as newline-separated human motion language."""
        if not isinstance(raw_angles_array, np.ndarray):
            raise TypeError("raw_angles_array must be a NumPy array")
        angles = np.asarray(raw_angles_array, dtype=float)
        if angles.ndim != 3 or angles.shape[1:] != (15, 3):
            raise ValueError("raw_angles_array must have shape (frames, 15, 3)")
        if angles.shape[0] == 0:
            raise ValueError("raw_angles_array must contain at least one frame")
        if not np.all(np.isfinite(angles)):
            raise ValueError("raw_angles_array must contain only finite values")

        lines = [self.encode_frame(angles[0])]
        for frame_index in range(1, angles.shape[0]):
            lines.append(self.encode_frame(angles[frame_index], angles[frame_index - 1]))
        return "\n".join(lines)


if __name__ == "__main__":
    # Extra joints and axes remain in the input but are intentionally ignored.
    dummy_angles = np.zeros((3, 15, 3), dtype=float)
    dummy_angles[0, 0] = [-180.0, 0.0, 180.0]
    dummy_angles[1, 4, 1] = 18.0
    dummy_angles[1, 6] = [36.0, 0.0, 0.0]
    dummy_angles[2] = dummy_angles[1]

    encoder = MotionLanguageEncoder()
    print(encoder.encode_sequence(dummy_angles))
