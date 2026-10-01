"""Encode joint-angle sequences as human motion language."""

from collections.abc import Mapping

import numpy as np


AXIS_MAPPING = {
    "A": (0, 0, "signed", 10),  # hips Roll
    "B": (0, 1, "signed", 10),  # hips Pitch
    "C": (0, 2, "signed", 10),  # hips Yaw
    "D": (1, 0, "signed", 10),  # neck Roll
    "E": (1, 1, "signed", 10),  # neck Pitch
    "F": (1, 2, "signed", 10),  # neck Yaw
    "G": (3, 0, "signed", 10),  # L_shoulder Roll
    "H": (3, 1, "signed", 10),  # L_shoulder Pitch
    "I": (3, 2, "signed", 10),  # L_shoulder Yaw
    "J": (4, 1, "absolute", 10),  # L_elbow Pitch
    "K": (6, 0, "signed", 10),  # R_shoulder Roll
    "L": (6, 1, "signed", 10),  # R_shoulder Pitch
    "M": (6, 2, "signed", 10),  # R_shoulder Yaw
    "N": (7, 1, "absolute", 10),  # R_elbow Pitch
}

# 下半身対応時に、以下をAXIS_MAPPINGへ追加する。
# "O": (9, 0, "signed", 10),  # L_hip Roll
# "P": (9, 1, "signed", 10),  # L_hip Pitch
# "Q": (9, 2, "signed", 10),  # L_hip Yaw
# "R": (10, 1, "absolute", 10),  # L_knee Pitch
# "S": (12, 0, "signed", 10),  # R_hip Roll
# "T": (12, 1, "signed", 10),  # R_hip Pitch
# "U": (12, 2, "signed", 10),  # R_hip Yaw
# "V": (13, 1, "absolute", 10),  # R_knee Pitch

AXIS_IDS = tuple(AXIS_MAPPING)


class MotionLanguageEncoder:
    """Convert ``(frames, 15, 3)`` Euler-angle arrays to ``<...>`` lines.

    The input angle order is ``[Roll, Pitch, Yaw]``. Each configured axis
    receives its own alphabetic ID and quantized value.
    """

    def __init__(self, num_levels=None, axis_levels=None):
        if num_levels is not None and (
            not isinstance(num_levels, (int, np.integer)) or num_levels < 1
        ):
            raise ValueError("num_levels must be a positive integer")
        if axis_levels is not None and not isinstance(axis_levels, Mapping):
            raise TypeError("axis_levels must be a mapping of axis IDs to levels")

        self.axis_levels = {
            axis_id: int(mapping[3]) for axis_id, mapping in AXIS_MAPPING.items()
        }
        if num_levels is not None:
            self.axis_levels = {
                axis_id: int(num_levels) for axis_id in AXIS_IDS
            }
        if axis_levels is not None:
            unknown_axes = set(axis_levels) - set(AXIS_IDS)
            if unknown_axes:
                raise ValueError(f"unknown axis IDs: {sorted(unknown_axes)}")
            self.axis_levels.update(axis_levels)
        if any(
            not isinstance(level, (int, np.integer)) or level < 1
            for level in self.axis_levels.values()
        ):
            raise ValueError("axis quantization levels must be positive integers")
        self.axis_levels = {
            axis_id: int(level) for axis_id, level in self.axis_levels.items()
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

    def _quantize_signed(self, value, num_levels):
        clipped = np.clip(value, -180.0, 180.0)
        quantized = np.floor((clipped + 180.0) / 360.0 * num_levels)
        return int(np.clip(quantized, 0, num_levels - 1))

    def _quantize_absolute(self, value, num_levels):
        clipped = np.clip(abs(value), 0.0, 180.0)
        quantized = np.floor(clipped / 180.0 * num_levels)
        return int(np.clip(quantized, 0, num_levels - 1))

    def _encode_axis(self, axis_id, frame_angles):
        joint_index, axis_index, quantization, _ = AXIS_MAPPING[axis_id]
        num_levels = self.axis_levels[axis_id]
        value = frame_angles[joint_index, axis_index]
        if quantization == "absolute":
            quantized = self._quantize_absolute(value, num_levels)
        else:
            quantized = self._quantize_signed(value, num_levels)
        return f"{axis_id}{quantized}"

    def encode_frame(self, current_frame_angles, prev_frame_angles=None):
        """Encode one frame, emitting only changed quantized joints when set."""
        current = self._validate_frame(current_frame_angles)
        previous = None
        if prev_frame_angles is not None:
            previous = self._validate_frame(prev_frame_angles)

        current_tokens = {
            axis_id: self._encode_axis(axis_id, current)
            for axis_id in AXIS_IDS
        }
        if previous is None:
            tokens = current_tokens.values()
        else:
            previous_tokens = {
                axis_id: self._encode_axis(axis_id, previous)
                for axis_id in AXIS_IDS
            }
            tokens = (
                current_tokens[axis_id]
                for axis_id in AXIS_IDS
                if current_tokens[axis_id] != previous_tokens[axis_id]
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
    dummy_angles = np.zeros((3, 15, 3), dtype=float)
    dummy_angles[0, 0] = [-180.0, 0.0, 180.0]
    dummy_angles[1, 0, 0] = 36.0
    dummy_angles[1, 4, 1] = 36.0
    dummy_angles[2] = dummy_angles[1]

    encoder = MotionLanguageEncoder()
    print(encoder.encode_sequence(dummy_angles))
    print(MotionLanguageEncoder(num_levels=16).encode_frame(dummy_angles[1]))
