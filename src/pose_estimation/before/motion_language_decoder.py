"""Decode human motion-language text into upper-body joint angles."""

from dataclasses import dataclass
import re

import numpy as np


@dataclass(frozen=True)
class AxisSpec:
    joint_index: int
    angle_index: int
    mode: str
    levels: int = 10


# This is a decoder-side protocol definition. It intentionally does not call
# MotionLanguageEncoder or read any original angle data.
AXIS_MAPPING = {
    "A": AxisSpec(0, 0, "signed"),
    "B": AxisSpec(0, 1, "signed"),
    "C": AxisSpec(0, 2, "signed"),
    "D": AxisSpec(1, 0, "signed"),
    "E": AxisSpec(1, 1, "signed"),
    "F": AxisSpec(1, 2, "signed"),
    "G": AxisSpec(2, 0, "signed"),
    "H": AxisSpec(2, 1, "signed"),
    "I": AxisSpec(2, 2, "signed"),
    "J": AxisSpec(3, 1, "absolute"),
    "K": AxisSpec(4, 0, "signed"),
    "L": AxisSpec(4, 1, "signed"),
    "M": AxisSpec(4, 2, "signed"),
    "N": AxisSpec(5, 1, "absolute"),
}

AXIS_IDS = tuple(AXIS_MAPPING)
TOKEN_PATTERN = re.compile(r"([A-N])(\d+)")


class MotionLanguageDecoder:
    """Decode one-line-per-frame ``<A5B5...>`` text.

    The returned angle array has shape ``(frames, 6, 3)`` and joint order
    ``hips, neck, L_shoulder, L_elbow, R_shoulder, R_elbow``.
    """

    def __init__(self, axis_levels=None):
        self.axis_levels = {axis_id: spec.levels for axis_id, spec in AXIS_MAPPING.items()}
        if axis_levels is not None:
            unknown = set(axis_levels) - set(AXIS_IDS)
            if unknown:
                raise ValueError(f"unknown axis IDs: {sorted(unknown)}")
            self.axis_levels.update(axis_levels)
        if any(not isinstance(level, int) or level < 1 for level in self.axis_levels.values()):
            raise ValueError("axis levels must be positive integers")

    @staticmethod
    def _parse_line(line, line_number):
        text = line.strip()
        if not text.startswith("<") or not text.endswith(">"):
            raise ValueError(f"line {line_number}: expected <...>")
        body = text[1:-1]
        tokens = []
        position = 0
        while position < len(body):
            match = TOKEN_PATTERN.match(body, position)
            if match is None:
                raise ValueError(f"line {line_number}: invalid token near {body[position:]!r}")
            tokens.append((match.group(1), int(match.group(2))))
            position = match.end()
        if len({axis_id for axis_id, _ in tokens}) != len(tokens):
            raise ValueError(f"line {line_number}: duplicate axis ID")
        return tokens

    def _dequantize(self, axis_id, quantized):
        spec = AXIS_MAPPING[axis_id]
        levels = self.axis_levels[axis_id]
        if quantized < 0 or quantized >= levels:
            raise ValueError(f"axis {axis_id}: quantized value outside 0..{levels - 1}")
        if spec.mode == "absolute":
            return 180.0 * (quantized + 0.5) / levels
        return -180.0 + 360.0 * (quantized + 0.5) / levels

    def decode_lines(self, lines):
        """Decode an iterable of motion-language lines into ``(frames, 6, 3)``."""
        if isinstance(lines, str):
            lines = lines.splitlines()
        parsed_lines = [self._parse_line(line, index) for index, line in enumerate(lines, 1) if line.strip()]
        if not parsed_lines:
            raise ValueError("motion-language text contains no frames")

        first_ids = {axis_id for axis_id, _ in parsed_lines[0]}
        if first_ids != set(AXIS_IDS):
            missing = sorted(set(AXIS_IDS) - first_ids)
            extra = sorted(first_ids - set(AXIS_IDS))
            raise ValueError(f"first frame must contain all axes; missing={missing}, extra={extra}")

        state = np.zeros((6, 3), dtype=float)
        frames = []
        for tokens in parsed_lines:
            next_state = state.copy()
            for axis_id, quantized in tokens:
                spec = AXIS_MAPPING[axis_id]
                next_state[spec.joint_index, spec.angle_index] = self._dequantize(axis_id, quantized)
            state = next_state
            frames.append(state.copy())
        return np.asarray(frames, dtype=float)

    def decode_file(self, path):
        """Read and decode a manually prepared text file."""
        with open(path, encoding="utf-8") as input_file:
            return self.decode_lines(input_file)


__all__ = ["AXIS_MAPPING", "AXIS_IDS", "MotionLanguageDecoder"]
