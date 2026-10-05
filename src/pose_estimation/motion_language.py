"""ROM-based motion-language encoder and stateful decoder."""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np

try:
    from .motion_config import AXIS_IDS, AXIS_MAPPING
except ImportError:
    from motion_config import AXIS_IDS, AXIS_MAPPING


_TOKEN = re.compile(r"([A-Z]+)(\d+)")


class MotionLanguageEncoder:
    """Encode changed quantized axes as one token string per frame."""

    @staticmethod
    def _validate(angles: np.ndarray) -> np.ndarray:
        values = np.asarray(angles, dtype=float)
        if values.ndim != 3 or values.shape[1:] != (15, 3):
            raise ValueError("euler_angles_array must have shape (frames, 15, 3)")
        if values.shape[0] == 0 or not np.all(np.isfinite(values)):
            raise ValueError("euler_angles_array must contain finite frames")
        return values

    @staticmethod
    def _quantize(value: float, minimum: float, maximum: float, levels: int) -> int:
        clipped = np.clip(value, minimum, maximum)
        position = (clipped - minimum) / (maximum - minimum)
        return int(np.clip(np.floor(position * levels), 0, levels - 1))

    def _frame_values(self, frame: np.ndarray) -> dict[str, int]:
        result = {}
        for axis_id in AXIS_IDS:
            joint, axis, _, levels, minimum, maximum = AXIS_MAPPING[axis_id]
            result[axis_id] = self._quantize(frame[joint, axis], minimum, maximum, levels)
        return result

    def encode(self, euler_angles_array: np.ndarray) -> list[str]:
        angles = self._validate(euler_angles_array)
        previous: dict[str, int] | None = None
        lines: list[str] = []
        for frame in angles:
            current = self._frame_values(frame)
            changed = current if previous is None else {
                axis_id: value for axis_id, value in current.items() if previous[axis_id] != value
            }
            lines.append(f"<{''.join(f'{axis_id}{value}' for axis_id, value in changed.items())}>")
            previous = current
        return lines

    def encode_sequence(self, euler_angles_array: np.ndarray) -> str:
        """Compatibility helper for the live capture loop."""
        return "\n".join(self.encode(euler_angles_array))


class MotionLanguageDecoder:
    """Decode token lines while carrying unspecified axes between frames."""

    @staticmethod
    def _parse(line: str, line_number: int) -> list[tuple[str, int]]:
        text = line.strip()
        if not text.startswith("<") or not text.endswith(">"):
            raise ValueError(f"line {line_number}: expected <...>")
        body = text[1:-1]
        position = 0
        tokens = []
        while position < len(body):
            match = _TOKEN.match(body, position)
            if match is None or match.group(1) not in AXIS_MAPPING:
                raise ValueError(f"line {line_number}: invalid token near {body[position:]!r}")
            tokens.append((match.group(1), int(match.group(2))))
            position = match.end()
        if len({axis_id for axis_id, _ in tokens}) != len(tokens):
            raise ValueError(f"line {line_number}: duplicate axis ID")
        return tokens

    @staticmethod
    def _dequantize(axis_id: str, quantized: int) -> float:
        _, _, _, levels, minimum, maximum = AXIS_MAPPING[axis_id]
        if not 0 <= quantized < levels:
            raise ValueError(f"axis {axis_id}: quantized value outside 0..{levels - 1}")
        return minimum + (maximum - minimum) * (quantized + 0.5) / levels

    def decode(self, token_lines: Iterable[str] | str) -> np.ndarray:
        lines = token_lines.splitlines() if isinstance(token_lines, str) else list(token_lines)
        parsed = [self._parse(line, index) for index, line in enumerate(lines, 1) if line.strip()]
        if not parsed:
            raise ValueError("token_lines contains no frames")
        state = np.zeros((15, 3), dtype=float)
        frames = []
        for tokens in parsed:
            for axis_id, quantized in tokens:
                joint, axis, _, _, _, _ = AXIS_MAPPING[axis_id]
                state[joint, axis] = self._dequantize(axis_id, quantized)
            frames.append(state.copy())
        return np.asarray(frames, dtype=float)

    def decode_lines(self, token_lines: Iterable[str] | str) -> np.ndarray:
        return self.decode(token_lines)

    def decode_file(self, path: str) -> np.ndarray:
        with open(path, encoding="utf-8") as input_file:
            return self.decode(input_file)


__all__ = ["MotionLanguageDecoder", "MotionLanguageEncoder"]