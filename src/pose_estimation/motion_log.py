"""Persistence helpers for detected motion-language sequences."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path


def save_motion_log(
    token_lines: Iterable[str] | str,
    output_dir: str | Path,
    timestamp: datetime | None = None,
) -> Path:
    """Save one detected motion sequence as newline-delimited token frames."""
    if isinstance(token_lines, str):
        lines = [line.strip() for line in token_lines.splitlines() if line.strip()]
    else:
        lines = [line.strip() for line in token_lines if line.strip()]
    if not lines:
        raise ValueError("token_lines must contain at least one frame")
    if any(not line.startswith("<") or not line.endswith(">") for line in lines):
        raise ValueError("token_lines must contain motion-language frames")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or datetime.now()
    path = destination / f"motion_log_{stamp:%Y%m%d_%H%M%S_%f}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


__all__ = ["save_motion_log"]