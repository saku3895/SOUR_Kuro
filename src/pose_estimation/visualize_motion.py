"""Reconstruct and visualize the configured 15-joint skeleton."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

try:
    from .motion_config import JOINT_INDEX, JOINT_NAMES, PARENTS
    from .motion_language import MotionLanguageDecoder
except ImportError:
    from motion_config import JOINT_INDEX, JOINT_NAMES, PARENTS
    from motion_language import MotionLanguageDecoder


DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "data_logs" / "motion_language" / "motion_language.txt"
DEFAULT_FPS = 30
AXIS_LIMITS = (-1.0, 1.0)

# Vectors are expressed in the local frame of each parent, in metres.
BONE_VECTORS = {
    ("hips", "neck"): np.array([0.0, 0.30, 0.0]),
    ("neck", "head"): np.array([0.0, 0.18, 0.0]),
    ("neck", "L_shoulder"): np.array([-0.18, 0.0, 0.0]),
    ("L_shoulder", "L_elbow"): np.array([-0.27, 0.0, 0.0]),
    ("L_elbow", "L_wrist"): np.array([-0.23, 0.0, 0.0]),
    ("neck", "R_shoulder"): np.array([0.18, 0.0, 0.0]),
    ("R_shoulder", "R_elbow"): np.array([0.27, 0.0, 0.0]),
    ("R_elbow", "R_wrist"): np.array([0.23, 0.0, 0.0]),
    ("hips", "L_hip"): np.array([-0.11, -0.05, 0.0]),
    ("L_hip", "L_knee"): np.array([0.0, -0.40, 0.0]),
    ("L_knee", "L_ankle"): np.array([0.0, -0.40, 0.0]),
    ("hips", "R_hip"): np.array([0.11, -0.05, 0.0]),
    ("R_hip", "R_knee"): np.array([0.0, -0.40, 0.0]),
    ("R_knee", "R_ankle"): np.array([0.0, -0.40, 0.0]),
}


def _rotation_matrix(angles: np.ndarray) -> np.ndarray:
    """Return ``Rz @ Rx @ Ry`` for [roll, pitch, yaw] degree angles."""
    roll, pitch, yaw = np.radians(angles)
    cz, sz = np.cos(roll), np.sin(roll)
    cx, sx = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    return rz @ rx @ ry


def reconstruct_3d_positions(euler_angles: np.ndarray) -> np.ndarray:
    """Reconstruct ``(frames, 15, 3)`` positions with hips at the origin."""
    angles = np.asarray(euler_angles, dtype=float)
    expected_shape = (len(JOINT_NAMES), 3)
    if angles.ndim != 3 or angles.shape[1:] != expected_shape:
        raise ValueError(f"euler_angles must have shape (frames, {expected_shape[0]}, 3)")
    if angles.shape[0] == 0 or not np.all(np.isfinite(angles)):
        raise ValueError("euler_angles must contain at least one finite frame")

    positions = np.zeros_like(angles)
    for frame_index, frame_angles in enumerate(angles):
        global_rotations = {
            "hips": _rotation_matrix(frame_angles[JOINT_INDEX["hips"]])
        }
        for child_name in JOINT_NAMES[1:]:
            parent_name = PARENTS[child_name]
            if parent_name is None:
                raise ValueError(f"joint {child_name!r} has no parent")
            bone = BONE_VECTORS[(parent_name, child_name)]
            parent_index = JOINT_INDEX[parent_name]
            child_index = JOINT_INDEX[child_name]
            parent_rotation = global_rotations[parent_name]
            positions[frame_index, child_index] = (
                positions[frame_index, parent_index] + parent_rotation @ bone
            )
            global_rotations[child_name] = (
                parent_rotation @ _rotation_matrix(frame_angles[child_index])
            )
    return positions


def animate_motion(
    positions_3d: np.ndarray,
    fps: int = DEFAULT_FPS,
    save_path: str | Path | None = None,
    show: bool = True,
) -> FuncAnimation:
    """Create, optionally save, and optionally display a 3D stick-figure animation."""
    positions = np.asarray(positions_3d, dtype=float)
    if positions.ndim != 3 or positions.shape[1:] != (len(JOINT_NAMES), 3):
        raise ValueError(f"positions_3d must have shape (frames, {len(JOINT_NAMES)}, 3)")
    if positions.shape[0] == 0 or not np.all(np.isfinite(positions)):
        raise ValueError("positions_3d must contain at least one finite frame")
    if fps <= 0:
        raise ValueError("fps must be positive")

    figure = plt.figure(figsize=(8, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.set_xlim(*AXIS_LIMITS)
    axis.set_ylim(*AXIS_LIMITS)
    axis.set_zlim(*AXIS_LIMITS)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=90.0, azim=-90.0)

    points = axis.scatter([], [], [], color="tab:blue", s=35)
    connections = [
        (JOINT_INDEX[parent], JOINT_INDEX[child])
        for child, parent in PARENTS.items()
        if parent is not None
    ]
    lines = [axis.plot([], [], [], color="tab:orange", linewidth=2.5)[0] for _ in connections]
    current_frame = [0]
    playback_speed = [1.0]
    paused = [False]

    def update(frame_index: int):
        current_frame[0] = frame_index
        frame = positions[frame_index]
        points._offsets3d = (frame[:, 0], frame[:, 1], frame[:, 2])
        for line, (parent_index, child_index) in zip(lines, connections):
            line.set_data(frame[[parent_index, child_index], 0], frame[[parent_index, child_index], 1])
            line.set_3d_properties(frame[[parent_index, child_index], 2])
        axis.set_title(f"Motion reconstruction: frame {frame_index + 1}/{len(positions)}")
        return [points, *lines]

    animation = FuncAnimation(
        figure,
        update,
        frames=len(positions),
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )

    def set_frame(frame_index: int):
        current_frame[0] = frame_index % len(positions)
        update(current_frame[0])
        figure.canvas.draw_idle()

    def set_speed(speed: float):
        playback_speed[0] = min(2.0, max(0.125, speed))
        animation.event_source.interval = 1000 / (fps * playback_speed[0])
        axis.set_title(
            f"Motion reconstruction: frame {current_frame[0] + 1}/{len(positions)} "
            f"({playback_speed[0]:g}x)"
        )
        figure.canvas.draw_idle()

    def on_key(event):
        if event.key == " ":
            paused[0] = not paused[0]
            if paused[0]:
                animation.event_source.stop()
            else:
                animation.event_source.start()
        elif event.key in ("right", "left"):
            paused[0] = True
            animation.event_source.stop()
            direction = 1 if event.key == "right" else -1
            set_frame(current_frame[0] + direction)
        elif event.key in ("[", "down"):
            set_speed(playback_speed[0] / 2)
        elif event.key in ("]", "up"):
            set_speed(playback_speed[0] * 2)

    figure.canvas.mpl_connect("key_press_event", on_key)
    set_frame(0)
    set_speed(playback_speed[0])
    if save_path is not None:
        output_path = Path(save_path)
        writer = "pillow" if output_path.suffix.lower() == ".gif" else None
        animation.save(output_path, writer=writer, fps=fps)
    if show:
        plt.show()
    return animation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    angles = MotionLanguageDecoder().decode_file(str(args.input))
    positions = reconstruct_3d_positions(angles)
    print(f"angles shape: {angles.shape}")
    print(f"positions shape: {positions.shape}")
    animate_motion(positions, fps=args.fps, save_path=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()


__all__ = ["BONE_VECTORS", "animate_motion", "reconstruct_3d_positions"]