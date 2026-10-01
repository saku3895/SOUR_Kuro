#!/usr/bin/env python3
"""Visualize upper-body skeleton reconstructed from motion-language text."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pose_estimation.motion_language_decoder import MotionLanguageDecoder
from src.pose_estimation.skeleton_definition import EDGES, JOINT_NAMES
from src.pose_estimation.skeleton_fk import angles_to_positions


DEFAULT_FPS = 30
SPEED_LEVELS = [0.25, 0.5, 1.0, 1.5, 2.0, 4.0]


def create_animation(
    positions,
    fps=DEFAULT_FPS,
    manual=False,
    show_labels=True,
    elev=20.0,
    azim=-60.0,
):
    """Create an interactive Matplotlib 3D animation with joint labels, speed & frame controls."""
    positions = np.asarray(positions, dtype=float)
    total_frames = len(positions)

    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.set_title("Motion Language Reconstruction")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.set_xlim(-0.7, 0.7)
    axis.set_ylim(-0.7, 0.7)
    axis.set_zlim(-0.7, 0.7)
    axis.set_box_aspect((1, 1, 1))

    # デフォルト画角の設定
    axis.view_init(elev=elev, azim=azim)

    points = axis.scatter([], [], [], c="tab:blue", s=45)
    lines = [axis.plot([], [], [], c="tab:orange", linewidth=2)[0] for _ in EDGES]

    # 関節ID・名称ラベルテキストの初期化
    joint_texts = []
    for i, name in enumerate(JOINT_NAMES):
        t = axis.text(
            0, 0, 0, f" {i}:{name}", fontsize=8, color="darkblue", weight="bold"
        )
        t.set_visible(show_labels)
        joint_texts.append(t)

    # UIステータス情報のオーバーレイ表示
    hud_text = axis.text2D(
        0.02,
        0.88,
        "",
        transform=axis.transAxes,
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    current_frame = [0]
    is_playing = [not manual]
    speed_idx = [2]  # デフォルト 1.0x
    labels_visible = [show_labels]
    base_interval = 1000.0 / fps

    def update_hud():
        status = "PLAYING" if is_playing[0] else "PAUSED"
        speed = SPEED_LEVELS[speed_idx[0]]
        lbl_status = "ON" if labels_visible[0] else "OFF"
        text = (
            f"Frame: {current_frame[0] + 1} / {total_frames}\n"
            f"Status: {status} | Speed: {speed:.2f}x | Labels: {lbl_status}\n"
            f"-----------------------------------\n"
            f"[Space] Play/Pause | [R] Restart\n"
            f"[←/→] Step Frame   | [↑/↓] Speed\n"
            f"[L] Toggle Labels  | [V] Print View"
        )
        hud_text.set_text(text)

    def update(frame_index):
        current_frame[0] = frame_index
        frame = positions[frame_index]
        points._offsets3d = (frame[:, 0], frame[:, 1], frame[:, 2])

        # 骨のライン描画更新
        for line, (parent_index, child_index) in zip(lines, EDGES):
            line.set_data(
                [frame[parent_index, 0], frame[child_index, 0]],
                [frame[parent_index, 1], frame[child_index, 1]],
            )
            line.set_3d_properties(
                [frame[parent_index, 2], frame[child_index, 2]]
            )

        # 関節ラベルの位置更新
        for i, text_obj in enumerate(joint_texts):
            x, y, z = frame[i]
            text_obj.set_position_3d((x + 0.015, y + 0.015, z + 0.015))

        update_hud()
        return [points, *lines, *joint_texts, hud_text]

    animation = FuncAnimation(
        figure,
        update,
        frames=total_frames,
        interval=base_interval / SPEED_LEVELS[speed_idx[0]],
        blit=False,
        repeat=True,
    )

    def show_frame(frame_index):
        current_frame[0] = frame_index
        update(frame_index)
        figure.canvas.draw_idle()

    def update_timer_interval():
        if animation.event_source is not None:
            animation.event_source.interval = (
                base_interval / SPEED_LEVELS[speed_idx[0]]
            )

    def on_key(event):
        key = event.key.lower() if event.key else ""

        if key == " ":
            if is_playing[0]:
                animation.event_source.stop()
                is_playing[0] = False
            else:
                animation.event_source.start()
                is_playing[0] = True
            update_hud()
            figure.canvas.draw_idle()

        elif key in ("r", "home"):
            current_frame[0] = 0
            if not is_playing[0]:
                is_playing[0] = True
                animation.event_source.start()
            show_frame(0)

        elif key == "right":
            if is_playing[0]:
                animation.event_source.stop()
                is_playing[0] = False
            next_f = min(current_frame[0] + 1, total_frames - 1)
            show_frame(next_f)

        elif key == "left":
            if is_playing[0]:
                animation.event_source.stop()
                is_playing[0] = False
            prev_f = max(current_frame[0] - 1, 0)
            show_frame(prev_f)

        elif key == "end":
            if is_playing[0]:
                animation.event_source.stop()
                is_playing[0] = False
            show_frame(total_frames - 1)

        elif key == "up":
            if speed_idx[0] < len(SPEED_LEVELS) - 1:
                speed_idx[0] += 1
                update_timer_interval()
                update_hud()
                figure.canvas.draw_idle()

        elif key == "down":
            if speed_idx[0] > 0:
                speed_idx[0] -= 1
                update_timer_interval()
                update_hud()
                figure.canvas.draw_idle()

        elif key == "l":
            labels_visible[0] = not labels_visible[0]
            for t in joint_texts:
                t.set_visible(labels_visible[0])
            update_hud()
            figure.canvas.draw_idle()

        elif key == "v":
            print(f"\n[現在の画角設定] elev={axis.elev:.1f}, azim={axis.azim:.1f}")

        elif key == "q":
            plt.close(figure)

    figure.canvas.mpl_connect("key_press_event", on_key)
    if manual:
        animation.event_source.stop()
        is_playing[0] = False
        show_frame(0)

    return figure, animation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text_file", type=Path)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument(
        "--save", type=Path, help="Save animation instead of only displaying it"
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Start paused and inspect frames with arrow keys",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Hide joint ID labels by default",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=8.4,
        help="Initial elevation angle for 3D view",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=-161.8,
        help="Initial azimuth angle for 3D view",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Build the figure without opening a GUI"
    )
    args = parser.parse_args()

    decoder = MotionLanguageDecoder()
    angles = decoder.decode_file(args.text_file)
    positions = angles_to_positions(angles)
    figure, animation = create_animation(
        positions,
        fps=args.fps,
        manual=args.manual,
        show_labels=not args.no_labels,
        elev=args.elev,
        azim=args.azim,
    )

    print(f"decoded frames: {len(angles)}")
    print(f"angles shape: {angles.shape}")
    print(f"positions shape: {positions.shape}")
    print(f"joints: {', '.join(JOINT_NAMES)}")
    if not args.no_show:
        print(
            "操作ガイド:\n"
            "  Space: 再生/一時停止\n"
            "  R / Home: リスタート\n"
            "  ← / →: 1フレーム移動\n"
            "  ↑ / ↓: 再生速度変更 (0.25x 〜 4.0x)\n"
            "  L: 関節ID・名称ラベルの表示/非表示切替\n"
            "  V: 現在の画角(elev, azim)をターミナル出力\n"
            "  Q: 終了"
        )

    if args.save is not None:
        animation.save(
            args.save, writer="pillow" if args.save.suffix.lower() == ".gif" else None
        )
        print(f"animation saved: {args.save}")
    if not args.no_show:
        plt.show()
    else:
        animation._init_draw()
        animation._draw_was_started = True
        plt.close(figure)


if __name__ == "__main__":
    main()