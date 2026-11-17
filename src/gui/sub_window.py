from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from pose_estimation import calculate_joint_angles
from pose_estimation import motion_config as m_config
from pose_estimation import visualize_motion
from pose_estimation.keypoint_json import load_keypoints_json


DEFAULT_POSE_PATH = Path(__file__).resolve().parents[2] / "data_logs" / "posture" / "t_pose.json"
POSE_FILES = {
    "Tポーズ": DEFAULT_POSE_PATH,
    "しゃがみ姿勢": DEFAULT_POSE_PATH.parent / "squat.json",
    "左腕挙上": DEFAULT_POSE_PATH.parent / "raised_left_arm.json",
}

COCO_BONES = (
    (0, 1), (1, 3), (0, 2), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)

# Local axes that represent the modeled anatomical degrees of freedom.
# Axis indices correspond to the columns of each global rotation matrix.
LOCAL_AXIS_INDICES = {
    "hips": (0, 1, 2),
    "neck": (0, 1, 2),
    "L_shoulder": (0, 1, 2),
    "R_shoulder": (0, 1, 2),
    "L_elbow": (0, 1),
    "R_elbow": (0, 1),
    "L_hip": (0, 1, 2),
    "R_hip": (0, 1, 2),
    "L_knee": (0,),
    "R_knee": (0,),
}

class TPoseWindow(QMainWindow):
    """Display the 17-point input pose and its 15-joint reconstruction."""

    def __init__(self, pose_path: str | Path = DEFAULT_POSE_PATH) -> None:
        super().__init__()
        self.setWindowTitle("Tポーズ可視化")
        self.resize(1200, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.plotters: list[QtInteractor] = []
        self.pose_selector = QComboBox()
        self.pose_selector.addItems(POSE_FILES)
        default_name = next(
            (name for name, path in POSE_FILES.items() if path == Path(pose_path)),
            "Tポーズ",
        )
        self.pose_selector.setCurrentText(default_name)
        self.pose_selector.currentTextChanged.connect(self._show_pose)

        self.main_layout = QVBoxLayout(central_widget)
        self.main_layout.setContentsMargins(8, 8, 8, 8)
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("姿勢:"))
        selector_layout.addWidget(self.pose_selector)
        selector_layout.addStretch()
        self.main_layout.addLayout(selector_layout)
        self.views_layout = QHBoxLayout()
        self.main_layout.addLayout(self.views_layout)

        self._show_pose(default_name)

    def _show_pose(self, pose_name: str) -> None:
        """Load and display the selected posture in both skeleton views."""
        for plotter in self.plotters:
            plotter.close()
        self.plotters.clear()
        while self.views_layout.count():
            item = self.views_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
            elif item.layout() is not None:
                self._delete_layout(item.layout())

        _, _, keypoints = load_keypoints_json(POSE_FILES[pose_name])
        angles = calculate_joint_angles.calculate_joint_angles(keypoints[np.newaxis, ...])

        self.views_layout.addLayout(self._create_view("17関節（入力JSON）", keypoints, COCO_BONES))
        self.views_layout.addLayout(self._create_angle_view(angles[0]))

    @staticmethod
    def _delete_layout(layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
            elif item.layout() is not None:
                TPoseWindow._delete_layout(item.layout())

    def _create_view(
        self,
        title: str,
        points: np.ndarray,
        bones: tuple[tuple[int, int], ...],
        rotations: np.ndarray | None = None,
    ) -> QVBoxLayout:
        plotter = QtInteractor(self)
        self.plotters.append(plotter)
        plotter.set_background("white")
        plotter.add_mesh(
            pv.PolyData(points),
            color="tab:blue",
            point_size=14,
            render_points_as_spheres=True,
        )
        plotter.add_mesh(
            self._line_mesh(points, bones),
            color="tab:orange",
            line_width=4,
        )
        if rotations is not None:
            self._add_local_axes(plotter, points, rotations)
        plotter.add_axes(line_width=2)
        plotter.show_bounds(
            bounds=self._bounds(points),
            location="outer",
            all_edges=True,
            ticks="both",
            font_size=8,
            xtitle="X",
            ytitle="Y",
            ztitle="Z",
            color="black",
        )
        plotter.reset_camera()
        plotter.camera.position = (0.0, 0.0, -1.0)
        plotter.camera.up = (0.0, -1.0, 0.0)

        view_layout = QVBoxLayout()
        view_layout.addWidget(QLabel(title))
        view_layout.addWidget(plotter.interactor)
        return view_layout

    def _create_angle_view(self, angles: np.ndarray) -> QVBoxLayout:
        """Create the 15-joint view from joint angles, not input coordinates."""
        if angles.shape != (len(m_config.JOINT_NAMES), 3):
            raise ValueError("angles must have shape (15, 3)")

        # Coordinates are only a rendering result. The pose itself is defined
        # by the joint-angle combination passed to this method.
        joint_positions = visualize_motion.reconstruct_3d_positions(angles[np.newaxis, ...])[0]
        joint_rotations = visualize_motion.reconstruct_global_rotations(
            angles[np.newaxis, ...]
        )[0]
        view_layout = self._create_view(
            "15関節（関節角度から再構成）",
            joint_positions,
            self._motion_bones(),
            joint_rotations,
        )
        angle_label = QLabel(self._format_angles(angles))
        angle_label.setWordWrap(True)
        view_layout.insertWidget(1, angle_label)
        return view_layout

    @staticmethod
    def _format_angles(angles: np.ndarray) -> str:
        values = []
        for joint_name, (roll, pitch, yaw) in zip(m_config.JOINT_NAMES, angles):
            values.append(
                f"{joint_name}: X={pitch:.1f}°, Y={yaw:.1f}°, Z={roll:.1f}°"
            )
        return "  ".join(values)

    @staticmethod
    def _add_local_axes(
        plotter: QtInteractor,
        points: np.ndarray,
        rotations: np.ndarray,
    ) -> None:
        """Add only the local axes allowed by each joint's modeled degrees of freedom."""
        if rotations.shape != (len(m_config.JOINT_NAMES), 3, 3):
            raise ValueError("rotations must have shape (15, 3, 3)")

        axis_colors = ("red", "green", "blue")
        axis_length = 0.08
        for joint_index, (position, rotation) in enumerate(zip(points, rotations)):
            joint_name = m_config.JOINT_NAMES[joint_index]
            for axis_index in LOCAL_AXIS_INDICES.get(joint_name, ()):
                plotter.add_mesh(
                    pv.Arrow(
                        start=position,
                        direction=rotation[:, axis_index],
                        scale=axis_length,
                        tip_length=0.25,
                        tip_radius=0.08,
                        shaft_radius=0.025,
                    ),
                    color=axis_colors[axis_index],
                )

    @staticmethod
    def _line_mesh(points: np.ndarray, bones: tuple[tuple[int, int], ...]) -> pv.PolyData:
        lines = np.asarray(
            [[2, parent, child] for parent, child in bones],
            dtype=np.int64,
        ).reshape(-1)
        return pv.PolyData(points, lines=lines)

    @staticmethod
    def _bounds(points: np.ndarray) -> tuple[float, float, float, float, float, float]:
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        center = (minimum + maximum) / 2.0
        radius = max(float(np.max(maximum - minimum)) * 0.65, 0.1)
        return (
            center[0] - radius, center[0] + radius,
            center[1] - radius, center[1] + radius,
            center[2] - radius, center[2] + radius,
        )

    @staticmethod
    def _motion_bones() -> tuple[tuple[int, int], ...]:
        return tuple(
            (m_config.JOINT_INDEX[parent], m_config.JOINT_INDEX[child])
            for child, parent in m_config.PARENTS.items()
            if parent is not None
        )

    def closeEvent(self, event) -> None:
        for plotter in self.plotters:
            plotter.close()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = TPoseWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
