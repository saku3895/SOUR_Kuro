from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont, QFontDatabase
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pose_estimation import calculate_joint_angles as calc_angles
from pose_estimation import motion_config as m_config
from pose_estimation import motion_language as m_lang
from pose_estimation import visualize_motion as vis_motion


def create_t_pose_coordinates() -> np.ndarray:
    """Return one COCO-17 frame used as the GUI's editable base pose."""
    return np.array(
        [
            (0.00, 0.65, 0.00),
            (-0.025, 0.69, 0.02),
            (0.025, 0.69, 0.02),
            (-0.06, 0.65, 0.00),
            (0.06, 0.65, 0.00),
            (-0.18, 0.45, 0.00),
            (0.18, 0.45, 0.00),
            (-0.45, 0.45, 0.00),
            (0.45, 0.45, 0.00),
            (-0.70, 0.45, 0.00),
            (0.70, 0.45, 0.00),
            (-0.10, 0.00, 0.00),
            (0.10, 0.00, 0.00),
            (-0.10, -0.40, 0.00),
            (0.10, -0.40, 0.00),
            (-0.10, -0.80, 0.00),
            (0.10, -0.80, 0.00),
        ],
        dtype=float,
    )[np.newaxis, ...]


class MotionTestSystem(QMainWindow):
    COCO_JOINTS = (
        "nose", "lefteye", "righteye", "leftear", "rightear",
        "leftshoulder", "rightshoulder", "leftelbow", "rightelbow",
        "leftwrist", "rightwrist", "lefthip", "righthip",
        "leftknee", "rightknee", "leftankle", "rightankle",
    )

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("動作言語 ユニットテスト・可視化システム")
        self.resize(1300, 850)
        self.encoder = m_lang.MotionLanguageEncoder()
        self.decoder = m_lang.MotionLanguageDecoder()
        self.current_coords = create_t_pose_coordinates()
        self.sequence_angles = np.zeros((1, len(m_config.JOINT_NAMES), 3), dtype=float)
        self.sequence_coords = vis_motion.reconstruct_3d_positions(self.sequence_angles)
        self.sequence_tokens: list[str] = []
        self.current_frame_idx = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self._init_ui()
        self._init_3d_viewer()
        self._set_frame(0)
        self._sync_coordinate_inputs(0)

    def _init_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(10, 10, 10, 10)

        file_group = QGroupBox("1. 動作言語ファイル")
        file_layout = QVBoxLayout()
        file_button = QPushButton("txtファイルを選択")
        file_button.clicked.connect(self.load_motion_file)
        self.file_label = QLabel("未選択")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(file_button)
        file_layout.addWidget(self.file_label)
        file_group.setLayout(file_layout)
        control_layout.addWidget(file_group)

        play_group = QGroupBox("2. 連続再生・ステップ確認")
        play_layout = QVBoxLayout()
        self.frame_label = QLabel("Frame: 0 / 0")
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setRange(0, 0)
        self.timeline_slider.valueChanged.connect(self.seek_frame)
        play_layout.addWidget(self.frame_label)
        play_layout.addWidget(self.timeline_slider)
        button_layout = QHBoxLayout()
        previous_button = QPushButton("◀ コマ戻し")
        previous_button.clicked.connect(self.prev_frame)
        self.play_button = QPushButton("▶ 再生")
        self.play_button.clicked.connect(self.toggle_playback)
        next_button = QPushButton("コマ送り ▶")
        next_button.clicked.connect(self.next_frame)
        button_layout.addWidget(previous_button)
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(next_button)
        play_layout.addLayout(button_layout)
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("再生速度 (ms/frame):"))
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(10, 1000)
        self.speed_spin.setValue(100)
        self.speed_spin.setDecimals(0)
        self.speed_spin.valueChanged.connect(self._update_timer_interval)
        speed_layout.addWidget(self.speed_spin)
        play_layout.addLayout(speed_layout)
        play_group.setLayout(play_layout)
        control_layout.addWidget(play_group)

        coordinate_group = QGroupBox("3. 座標から動作言語へエンコード")
        coordinate_layout = QVBoxLayout()
        self.joint_combo = QComboBox()
        self.joint_combo.addItems(self.COCO_JOINTS)
        self.joint_combo.currentIndexChanged.connect(self._sync_coordinate_inputs)
        coordinate_layout.addWidget(self.joint_combo)
        values_layout = QHBoxLayout()
        self.coordinate_spinboxes: list[QDoubleSpinBox] = []
        for axis_name in ("X", "Y", "Z"):
            values_layout.addWidget(QLabel(f"{axis_name}:"))
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-1000.0, 1000.0)
            spinbox.setDecimals(4)
            spinbox.setSingleStep(0.1)
            self.coordinate_spinboxes.append(spinbox)
            values_layout.addWidget(spinbox)
        coordinate_layout.addLayout(values_layout)
        encode_button = QPushButton("座標更新してエンコード実行")
        encode_button.clicked.connect(self.encode_from_coords)
        coordinate_layout.addWidget(encode_button)
        coordinate_group.setLayout(coordinate_layout)
        control_layout.addWidget(coordinate_group)

        language_group = QGroupBox("4. 動作言語から座標へデコード")
        language_layout = QVBoxLayout()
        self.language_input = QTextEdit()
        self.language_input.setPlaceholderText("例: <A0B4C6...>")
        self.language_input.setFixedHeight(90)
        language_layout.addWidget(self.language_input)
        decode_button = QPushButton("デコード実行と姿勢更新")
        decode_button.clicked.connect(self.decode_from_language)
        language_layout.addWidget(decode_button)
        language_group.setLayout(language_layout)
        control_layout.addWidget(language_group)

        log_group = QGroupBox("5. 計算過程ログ")
        log_layout = QVBoxLayout()
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        log_layout.addWidget(self.log_output)
        log_group.setLayout(log_layout)
        control_layout.addWidget(log_group, stretch=1)
        main_layout.addLayout(control_layout, stretch=1)

        self.plotter = QtInteractor(self)
        main_layout.addWidget(self.plotter.interactor, stretch=2)

    def _init_3d_viewer(self) -> None:
        self.plotter.show_grid()
        self.plotter.add_axes()
        self.plotter.view_xy()
        self.plotter.camera.up = (0, 1, 0)
        self.joint_meshes = []
        self.bone_meshes = []
        left_indices = {3, 4, 5, 9, 10, 11}
        right_indices = {6, 7, 8, 12, 13, 14}
        for index in range(len(m_config.JOINT_NAMES)):
            color = "red" if index in left_indices else "blue" if index in right_indices else "green"
            self.joint_meshes.append(
                self.plotter.add_mesh(pv.Sphere(radius=0.03), color=color)
            )
        self.bones_indices = [
            (m_config.JOINT_INDEX[parent], m_config.JOINT_INDEX[child])
            for child, parent in m_config.PARENTS.items()
            if parent is not None
        ]
        for _ in self.bones_indices:
            self.bone_meshes.append(
                self.plotter.add_mesh(
                    pv.Cylinder(
                        center=(0, 0, 0),
                        direction=(0, 1, 0),
                        radius=0.01,
                        height=1.0,
                    ),
                    color="gray",
                )
            )

    def load_motion_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "動作言語ファイルを選択", str(Path.cwd()),
            "Text files (*.txt);;All files (*)",
        )
        if not file_path:
            return
        try:
            path = Path(file_path)
            self.sequence_tokens = [
                line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.sequence_angles = self.decoder.decode_lines(self.sequence_tokens)
            self.sequence_coords = vis_motion.reconstruct_3d_positions(self.sequence_angles)
            self.file_label.setText(path.name)
            self._set_frame(0)
            self.log_message(
                f"読み込み完了: {path}\n"
                f"フレーム数: {len(self.sequence_coords)}\n"
                f"角度形状: {self.sequence_angles.shape}\n"
                f"座標形状: {self.sequence_coords.shape}"
            )
        except (OSError, ValueError) as error:
            self.log_message(f"ファイル読み込みエラー: {error}")

    def encode_from_coords(self) -> None:
        joint_index = self.joint_combo.currentIndex()
        self.current_coords[0, joint_index] = [
            spinbox.value() for spinbox in self.coordinate_spinboxes
        ]
        try:
            angles = calc_angles.calculate_joint_angles(self.current_coords)
            tokens = self.encoder.encode(angles)
            token_text = "\n".join(tokens)
            self.language_input.setPlainText(token_text)
            self.log_message(
                f"エンコード完了: {self.COCO_JOINTS[joint_index]} を更新\n"
                f"Euler角形状: {angles.shape}\n動作言語:\n{token_text}"
            )
        except (TypeError, ValueError) as error:
            self.log_message(f"エンコードエラー: {error}")

    def decode_from_language(self) -> None:
        token_text = self.language_input.toPlainText().strip()
        if not token_text:
            self.log_message("デコードする動作言語が入力されていません")
            return
        try:
            self.sequence_tokens = [line for line in token_text.splitlines() if line.strip()]
            self.sequence_angles = self.decoder.decode_lines(self.sequence_tokens)
            self.sequence_coords = vis_motion.reconstruct_3d_positions(self.sequence_angles)
            self.file_label.setText("手動入力")
            self._set_frame(0)
            self.log_message(
                f"デコード完了: {len(self.sequence_coords)}フレーム\n"
                f"角度形状: {self.sequence_angles.shape}\n"
                f"座標形状: {self.sequence_coords.shape}"
            )
        except ValueError as error:
            self.log_message(f"デコードエラー: {error}")

    def _sync_coordinate_inputs(self, index: int) -> None:
        if not hasattr(self, "coordinate_spinboxes"):
            return
        for spinbox, value in zip(self.coordinate_spinboxes, self.current_coords[0, index]):
            spinbox.blockSignals(True)
            spinbox.setValue(float(value))
            spinbox.blockSignals(False)

    def _set_frame(self, frame_index: int) -> None:
        if self.sequence_coords is None or len(self.sequence_coords) == 0:
            self.current_frame_idx = 0
            self.timeline_slider.setRange(0, 0)
            self.frame_label.setText("Frame: 0 / 0")
            return
        last_index = len(self.sequence_coords) - 1
        self.current_frame_idx = max(0, min(frame_index, last_index))
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setRange(0, last_index)
        self.timeline_slider.setValue(self.current_frame_idx)
        self.timeline_slider.blockSignals(False)
        self.frame_label.setText(
            f"Frame: {self.current_frame_idx + 1} / {len(self.sequence_coords)}"
        )
        self.update_3d_visualization(self.sequence_coords[self.current_frame_idx])

    def seek_frame(self, frame_index: int) -> None:
        self._set_frame(frame_index)

    def next_frame(self) -> None:
        if self.sequence_coords is None:
            self.timer.stop()
            self.play_button.setText("▶ 再生")
            self.log_message("再生するシーケンスがありません")
            return
        if self.current_frame_idx >= len(self.sequence_coords) - 1:
            self.timer.stop()
            self.play_button.setText("▶ 再生")
            return
        self._set_frame(self.current_frame_idx + 1)

    def prev_frame(self) -> None:
        if self.sequence_coords is not None:
            self._set_frame(self.current_frame_idx - 1)

    def toggle_playback(self) -> None:
        if self.sequence_coords is None:
            self.log_message("再生するシーケンスがありません")
            return
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("▶ 再生")
            self.log_message("一時停止")
            return
        if self.current_frame_idx >= len(self.sequence_coords) - 1:
            self._set_frame(0)
        self._update_timer_interval()
        self.timer.start()
        self.play_button.setText("■ 一時停止")
        self.log_message(f"再生開始: {int(self.speed_spin.value())} ms/frame")

    def _update_timer_interval(self) -> None:
        self.timer.setInterval(int(self.speed_spin.value()))

    def update_3d_visualization(self, joint_positions: np.ndarray) -> None:
        positions = np.asarray(joint_positions, dtype=float)
        if positions.shape != (len(m_config.JOINT_NAMES), 3):
            raise ValueError("joint_positions must have shape (15, 3)")
        for actor, position in zip(self.joint_meshes, positions):
            actor.user_matrix = self._translation_matrix(position)
        for actor, (parent_index, child_index) in zip(self.bone_meshes, self.bones_indices):
            start = positions[parent_index]
            end = positions[child_index]
            vector = end - start
            length = float(np.linalg.norm(vector))
            if length <= 1e-8:
                actor.visibility = False
                continue
            actor.visibility = True
            actor.user_matrix = self._cylinder_transform(
                (start + end) / 2.0, vector / length, length
            )
        self.plotter.render()

    @staticmethod
    def _translation_matrix(position: np.ndarray) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, 3] = position
        return matrix

    @staticmethod
    def _cylinder_transform(center: np.ndarray, direction: np.ndarray, length: float) -> np.ndarray:
        source_axis = np.array([0.0, 1.0, 0.0])
        cross = np.cross(source_axis, direction)
        dot = float(np.dot(source_axis, direction))
        cross_length = float(np.linalg.norm(cross))
        rotation = np.eye(3)
        if cross_length > 1e-8:
            skew = np.array([
                [0.0, -cross[2], cross[1]],
                [cross[2], 0.0, -cross[0]],
                [-cross[1], cross[0], 0.0],
            ])
            rotation = np.eye(3) + skew + skew @ skew * ((1.0 - dot) / cross_length**2)
        elif dot < 0:
            rotation = np.diag([1.0, -1.0, -1.0])
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = rotation
        scale_matrix = np.diag([1.0, length, 1.0, 1.0])
        transform = rotation_matrix @ scale_matrix
        transform[:3, 3] = center
        return transform

    def log_message(self, message: str) -> None:
        self.log_output.append(message)
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.plotter.close()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    available_fonts = set(QFontDatabase.families())
    japanese_font = next(
        (
            family
            for family in (
                "Noto Sans CJK JP",
                "Noto Sans JP",
                "IPAGothic",
                "Yu Gothic",
            )
            if family in available_fonts
        ),
        None,
    )
    if japanese_font is not None:
        app.setFont(QFont(japanese_font))
    else:
        print("Warning: no Japanese-capable Qt font found; install fonts-noto-cjk.")
    window = MotionTestSystem()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
