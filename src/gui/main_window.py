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
    QCheckBox,
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

from pose_estimation import motion_config as m_config
from pose_estimation import motion_language as m_lang
from pose_estimation import visualize_motion as vis_motion


class MotionTestSystem(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("動作言語 ユニットテスト・可視化システム")
        self.resize(1300, 850)

        self.encoder = m_lang.MotionLanguageEncoder()
        self.decoder = m_lang.MotionLanguageDecoder()

        # 15関節の指定角度 (1, 15, 3) [Roll, Pitch, Yaw]
        self.manual_angles = np.zeros((1, len(m_config.JOINT_NAMES), 3), dtype=float)
        
        self.sequence_angles = self.manual_angles.copy()
        self.sequence_coords = vis_motion.reconstruct_3d_positions(self.sequence_angles)
        self.sequence_tokens: list[str] = []
        self.current_frame_idx = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        self._init_ui()
        self._init_3d_viewer()
        self.reset_pose(log_message=False)

    def _init_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(10, 10, 10, 10)

        # 1. 動作言語ファイル読み込み
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

        # 2. 連続再生・ステップ確認
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

        # 3D表示トグル（ローカル軸の表示/非表示）
        self.show_axes_checkbox = QCheckBox("関節のローカル座標軸を表示")
        self.show_axes_checkbox.setChecked(True)
        self.show_axes_checkbox.stateChanged.connect(self.toggle_axes_visibility)
        play_layout.addWidget(self.show_axes_checkbox)

        play_group.setLayout(play_layout)
        control_layout.addWidget(play_group)

        # 3. 関節角度指定 ➔ 動作言語エンコード
        angle_group = QGroupBox("3. 関節角度から動作言語へエンコード")
        angle_layout = QVBoxLayout()
        
        self.joint_combo = QComboBox()
        self.joint_combo.addItems(m_config.JOINT_NAMES)  # 15関節を選択
        self.joint_combo.currentIndexChanged.connect(self._sync_angle_inputs)
        angle_layout.addWidget(self.joint_combo)

        values_layout = QHBoxLayout()
        self.angle_spinboxes: list[QDoubleSpinBox] = []
        self.angle_sliders: list[QSlider] = []  # スライダーのリストを追加
        axis_names = ("Roll (deg):", "Pitch (deg):", "Yaw (deg):")
        
        for axis_name in axis_names:
            axis_layout = QVBoxLayout()
            axis_layout.addWidget(QLabel(axis_name))
            
            # スライダーの設定
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(-180, 180)  # 角度可動域 (-180° 〜 180°)
            slider.setValue(0)
            self.angle_sliders.append(slider)
            axis_layout.addWidget(slider)

            # スピンボックスの設定
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-180.0, 180.0)
            spinbox.setDecimals(1)
            spinbox.setSingleStep(1.0)
            self.angle_spinboxes.append(spinbox)
            axis_layout.addWidget(spinbox)

            # スライダーとスピンボックスの値を互いに同期させる
            slider.valueChanged.connect(lambda v, sb=spinbox: sb.setValue(v))
            spinbox.valueChanged.connect(lambda v, sl=slider: sl.setValue(int(v)))

            # ★ 追加：値が変わるたびにエンコード＆姿勢更新（ログ出力はスキップ） ★
            spinbox.valueChanged.connect(lambda _: self.encode_from_angles(log_output=False))

            values_layout.addLayout(axis_layout)
            
        angle_layout.addLayout(values_layout)

        encode_button = QPushButton("角度更新してエンコード実行")
        # ★ 変更：ボタン押下時はログを出力する
        encode_button.clicked.connect(lambda: self.encode_from_angles(log_output=True))
        angle_layout.addWidget(encode_button)

        reset_button = QPushButton("姿勢を初期状態(全関節0°)にリセット")
        reset_button.clicked.connect(self.reset_pose)
        angle_layout.addWidget(reset_button)

        angle_group.setLayout(angle_layout)
        control_layout.addWidget(angle_group)

        # 4. 動作言語から座標へデコード
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

        # 5. 計算過程ログ
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
        self.joint_meshes = []
        self.bone_meshes = []
        self.axis_actors = []  # 各関節のローカル座標軸表示用

        left_indices = {3, 4, 5, 9, 10, 11}
        right_indices = {6, 7, 8, 12, 13, 14}
        
        # 15関節の球体メッシュおよび軸矢印
        for index in range(len(m_config.JOINT_NAMES)):
            color = "red" if index in left_indices else "blue" if index in right_indices else "green"
            self.joint_meshes.append(
                self.plotter.add_mesh(pv.Sphere(radius=0.025), color=color)
            )
            
            axes = pv.Axes(show_actor=True)
            actor = axes.axes_actor
            
            # ★ここが重要: 軸の「棒の長さ」を直接指定 (X, Y, Z)
            actor.SetTotalLength(0.08, 0.08, 0.08) 
            
            # 文字ラベルを非表示
            actor.SetXAxisLabelText("")
            actor.SetYAxisLabelText("")
            actor.SetZAxisLabelText("")
            
            # 矢印の太さ・先端コーンの比率調整
            actor.SetShaftTypeToCylinder()               
            actor.SetCylinderRadius(0.1)                # 軸の太さ
            actor.SetConeRadius(0.5)                     # 矢印先端の太さ
            actor.SetNormalizedTipLength(0.3, 0.3, 0.3)  # 全長に対する矢印部分の割合
            
            added_actor = self.plotter.add_actor(actor)
            if isinstance(added_actor, tuple):
                added_actor = added_actor[0]
            self.axis_actors.append(added_actor)

        # 骨（円柱）の生成
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

        self.plotter.show_bounds(
            bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
            location="outer",
            all_edges=True,
            ticks="both",
            font_size=8,
            xtitle="X Axis",
            ytitle="Y Axis",
            ztitle="Z Axis",
            color="black",
        )
        self.plotter.add_axes(line_width=2)
        self.plotter.view_xy()
        self.plotter.camera.up = (0, 1, 0)

    def toggle_axes_visibility(self, state: int) -> None:
        """ローカル座標軸の表示/非表示を切り替え"""
        visible = bool(state)
        for actor in self.axis_actors:
            if hasattr(actor, "SetVisibility"):
                actor.SetVisibility(visible)
            elif hasattr(actor, "visibility"):
                actor.visibility = visible
        self.plotter.render()

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

    def encode_from_angles(self, log_output: bool = True) -> None:
        """指定された関節角度から動作言語へエンコードし、3D姿勢も同時に更新する"""
        # PyQtのシグナル(clicked等)から呼ばれた際の引数(checked: bool)を補正
        if not isinstance(log_output, bool):
            log_output = True

        joint_index = self.joint_combo.currentIndex()
        self.manual_angles[0, joint_index] = [
            spinbox.value() for spinbox in self.angle_spinboxes
        ]
        try:
            tokens = self.encoder.encode(self.manual_angles)
            token_text = "\n".join(tokens)
            self.language_input.setPlainText(token_text)

            # 角度から3D座標を順運動学(FK)で再構成してビューアに反映
            coords = vis_motion.reconstruct_3d_positions(self.manual_angles)
            self.sequence_angles = self.manual_angles.copy()
            self.sequence_coords = coords
            self.sequence_tokens = tokens
            self.file_label.setText("手動角度入力")
            self._set_frame(0)

            # ★ スライダー操作中はログ出力しないように分岐 ★
            if log_output:
                joint_name = m_config.JOINT_NAMES[joint_index]
                r, p, y = self.manual_angles[0, joint_index]
                self.log_message(
                    f"エンコード完了: 関節[{joint_name}] ➔ Roll:{r:.1f}°, Pitch:{p:.1f}°, Yaw:{y:.1f}°\n"
                    f"動作言語:\n{token_text}"
                )
        except (TypeError, ValueError) as error:
            if log_output:
                self.log_message(f"エンコードエラー: {error}")

    def reset_pose(self, log_message: bool = True) -> None:
        """すべての関節角度を0°にリセットし、可視化と動作言語を更新する"""
        self.manual_angles = np.zeros((1, len(m_config.JOINT_NAMES), 3), dtype=float)
        try:
            self.sequence_tokens = self.encoder.encode(self.manual_angles)
            token_text = "\n".join(self.sequence_tokens)
            self.language_input.setPlainText(token_text)
            self.sequence_angles = self.manual_angles.copy()
            self.sequence_coords = vis_motion.reconstruct_3d_positions(self.manual_angles)
            self.file_label.setText("初期姿勢 (全関節0°)")
            self._set_frame(0)
            self._sync_angle_inputs(self.joint_combo.currentIndex())
            if log_message:
                self.log_message("関節角度を初期状態 (0°) にリセットしました")
        except (TypeError, ValueError) as error:
            self.log_message(f"初期姿勢のリセットエラー: {error}")

    def decode_from_language(self) -> None:
        token_text = self.language_input.toPlainText().strip()
        if not token_text:
            self.log_message("デコードする動作言語が入力されていません")
            return
        try:
            self.sequence_tokens = [line for line in token_text.splitlines() if line.strip()]
            self.sequence_angles = self.decoder.decode_lines(self.sequence_tokens)
            self.sequence_coords = vis_motion.reconstruct_3d_positions(self.sequence_angles)
            
            # デコードされた1フレーム目の角度を手動角度入力用バッファにも反映
            if len(self.sequence_angles) > 0:
                self.manual_angles = self.sequence_angles[0:1].copy()
                self._sync_angle_inputs(self.joint_combo.currentIndex())

            self.file_label.setText("手動入力")
            self._set_frame(0)
            self.log_message(
                f"デコード完了: {len(self.sequence_coords)}フレーム\n"
                f"角度形状: {self.sequence_angles.shape}\n"
                f"座標形状: {self.sequence_coords.shape}"
            )
        except ValueError as error:
            self.log_message(f"デコードエラー: {error}")

    def _sync_angle_inputs(self, index: int) -> None:
        """選択された関節の現在の角度 [Roll, Pitch, Yaw] をスピンボックスとスライダーに同期"""
        if not hasattr(self, "angle_spinboxes"):
            return
        for spinbox, slider, value in zip(self.angle_spinboxes, self.angle_sliders, self.manual_angles[0, index]):
            # シグナルループを防ぐため一時的にイベントをブロック
            spinbox.blockSignals(True)
            slider.blockSignals(True)
            
            spinbox.setValue(float(value))
            slider.setValue(int(value))
            
            slider.blockSignals(False)
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
        self.update_3d_visualization(
            self.sequence_coords[self.current_frame_idx],
            self.sequence_angles[self.current_frame_idx],
        )

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

    def update_3d_visualization(
        self, joint_positions: np.ndarray, joint_angles: np.ndarray
    ) -> None:
        positions = np.asarray(joint_positions, dtype=float)
        if positions.shape != (len(m_config.JOINT_NAMES), 3):
            raise ValueError("joint_positions must have shape (15, 3)")
        angles = np.asarray(joint_angles, dtype=float)
        if angles.shape != (len(m_config.JOINT_NAMES), 3):
            raise ValueError("joint_angles must have shape (15, 3)")
        rotations = vis_motion.reconstruct_global_rotations(angles[np.newaxis, ...])[0]

        # 各関節の球体とミニ座標軸を更新
        for index, (actor, axis_actor, position) in enumerate(zip(self.joint_meshes, self.axis_actors, positions)):
            trans_mat = self._translation_matrix(position)
            self._set_actor_matrix(actor, trans_mat)
            axis_transform = trans_mat.copy()
            axis_transform[:3, :3] = rotations[index]
            self._set_actor_matrix(axis_actor, axis_transform)

        # 骨（円柱）を更新
        for actor, (parent_index, child_index) in zip(self.bone_meshes, self.bones_indices):
            start = positions[parent_index]
            end = positions[child_index]
            vector = end - start
            length = float(np.linalg.norm(vector))
            if length <= 1e-8:
                actor.visibility = False
                continue
            actor.visibility = True
            self._set_actor_matrix(
                actor, 
                self._cylinder_transform((start + end) / 2.0, vector / length, length)
            )
        self.plotter.render()

    @staticmethod
    def _set_actor_matrix(actor, matrix: np.ndarray) -> None:
        """PyVista / VTK アクターの型に応じて安全に行列（位置・姿勢）をセットする"""
        if isinstance(actor, tuple):
            actor = actor[0]
            
        if hasattr(actor, "user_matrix"):
            actor.user_matrix = matrix
        elif hasattr(actor, "SetUserMatrix"):
            import vtk
            vtk_mat = vtk.vtkMatrix4x4()
            for r in range(4):
                for c in range(4):
                    vtk_mat.SetElement(r, c, float(matrix[r, c]))
            actor.SetUserMatrix(vtk_mat)

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
                "Meiryo",
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