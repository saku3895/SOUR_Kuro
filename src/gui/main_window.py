from __future__ import annotations

import json
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
    QMessageBox,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pose_estimation import motion_config as m_config
from pose_estimation import motion_language as m_lang
from pose_estimation import calculate_joint_angles as calc_angles
from gui.keypoint_json import load_keypoints_json
from gui import visualize_motion as vis_motion


DEFAULT_POSE_PATH = Path(__file__).resolve().parents[2] / "data" / "posture" / "t_pose.json"
INITIAL_CAMERA_POSITION = (0.0, 0.0, -7.0)
INITIAL_CAMERA_FOCAL_POINT = (0.0, 0.0, 0.0)
INITIAL_CAMERA_UP = (0.0, -1.0, 0.0)


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
        self.sequence_keypoints: np.ndarray | None = None
        self.input_mode = "angles"
        self.current_frame_idx = 0
        self.selected_joint_index = 0

        # 手動姿勢変更の履歴管理（Undo / Redo）
        self.history_stack: list[np.ndarray] = []
        self.redo_stack: list[np.ndarray] = []
        self._is_history_action = False
        self._pre_edit_angles: np.ndarray | None = None
        self._edit_debounce_timer = QTimer(self)
        self._edit_debounce_timer.setSingleShot(True)
        self._edit_debounce_timer.setInterval(400)  # 400ms操作が途切れたら1つの姿勢変更として確定
        self._edit_debounce_timer.timeout.connect(self._commit_history_snapshot)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        self._init_ui()
        self._init_3d_viewer()
        self._load_pose_json(DEFAULT_POSE_PATH, log_message=False)

    @staticmethod
    def _t_pose_angles() -> np.ndarray:
        """Return relative angles for the neutral T-pose in the Y-bone model."""
        angles = np.zeros((len(m_config.JOINT_NAMES), 3), dtype=float)
        # angles[:, 2] = 90.0
        angles[m_config.JOINT_INDEX["hips"]] = (0.0, 0.0, 0.0)
        # 腕を横（X軸方向）に上げるための純粋なRoll（Z軸）回転のみを設定
        angles[m_config.JOINT_INDEX["L_shoulder"]] = (-90.0, 0.0, 0.0)
        angles[m_config.JOINT_INDEX["R_shoulder"]] = (90.0, 0.0, 0.0)
        # 脚は初期状態で既に下（Y軸方向）を向いているため、0で正常
        angles[m_config.JOINT_INDEX["L_hip"]] = (0.0, 0.0, 0.0)
        angles[m_config.JOINT_INDEX["R_hip"]] = (0.0, 0.0, 0.0)
        return angles

    def _init_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(10, 10, 10, 10)

        # 動作座標ファイル読み込み
        file_group = QGroupBox("動作座標ファイル")
        file_layout = QVBoxLayout()
        load_button = QPushButton("17点座標ファイル (.npy / .json) を選択")
        load_button.clicked.connect(self.load_coordinate_file)
        self.file_label = QLabel("未選択")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(load_button)
        file_layout.addWidget(self.file_label)
        file_group.setLayout(file_layout)
        control_layout.addWidget(file_group)

        # 連続再生・ステップ確認
        play_group = QGroupBox("連続再生・ステップ確認")
        play_layout = QVBoxLayout()
        self.frame_label = QLabel("Frame: 0 / 0")
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setRange(0, 0)
        self.timeline_slider.valueChanged.connect(self.seek_frame)
        play_layout.addWidget(self.frame_label)
        play_layout.addWidget(self.timeline_slider)
        
        button_layout = QHBoxLayout()
        undo_button = QPushButton("↶ 元に戻す")
        undo_button.setToolTip("1つ前の姿勢状態に戻る")
        undo_button.clicked.connect(self.undo_state)
        previous_button = QPushButton("◀ コマ戻し")
        previous_button.clicked.connect(self.prev_frame)
        self.play_button = QPushButton("▶ 再生")
        self.play_button.clicked.connect(self.toggle_playback)
        next_button = QPushButton("コマ送り ▶")
        next_button.clicked.connect(self.next_frame)
        redo_button = QPushButton("進む ↷")
        redo_button.setToolTip("戻した姿勢状態を進める")
        redo_button.clicked.connect(self.redo_state)
        button_layout.addWidget(undo_button)
        button_layout.addWidget(previous_button)
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(next_button)
        button_layout.addWidget(redo_button)
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

        # 関節角度から動作言語へエンコード
        self.angle_group = QGroupBox("関節角度から動作言語へエンコード")
        angle_layout = QVBoxLayout()
        
        # 選択関節表示ラベル（3Dビューでのクリック選択と連動）
        joint_info_layout = QHBoxLayout()
        joint_info_layout.addWidget(QLabel("選択中関節:"))
        self.joint_label = QLabel("hips (index: 0)")
        self.joint_label.setStyleSheet("font-weight: bold; color: #1a73e8;")
        joint_info_layout.addWidget(self.joint_label)
        joint_info_layout.addStretch()
        angle_layout.addLayout(joint_info_layout)

        values_layout = QHBoxLayout()
        self.angle_spinboxes: list[QDoubleSpinBox] = []
        self.angle_sliders: list[QSlider] = []
        axis_names = ("Roll (deg):", "Pitch (deg):", "Yaw (deg):")
        
        for axis_name in axis_names:
            axis_layout = QVBoxLayout()
            axis_layout.addWidget(QLabel(axis_name))
            
            # スライダーの設定
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(-180, 180)
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

            # スライダー操作開始・終了シグナル
            slider.sliderPressed.connect(self._on_slider_pressed)
            slider.sliderReleased.connect(self._on_slider_released)

            # スライダー/スピンボックスの値変化でリアルタイムにエンコード＆姿勢更新
            spinbox.valueChanged.connect(lambda _: self.encode_from_angles(log_output=False))

            values_layout.addLayout(axis_layout)
            
        angle_layout.addLayout(values_layout)

        encode_button = QPushButton("角度更新してエンコード実行")
        encode_button.clicked.connect(lambda: self.encode_from_angles(log_output=True))
        angle_layout.addWidget(encode_button)

        save_gt_button = QPushButton("テスト用正解データとして保存")
        save_gt_button.clicked.connect(self.save_ground_truth_data)
        angle_layout.addWidget(save_gt_button)

        self.angle_group.setLayout(angle_layout)
        control_layout.addWidget(self.angle_group)

        # 動作言語から座標へデコード
        language_group = QGroupBox("動作言語から座標へデコード")
        language_layout = QVBoxLayout()
        self.language_input = QTextEdit()
        self.language_input.setPlaceholderText("例: <A0B4C6...>")
        self.language_input.setFixedHeight(90)
        language_layout.addWidget(self.language_input)
        self.decode_button = QPushButton("デコード実行と姿勢更新")
        self.decode_button.clicked.connect(self.decode_from_language)
        language_layout.addWidget(self.decode_button)
        language_group.setLayout(language_layout)
        control_layout.addWidget(language_group)

        main_layout.addLayout(control_layout, stretch=1)

        views_layout = QHBoxLayout()
        keypoint_view_layout = QVBoxLayout()
        keypoint_view_layout.addWidget(QLabel("17関節（入力）"))
        self.keypoint_plotter = QtInteractor(self)
        keypoint_view_layout.addWidget(self.keypoint_plotter.interactor)
        views_layout.addLayout(keypoint_view_layout)

        joint_view_layout = QVBoxLayout()
        joint_view_header = QHBoxLayout()
        joint_view_header.addWidget(QLabel("15関節（再構成）"))
        joint_view_header.addStretch()
        reset_pose_button = QPushButton("初期状態(Tポーズ)にリセット")
        reset_pose_button.clicked.connect(self.reset_pose)
        joint_view_header.addWidget(reset_pose_button)
        reset_camera_button = QPushButton("カメラ視点をリセット")
        reset_camera_button.clicked.connect(self._reset_camera_views)
        joint_view_header.addWidget(reset_camera_button)
        joint_view_layout.addLayout(joint_view_header)
        self.plotter = QtInteractor(self)
        joint_view_layout.addWidget(self.plotter.interactor)
        views_layout.addLayout(joint_view_layout)
        main_layout.addLayout(views_layout, stretch=3)

    def _init_3d_viewer(self) -> None:
        self.joint_meshes = []
        self.bone_meshes = []
        self.axis_actors = []  # 各関節のローカル座標軸表示用
        self.keypoint_meshes = []
        self.keypoint_bone_meshes = []

        self.left_indices = {3, 4, 5, 9, 10, 11}
        self.right_indices = {6, 7, 8, 12, 13, 14}
        
        # 15関節の球体メッシュおよび軸矢印
        for index in range(len(m_config.JOINT_NAMES)):
            color = "red" if index in self.left_indices else "blue" if index in self.right_indices else "green"
            self.joint_meshes.append(
                self.plotter.add_mesh(pv.Sphere(radius=0.03), color=color, pickable=True)
            )
            
            for axis_index, color in self._local_axis_specs(index):
                actor = self.plotter.add_mesh(
                    pv.Arrow(
                        start=(0.0, 0.0, 0.0),
                        direction=np.eye(3)[:, axis_index],
                        scale=0.08,
                        tip_length=0.25,
                        tip_radius=0.08,
                        shaft_radius=0.025,
                    ),
                    color=color,
                    pickable=False,
                )
                self.axis_actors.append((index, axis_index, actor))

        # 3Dビュー上での関節ピッキング（クリックによる選択）を有効化
        self.plotter.enable_mesh_picking(
            callback=self._on_joint_picked,
            left_clicking=True,
            use_actor=True,
            show=False,
            show_message=False,
            style=None,
        )
        if hasattr(self.plotter, "picker") and hasattr(self.plotter.picker, "SetTolerance"):
            self.plotter.picker.SetTolerance(0.035)

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
                    pickable=False,
                )
            )

        self.keypoint_bones_indices = [
            (0, 1), (1, 3), (0, 2), (2, 4),
            (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16),
        ]
        for index in range(17):
            self.keypoint_meshes.append(
                self.keypoint_plotter.add_mesh(pv.Sphere(radius=0.025), color="orange")
            )
        for _ in self.keypoint_bones_indices:
            self.keypoint_bone_meshes.append(
                self.keypoint_plotter.add_mesh(
                    pv.Cylinder(
                        center=(0, 0, 0),
                        direction=(0, 1, 0),
                        radius=0.01,
                        height=1.0,
                    ),
                    color="purple",
                )
            )
        self._set_keypoint_visibility(False)
        for plotter in (self.plotter, self.keypoint_plotter):
            plotter.show_bounds(
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
            plotter.add_axes(line_width=2)
        self._reset_camera_views()

    def _reset_camera_views(self) -> None:
        """Reset both 3D views to the front-facing initial camera."""
        for plotter in (self.plotter, self.keypoint_plotter):
            plotter.camera.position = INITIAL_CAMERA_POSITION
            plotter.camera.focal_point = INITIAL_CAMERA_FOCAL_POINT
            plotter.camera.up = INITIAL_CAMERA_UP
            plotter.render()

    def toggle_axes_visibility(self, state: int) -> None:
        """ローカル座標軸の表示/非表示を切り替え"""
        visible = bool(state)
        for _, _, actor in self.axis_actors:
            if hasattr(actor, "SetVisibility"):
                actor.SetVisibility(visible)
            elif hasattr(actor, "visibility"):
                actor.visibility = visible
        self.plotter.render()
    
    @staticmethod
    def _local_axis_specs(index: int) -> tuple[tuple[int, str], ...]:
        axis_map = {
            "hips": (0, 1, 2), "neck": (0, 1, 2),
            "L_shoulder": (0, 1, 2), "R_shoulder": (0, 1, 2),
            "L_elbow": (0, 1), "R_elbow": (0, 1),
            "L_hip": (0, 1, 2), "R_hip": (0, 1, 2),
            "L_knee": (0,), "R_knee": (0,),
        }
        colors = ("red", "green", "blue")
        name = m_config.JOINT_NAMES[index]
        return tuple((axis, colors[axis]) for axis in axis_map.get(name, ()))

    def _on_joint_picked(self, actor) -> None:
        """3Dビュー上で関節がクリックされた時のコールバック"""
        if actor is None:
            return
        target_index = None
        for idx, joint_actor in enumerate(self.joint_meshes):
            underlying = joint_actor[0] if isinstance(joint_actor, tuple) else joint_actor
            picked = actor[0] if isinstance(actor, tuple) else actor
            if underlying == picked:
                target_index = idx
                break

        if target_index is not None:
            self._select_joint(target_index)

    def _select_joint(self, index: int) -> None:
        """指定された関節を選択し、UIとハイライト表示を更新"""
        self.selected_joint_index = index
        joint_name = m_config.JOINT_NAMES[index]
        self.joint_label.setText(f"{joint_name} (index: {index})")

        # 関節の球体ハイライト表示を更新（選択関節を黄色、その他を通常色に）
        for idx, joint_actor in enumerate(self.joint_meshes):
            act = joint_actor[0] if isinstance(joint_actor, tuple) else joint_actor
            if hasattr(act, "GetProperty"):
                prop = act.GetProperty()
                if idx == index:
                    prop.SetColor(1.0, 1.0, 0.0)  # 選択された関節: 黄色
                else:
                    if idx in self.left_indices:
                        prop.SetColor(1.0, 0.0, 0.0)  # 左: 赤
                    elif idx in self.right_indices:
                        prop.SetColor(0.0, 0.0, 1.0)  # 右: 青
                    else:
                        prop.SetColor(0.0, 1.0, 0.0)  # 中央: 緑
        self.plotter.render()

        # スライダーとスピンボックスを選択関節の値に同期
        self._sync_angle_inputs(index)

    def load_coordinate_file(self) -> None:
        """17点座標ファイル (.npy または .json) を読み込む"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "17点座標ファイルを選択", str(Path.cwd()),
            "Coordinate files (*.npy *.json);;NumPy files (*.npy);;JSON files (*.json);;All files (*)",
        )
        if not file_path:
            return
        path = Path(file_path)
        try:
            if path.suffix.lower() == ".npy":
                data = np.load(file_path)
                # 形状判定: (17, 3) または (F, 17, 3)
                if data.ndim == 2 and data.shape == (17, 3):
                    keypoints_seq = data[np.newaxis, ...]
                elif data.ndim == 3 and data.shape[1:] == (17, 3):
                    keypoints_seq = data
                else:
                    raise ValueError(f"不正な.npy形状です: {data.shape} (期待: (17, 3) または (Frames, 17, 3))")
                
                self.sequence_keypoints = keypoints_seq
                self.sequence_angles = calc_angles.calculate_joint_angles(self.sequence_keypoints)
                self.sequence_coords = vis_motion.reconstruct_3d_positions(self.sequence_angles)
                motion_angles = m_lang.to_motion_angles(self.sequence_angles)
                self.sequence_tokens = self.encoder.encode(motion_angles)
                self.manual_angles = motion_angles[0:1].copy()
                self.input_mode = "keypoints"
                self.file_label.setText(path.name)
                self._select_joint(self.selected_joint_index)
                self._set_frame(0)
                self.log_message(
                    f"座標.npy読み込み完了: {path.name}\n"
                    f"フレーム数: {len(self.sequence_keypoints)}\n"
                    f"角度形状: {self.sequence_angles.shape}\n"
                    f"座標形状: {self.sequence_coords.shape}"
                )
            elif path.suffix.lower() == ".json":
                pose_name, scale, keypoints = load_keypoints_json(file_path)
                self.sequence_keypoints = keypoints[np.newaxis, ...]
                self.sequence_angles = calc_angles.calculate_joint_angles(self.sequence_keypoints)
                self.sequence_coords = vis_motion.reconstruct_3d_positions(self.sequence_angles)
                motion_angles = m_lang.to_motion_angles(self.sequence_angles)
                self.sequence_tokens = self.encoder.encode(motion_angles)
                self.manual_angles = motion_angles[0:1].copy()
                self.input_mode = "keypoints"
                self.file_label.setText(path.name)
                self._select_joint(self.selected_joint_index)
                self._set_frame(0)
                self.log_message(
                    f"座標JSON読み込み完了: {pose_name}\n"
                    f"scale: {scale:g}\n"
                    f"角度形状: {self.sequence_angles.shape}\n"
                    f"座標形状: {self.sequence_keypoints.shape}"
                )
            else:
                raise ValueError(f"対応していないファイル形式です: {path.suffix}")
        except (OSError, ValueError) as error:
            self.log_message(f"座標ファイル読み込みエラー: {error}")

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
            motion_angles = self.decoder.decode_lines(self.sequence_tokens)
            self.sequence_angles = m_lang.from_motion_angles(motion_angles)
            self.sequence_coords = vis_motion.reconstruct_3d_positions(self.sequence_angles)
            if self.sequence_keypoints is None:
                _, _, keypoints = load_keypoints_json(DEFAULT_POSE_PATH)
                self.sequence_keypoints = keypoints[np.newaxis, ...]
            self.input_mode = "angles"
            self.manual_angles = motion_angles[0:1].copy()
            self._select_joint(self.selected_joint_index)
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

    def load_keypoint_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "17点座標JSONファイルを選択", str(Path.cwd()),
            "JSON files (*.json);;All files (*)",
        )
        if not file_path:
            return
        try:
            pose_name, scale, keypoints = load_keypoints_json(file_path)
            self.sequence_keypoints = keypoints[np.newaxis, ...]
            self.sequence_angles = calc_angles.calculate_joint_angles(self.sequence_keypoints)
            self.sequence_coords = vis_motion.reconstruct_3d_positions(self.sequence_angles)
            motion_angles = m_lang.to_motion_angles(self.sequence_angles)
            self.sequence_tokens = self.encoder.encode(motion_angles)
            self.manual_angles = motion_angles[0:1].copy()
            self.input_mode = "keypoints"
            self.file_label.setText(Path(file_path).name)
            self._select_joint(self.selected_joint_index)
            self._set_frame(0)
            self.log_message(
                f"座標JSON読み込み完了: {pose_name}\n"
                f"scale: {scale:g}\n"
                f"角度形状: {self.sequence_angles.shape}\n"
                f"座標形状: {self.sequence_keypoints.shape}"
            )
        except (OSError, ValueError) as error:
            self.log_message(f"座標JSON読み込みエラー: {error}")

    def _load_pose_json(self, path: Path, log_message: bool = True) -> None:
        """Load a single 17-point pose as the initial JSON-backed state."""
        try:
            pose_name, scale, keypoints = load_keypoints_json(path)
            self.sequence_keypoints = keypoints[np.newaxis, ...]
            self.sequence_angles = calc_angles.calculate_joint_angles(self.sequence_keypoints)
            self.sequence_coords = vis_motion.reconstruct_3d_positions(self.sequence_angles)
            motion_angles = m_lang.to_motion_angles(self.sequence_angles)
            self.sequence_tokens = self.encoder.encode(motion_angles)
            self.manual_angles = motion_angles[0:1].copy()
            self.input_mode = "keypoints"
            self.file_label.setText(path.name)
            self._select_joint(self.selected_joint_index)
            self._set_frame(0)
            if log_message:
                self.log_message(
                    f"座標JSON読み込み完了: {pose_name}\n"
                    f"scale: {scale:g}\n"
                    f"角度形状: {self.sequence_angles.shape}\n"
                    f"座標形状: {self.sequence_keypoints.shape}"
                )
        except (OSError, ValueError) as error:
            self.log_message(f"初期姿勢の読み込みエラー: {error}")

    @staticmethod
    def _keypoints_to_joint_positions(keypoints: np.ndarray) -> np.ndarray:
        points = {name: keypoints[index] for name, index in calc_angles.COCO_INDEX.items()}
        positions = np.empty((len(m_config.JOINT_NAMES), 3), dtype=float)
        positions[m_config.JOINT_INDEX["hips"]] = (points["lefthip"] + points["righthip"]) / 2
        positions[m_config.JOINT_INDEX["neck"]] = (
            points["leftshoulder"] + points["rightshoulder"]
        ) / 2
        positions[m_config.JOINT_INDEX["head"]] = np.mean(
            [points["nose"], points["lefteye"], points["righteye"],
             points["leftear"], points["rightear"]],
            axis=0,
        )
        for side in ("L", "R"):
            source = "left" if side == "L" else "right"
            for joint in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
                positions[m_config.JOINT_INDEX[f"{side}_{joint}"]] = points[f"{source}{joint}"]
        return positions

    def _set_keypoint_visibility(self, visible: bool) -> None:
        for actor in (*self.keypoint_meshes, *self.keypoint_bone_meshes):
            actor.visibility = visible
        self.keypoint_plotter.render()

    def _set_joint_visibility(self, visible: bool) -> None:
        for actor in (*self.joint_meshes, *self.bone_meshes):
            actor.visibility = visible

    def _on_slider_pressed(self) -> None:
        """スライダーのドラッグ操作開始時"""
        if self._is_history_action:
            return
        if self._pre_edit_angles is None:
            self._pre_edit_angles = self.manual_angles.copy()
        if self._edit_debounce_timer.isActive():
            self._edit_debounce_timer.stop()

    def _on_slider_released(self) -> None:
        """スライダーのドラッグ操作終了時"""
        if self._is_history_action:
            return
        self._commit_history_snapshot()

    def _commit_history_snapshot(self) -> None:
        """一連の姿勢編集操作を1つのUndoスナップショットとして確定"""
        if self._is_history_action:
            return
        if self._pre_edit_angles is not None:
            if not np.array_equal(self._pre_edit_angles, self.manual_angles):
                self.history_stack.append(self._pre_edit_angles.copy())
                if len(self.history_stack) > 50:
                    self.history_stack.pop(0)
                self.redo_stack.clear()
            self._pre_edit_angles = None

    def encode_from_angles(self, log_output: bool = True) -> None:
        """指定された関節角度から動作言語へエンコードし、3D姿勢も同時に更新する"""
        # PyQtのシグナル(clicked等)から呼ばれた際の引数(checked: bool)を補正
        if not isinstance(log_output, bool):
            log_output = True

        # 操作開始前の姿勢を記録し、操作終了判定タイマー（デバウンス）をリセット
        if not self._is_history_action:
            if self._pre_edit_angles is None:
                self._pre_edit_angles = self.manual_angles.copy()
            self._edit_debounce_timer.start()

        joint_index = self.selected_joint_index
        self.manual_angles[0, joint_index] = [
            spinbox.value() for spinbox in self.angle_spinboxes
        ]
        try:
            tokens = self.encoder.encode(self.manual_angles)
            token_text = "\n".join(tokens)
            self.language_input.setPlainText(token_text)

            # 角度から3D座標を順運動学(FK)で再構成してビューアに反映
            self.sequence_angles = m_lang.from_motion_angles(self.manual_angles)
            coords = vis_motion.reconstruct_3d_positions(self.sequence_angles)
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
        """Reset to the posture represented by ``t_pose.json``."""
        self._load_pose_json(DEFAULT_POSE_PATH, log_message=False)

    def save_ground_truth_data(self) -> None:
        """現在の姿勢データをユニットテスト用の正解データ(JSONまたはNPZ)として保存"""
        if self.sequence_coords is None or self.sequence_angles is None:
            self.log_message("保存可能な姿勢データがありません")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "テスト用正解データを保存",
            str(Path.cwd() / "ground_truth.json"),
            "JSON files (*.json);;NumPy Zip files (*.npz);;All files (*)",
        )
        if not file_path:
            return

        frame_idx = self.current_frame_idx
        coords_15 = self.sequence_coords[frame_idx]
        angles_15 = self.sequence_angles[frame_idx]
        
        # 現在の動作言語トークン
        motion_angles = m_lang.to_motion_angles(angles_15[np.newaxis, ...])
        tokens = self.encoder.encode(motion_angles)
        token_str = tokens[0] if tokens else ""

        path = Path(file_path)
        try:
            if path.suffix.lower() == ".npz":
                save_dict = {
                    "coords_15": coords_15,
                    "angles_15": angles_15,
                    "motion_token": np.array(token_str),
                    "joint_names": np.array(m_config.JOINT_NAMES),
                }
                if self.sequence_keypoints is not None:
                    kp_idx = min(frame_idx, len(self.sequence_keypoints) - 1)
                    save_dict["keypoints_17"] = self.sequence_keypoints[kp_idx]
                np.savez_compressed(path, **save_dict)
            else:
                data_dict = {
                    "joint_names": m_config.JOINT_NAMES,
                    "coords_15": coords_15.tolist(),
                    "angles_15": angles_15.tolist(),
                    "motion_token": token_str,
                }
                if self.sequence_keypoints is not None:
                    kp_idx = min(frame_idx, len(self.sequence_keypoints) - 1)
                    data_dict["keypoints_17"] = self.sequence_keypoints[kp_idx].tolist()

                path.write_text(json.dumps(data_dict, indent=2, ensure_ascii=False), encoding="utf-8")

            self.log_message(
                f"正解データを保存しました: {path.name}\n"
                f"15関節3D座標: shape {coords_15.shape}\n"
                f"関節オイラー角: shape {angles_15.shape}\n"
                f"動作言語トークン: {token_str}"
            )
        except Exception as error:
            self.log_message(f"正解データの保存に失敗しました: {error}")

    def decode_from_language(self) -> None:
        token_text = self.language_input.toPlainText().strip()
        if not token_text:
            self.log_message("デコードする動作言語が入力されていません")
            return
        try:
            self.sequence_tokens = [line for line in token_text.splitlines() if line.strip()]
            motion_angles = self.decoder.decode_lines(self.sequence_tokens)
            self.sequence_angles = m_lang.from_motion_angles(motion_angles)
            self.sequence_coords = vis_motion.reconstruct_3d_positions(self.sequence_angles)
            self.input_mode = "angles"
            
            # デコードされた1フレーム目の角度を手動角度入力用バッファにも反映
            if len(self.sequence_angles) > 0:
                self.manual_angles = motion_angles[0:1].copy()
                self._sync_angle_inputs(self.selected_joint_index)

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
        joint_limits = {
            axis: (minimum, maximum)
            for joint_index, axis, _, _, minimum, maximum in m_config.AXIS_MAPPING.values()
            if joint_index == index
        }
        for axis_index, (spinbox, slider, value) in enumerate(
            zip(self.angle_spinboxes, self.angle_sliders, self.manual_angles[0, index])
        ):
            limits = joint_limits.get(axis_index)
            enabled = limits is not None
            minimum, maximum = limits if limits is not None else (-180.0, 180.0)
            spinbox.blockSignals(True)
            slider.blockSignals(True)
            spinbox.setRange(minimum, maximum)
            slider.setRange(int(minimum), int(maximum))
            spinbox.setEnabled(enabled)
            slider.setEnabled(enabled)
            spinbox.setValue(float(np.clip(value, minimum, maximum)))
            slider.setValue(int(np.clip(value, minimum, maximum)))
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
        if self.sequence_keypoints is not None:
            keypoint_index = min(
                self.current_frame_idx, len(self.sequence_keypoints) - 1
            )
            self.update_keypoint_visualization(
                self.sequence_keypoints[keypoint_index],
                self.sequence_angles[self.current_frame_idx],
            )

    def seek_frame(self, frame_index: int) -> None:
        self._set_frame(frame_index)

    def next_frame(self) -> None:
        if self.sequence_coords is None:
            self.timer.stop()
            self._set_manual_controls_enabled(True)
            self.play_button.setText("▶ 再生")
            self.log_message("再生するシーケンスがありません")
            return
        if self.current_frame_idx >= len(self.sequence_coords) - 1:
            self.timer.stop()
            self._set_manual_controls_enabled(True)
            self.play_button.setText("▶ 再生")
            return
        self._set_frame(self.current_frame_idx + 1)

    def prev_frame(self) -> None:
        if self.sequence_coords is not None:
            self._set_frame(self.current_frame_idx - 1)

    def undo_state(self) -> None:
        """1つ前の姿勢状態に戻る（Undo）"""
        if self._edit_debounce_timer.isActive():
            self._edit_debounce_timer.stop()
            self._commit_history_snapshot()
        if not self.history_stack:
            return
        self.redo_stack.append(self.manual_angles.copy())
        prev_angles = self.history_stack.pop()
        self._apply_angles_to_state(prev_angles)

    def redo_state(self) -> None:
        """戻した姿勢状態を進める（Redo）"""
        if self._edit_debounce_timer.isActive():
            self._edit_debounce_timer.stop()
            self._commit_history_snapshot()
        if not self.redo_stack:
            return
        self.history_stack.append(self.manual_angles.copy())
        next_angles = self.redo_stack.pop()
        self._apply_angles_to_state(next_angles)

    def _apply_angles_to_state(self, angles: np.ndarray) -> None:
        """指定した角度配列を現在のUIおよび3D姿勢に反映"""
        self._is_history_action = True
        try:
            self.manual_angles = angles.copy()
            self._sync_angle_inputs(self.selected_joint_index)
            tokens = self.encoder.encode(self.manual_angles)
            self.language_input.setPlainText("\n".join(tokens))
            self.sequence_angles = m_lang.from_motion_angles(self.manual_angles)
            self.sequence_coords = vis_motion.reconstruct_3d_positions(self.sequence_angles)
            self.sequence_tokens = tokens
            self.file_label.setText("手動角度入力")
            self._set_frame(0)
        finally:
            self._is_history_action = False

    def _set_manual_controls_enabled(self, enabled: bool) -> None:
        """連続再生中の手動編集の排他制御（無効化/有効化）"""
        if hasattr(self, "angle_group"):
            self.angle_group.setEnabled(enabled)
        if hasattr(self, "decode_button"):
            self.decode_button.setEnabled(enabled)

    def toggle_playback(self) -> None:
        if self.sequence_coords is None:
            self.log_message("再生するシーケンスがありません")
            return
        if self.timer.isActive():
            self.timer.stop()
            self._set_manual_controls_enabled(True)
            self.play_button.setText("▶ 再生")
            self.log_message("一時停止")
            return
        if self.current_frame_idx >= len(self.sequence_coords) - 1:
            self._set_frame(0)
        self._update_timer_interval()
        self._set_manual_controls_enabled(False)
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
        self._set_joint_visibility(True)
        self._set_keypoint_visibility(False)

        # 各関節の球体と、許可されたローカル軸を更新
        for index, (actor, position) in enumerate(zip(self.joint_meshes, positions)):
            trans_mat = self._translation_matrix(position)
            self._set_actor_matrix(actor, trans_mat)
            for axis_index, joint_axis_index, axis_actor in self.axis_actors:
                if axis_index != index:
                    continue
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

    def update_keypoint_visualization(
        self, keypoints: np.ndarray, joint_angles: np.ndarray
    ) -> None:
        points = np.asarray(keypoints, dtype=float)
        if points.shape != (17, 3):
            raise ValueError("keypoints must have shape (17, 3)")
        self._set_keypoint_visibility(True)
        for actor, position in zip(self.keypoint_meshes, points):
            self._set_actor_matrix(actor, self._translation_matrix(position))
        for actor, (parent_index, child_index) in zip(
            self.keypoint_bone_meshes, self.keypoint_bones_indices
        ):
            start = points[parent_index]
            end = points[child_index]
            vector = end - start
            length = float(np.linalg.norm(vector))
            actor.visibility = length > 1e-8
            if actor.visibility:
                self._set_actor_matrix(
                    actor,
                    self._cylinder_transform(
                        (start + end) / 2.0, vector / length, length
                    ),
                )
        self.keypoint_plotter.render()

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
        if "エラー" in message:
            QMessageBox.critical(self, "エラー", message)

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.plotter.close()
        self.keypoint_plotter.close()
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
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())