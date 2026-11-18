#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

NUM_JOINTS = 17
FPS = 60
MODEL_NAME = "luxonis/yolo26-nano-pose-estimation:coco-512x288"
YOLO26_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


class JointFilter:
    def __init__(self, window_size: int = 3) -> None:
        self.history = {index: deque(maxlen=window_size) for index in range(NUM_JOINTS)}
        self.last_valid = {index: (0.0, 0.0, 0.0) for index in range(NUM_JOINTS)}

    def update(self, joint_id: int, x: float, y: float, z: float) -> tuple[float, float, float]:
        if z <= 0:
            previous = self.last_valid[joint_id]
            if previous[2] > 0:
                x, y, z = previous
            else:
                return x, y, z
        self.history[joint_id].append((x, y, z))
        self.last_valid[joint_id] = (x, y, z)
        values = np.asarray(self.history[joint_id])
        return tuple(np.mean(values, axis=0).tolist())


def frame_norm(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    norm_values = np.full(len(bbox), frame.shape[0])
    norm_values[::2] = frame.shape[1]
    return tuple((np.clip(np.asarray(bbox), 0, 1) * norm_values).astype(int).tolist())


class PoseCaptureWindow(QMainWindow):
    def __init__(self, depth_source: str) -> None:
        super().__init__()
        self.setWindowTitle("姿勢座標取得")
        self.resize(1280, 600)

        self.device = dai.Device()
        self.pipeline = self._build_pipeline(depth_source)
        self.rgb_queue = self.detection_network.passthrough.createOutputQueue(
            maxSize=4, blocking=False
        )
        self.detection_queue = self.spatial_calculator.outputDetections.createOutputQueue(
            maxSize=4, blocking=False
        )
        self.pipeline.start()

        self.joint_filter = JointFilter()
        self.latest_coordinates: np.ndarray | None = None
        self.frame_id = 0
        self.last_frame_time = time.monotonic()
        self.fps_history: deque[float] = deque(maxlen=10)
        self.latest_detections = []

        # 録画状態の管理
        self.is_recording = False
        self.recorded_coordinates: list[np.ndarray] = []
        self.video_writer: cv2.VideoWriter | None = None
        self.video_output_path: Path | None = None
        self.motion_output_path: Path | None = None

        self._init_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

    def _build_pipeline(self, depth_source: str) -> dai.Pipeline:
        pipeline = dai.Pipeline(self.device)
        camera = pipeline.create(dai.node.Camera).build(sensorFps=FPS)
        mono_left = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_B, sensorFps=FPS
        )
        mono_right = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_C, sensorFps=FPS
        )
        mono_left_output = mono_left.requestOutput((640, 400), fps=FPS)
        mono_right_output = mono_right.requestOutput((640, 400), fps=FPS)

        capabilities = dai.ImgFrameCapability()
        capabilities.fps.fixed(FPS)
        capabilities.enableUndistortion = True
        self.detection_network = pipeline.create(dai.node.DetectionNetwork).build(
            camera, MODEL_NAME, capabilities
        )
        self.spatial_calculator = pipeline.create(dai.node.SpatialLocationCalculator)
        self.spatial_calculator.initialConfig.setCalculateSpatialKeypoints(True)
        self.detection_network.out.link(self.spatial_calculator.inputDetections)

        if depth_source == "stereo":
            depth = pipeline.create(dai.node.StereoDepth).build(
                mono_left_output,
                mono_right_output,
                presetMode=dai.node.StereoDepth.PresetMode.FAST_ACCURACY,
            )
            if self.device.getPlatform() == dai.Platform.RVC2:
                self.detection_network.passthrough.link(depth.inputAlignTo)
                depth.depth.link(self.spatial_calculator.inputDepth)
        else:
            depth = pipeline.create(dai.node.NeuralDepth).build(
                mono_left_output,
                mono_right_output,
                dai.DeviceModelZoo.NEURAL_DEPTH_MEDIUM,
            )

        if self.device.getPlatform() == dai.Platform.RVC4:
            align = pipeline.create(dai.node.ImageAlign)
            depth.depth.link(align.input)
            self.detection_network.passthrough.link(align.inputAlignTo)
            align.outputAligned.link(self.spatial_calculator.inputDepth)
        return pipeline

    def _init_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        image_layout = QHBoxLayout()
        self.raw_view = QLabel("カメラ映像")
        self.skeleton_view = QLabel("骨格映像")
        for view in (self.raw_view, self.skeleton_view):
            view.setAlignment(Qt.AlignmentFlag.AlignCenter)
            view.setMinimumSize(480, 360)
            view.setStyleSheet("background-color: black;")
            image_layout.addWidget(view)
        layout.addLayout(image_layout)

        controls = QHBoxLayout()
        self.record_button = QPushButton("● 録画開始")
        self.record_button.setStyleSheet("font-weight: bold; padding: 6px 12px;")
        self.record_button.clicked.connect(self.toggle_recording)
        quit_button = QPushButton("終了")
        quit_button.clicked.connect(self.close)
        self.status_label = QLabel("姿勢を検出しています")
        controls.addWidget(self.record_button)
        controls.addWidget(quit_button)
        controls.addWidget(self.status_label, stretch=1)
        layout.addLayout(controls)

    def update_frame(self) -> None:
        rgb_packet = self.rgb_queue.tryGet()
        detection_packet = self.detection_queue.tryGet()
        if detection_packet is not None:
            self.latest_detections = detection_packet.detections
        if rgb_packet is None:
            return

        frame = rgb_packet.getCvFrame()
        skeleton, coordinates = self._make_skeleton_frame(frame, self.latest_detections)
        self.latest_coordinates = coordinates
        self.frame_id += 1

        # 録画中のデータ蓄積
        if self.is_recording:
            # 17点座標の記録 (未検出フレームは直前の姿勢またはゼロで埋めてフレーム整合性を維持)
            if coordinates is not None:
                coord_frame = coordinates.copy()
            elif self.recorded_coordinates:
                coord_frame = self.recorded_coordinates[-1].copy()
            else:
                coord_frame = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
            self.recorded_coordinates.append(coord_frame)

            # 横並び動画フレームの書き出し (生映像 + 黒背景骨格映像)
            side_by_side = np.hstack((frame, skeleton))
            if self.video_writer is not None:
                self.video_writer.write(side_by_side)

        now = time.monotonic()
        interval = now - self.last_frame_time
        self.last_frame_time = now
        if interval > 0:
            self.fps_history.append(1.0 / interval)

        state = "検出中" if coordinates is not None else "姿勢未検出"
        fps = np.mean(self.fps_history) if self.fps_history else 0.0
        rec_status = f" [● REC: {len(self.recorded_coordinates)} frames]" if self.is_recording else ""
        self.status_label.setText(f"{state}  FPS: {fps:.1f}  frame: {self.frame_id}{rec_status}")
        self._set_image(self.raw_view, frame)
        self._set_image(self.skeleton_view, skeleton)

    def _make_skeleton_frame(
        self, frame: np.ndarray, detections: list
    ) -> tuple[np.ndarray, np.ndarray | None]:
        skeleton = np.zeros_like(frame)
        main_detection = self._nearest_detection(detections)
        if main_detection is None:
            return skeleton, None

        keypoints = main_detection.getKeypoints()
        if len(keypoints) < NUM_JOINTS:
            return skeleton, None
        image_points = [
            frame_norm(frame, (point.imageCoordinates.x, point.imageCoordinates.y))
            for point in keypoints[:NUM_JOINTS]
        ]
        person_depth = main_detection.spatialCoordinates.z / 1000.0
        smooth_joints: dict[int, tuple[float, float, float]] = {}
        for index, keypoint in enumerate(keypoints[:NUM_JOINTS]):
            score = getattr(keypoint, "score", getattr(keypoint, "confidence", 1.0))
            if score < 0.4:
                continue
            x = keypoint.spatialCoordinates.x / 1000.0
            y = keypoint.spatialCoordinates.y / 1000.0
            z = keypoint.spatialCoordinates.z / 1000.0
            if z <= 0 and person_depth > 0:
                z = person_depth
            smooth_joints[index] = self.joint_filter.update(index, x, y, z)

        if 11 not in smooth_joints or 12 not in smooth_joints:
            return skeleton, None
        base = (np.asarray(smooth_joints[11]) + np.asarray(smooth_joints[12])) / 2.0
        coordinates = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
        for index in range(NUM_JOINTS):
            if index not in smooth_joints or smooth_joints[index][2] <= 0:
                return skeleton, None
            coordinates[index] = np.asarray(smooth_joints[index]) - base

        for point in image_points:
            cv2.circle(skeleton, point[:2], 3, (0, 255, 0), -1)
        for start, end in YOLO26_EDGES:
            cv2.line(skeleton, image_points[start][:2], image_points[end][:2], (0, 255, 0), 2)
        return skeleton, coordinates

    @staticmethod
    def _nearest_detection(detections: list):
        valid = [
            detection for detection in detections
            if detection.spatialCoordinates.z > 0
        ]
        return min(valid, key=lambda detection: detection.spatialCoordinates.z, default=None)

    @staticmethod
    def _set_image(label: QLabel, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        label.setPixmap(pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def toggle_recording(self) -> None:
        """録画の開始 / 停止を切り替える"""
        if not self.is_recording:
            # 録画開始
            self.is_recording = True
            self.recorded_coordinates = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_dir = Path(__file__).resolve().parents[2] / "data"

            # 動画ライター準備 (幅2倍: 生映像 + 骨格映像)
            video_dir = data_dir / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            self.video_output_path = video_dir / f"video_{timestamp}.mp4"
            self.motion_output_path = data_dir / "motion_position" / f"motion_{timestamp}.npy"
            self.motion_output_path.parent.mkdir(parents=True, exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            # 640x400 のフレームが横に2枚並ぶため 1280x400
            self.video_writer = cv2.VideoWriter(
                str(self.video_output_path),
                fourcc,
                FPS,
                (640 * 2, 400),
            )

            self.record_button.setText("■ 録画停止")
            self.record_button.setStyleSheet("background-color: #d93025; color: white; font-weight: bold; padding: 6px 12px;")
            self.status_label.setText(f"録画を開始しました: {self.video_output_path.name}")
        else:
            # 録画停止
            self.is_recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None

            num_frames = len(self.recorded_coordinates)
            if num_frames > 0 and self.motion_output_path is not None:
                # (Frames, 17, 3) の NumPy 配列として保存
                motion_array = np.asarray(self.recorded_coordinates, dtype=np.float32)
                np.save(self.motion_output_path, motion_array)
                msg = f"保存完了: {self.motion_output_path.name} ({num_frames} frames), {self.video_output_path.name}"
            else:
                msg = "録画フレームがありませんでした"

            self.record_button.setText("● 録画開始")
            self.record_button.setStyleSheet("font-weight: bold; padding: 6px 12px;")
            self.status_label.setText(msg)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.is_recording:
            self.toggle_recording()
        self.timer.stop()
        if self.pipeline.isRunning():
            self.pipeline.stop()
        self.device.close()
        event.accept()


def main() -> int:
    parser = argparse.ArgumentParser(description="姿勢座標取得ウィンドウ")
    parser.add_argument("--depthSource", choices=("stereo", "neural"), default="stereo")
    args = parser.parse_args()
    app = QApplication(sys.argv)
    window = PoseCaptureWindow(args.depthSource)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
