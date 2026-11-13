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
        self.save_button = QPushButton("現在の座標を保存")
        self.save_button.clicked.connect(self.save_current_coordinates)
        quit_button = QPushButton("終了")
        quit_button.clicked.connect(self.close)
        self.status_label = QLabel("姿勢を検出しています")
        controls.addWidget(self.save_button)
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
        now = time.monotonic()
        interval = now - self.last_frame_time
        self.last_frame_time = now
        if interval > 0:
            self.fps_history.append(1.0 / interval)
        state = "検出中" if coordinates is not None else "姿勢未検出"
        fps = np.mean(self.fps_history) if self.fps_history else 0.0
        self.status_label.setText(f"{state}  FPS: {fps:.1f}  frame: {self.frame_id}")
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

    def save_current_coordinates(self) -> None:
        if self.latest_coordinates is None:
            self.status_label.setText("保存できません: 姿勢が検出されていません")
            return
        output_dir = Path(__file__).resolve().parents[2] / "data_logs" / "keypoints"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now()
        path = output_dir / f"keypoints_{timestamp:%Y%m%d_%H%M%S_%f}.json"
        payload = {
            "timestamp": timestamp.isoformat(timespec="milliseconds"),
            "frame_id": self.frame_id,
            "coordinate_unit": "meter",
            "coordinate_origin": "hip_center",
            "keypoints": {
                name: self.latest_coordinates[index].astype(float).tolist()
                for index, name in enumerate(KEYPOINT_NAMES)
            },
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.status_label.setText(f"保存しました: {path.name}")

    def closeEvent(self, event: QCloseEvent) -> None:
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
