#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import time
import argparse
import csv
from datetime import datetime
from collections import deque
from pathlib import Path

try:
    from .calculate_joint_angles_array import (
        OUTPUT_JOINTS,
        calculate_joint_angles,
    )
    from .motion_trigger import MotionTriggerEngine
except ImportError:
    from calculate_joint_angles_array import OUTPUT_JOINTS, calculate_joint_angles
    from motion_trigger import MotionTriggerEngine

NUM_JOINTS = 17

YOLO26_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

parser = argparse.ArgumentParser()
parser.add_argument("--depthSource", type=str, default="stereo", choices=["stereo", "neural"])
args = parser.parse_args()

device = dai.Device()
fps = 60

MODEL_NAME = "luxonis/yolo26-nano-pose-estimation:coco-512x288"

class JointFilter:
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.history = {i: deque(maxlen=window_size) for i in range(NUM_JOINTS)}
        self.last_valid = {i: (0.0, 0.0, 0.0) for i in range(NUM_JOINTS)}

    def update(self, joint_id, x, y, z):
        if joint_id >= NUM_JOINTS: return x, y, z
        if z <= 0:
            if self.last_valid[joint_id][2] > 0: x, y, z = self.last_valid[joint_id]
            else: return x, y, z
        self.history[joint_id].append((x, y, z))
        self.last_valid[joint_id] = (x, y, z)
        arr = np.array(self.history[joint_id])
        return np.mean(arr[:, 0]), np.mean(arr[:, 1]), np.mean(arr[:, 2])

joint_filter = JointFilter(window_size=3)

# ⚡ モジュールの初期化（コントローラー管理下）
motion_trigger = MotionTriggerEngine(
    threshold_energy=4.0,
    release_threshold=2.0,
    min_trigger_frames=6,
    stability_frames=8,
)

SIGNAL_WINDOW_SIZE = 60
angle_energy_history = deque([0.0] * SIGNAL_WINDOW_SIZE, maxlen=SIGNAL_WINDOW_SIZE)


def save_angle_sequence_csv(angle_sequence):
    """Save one completed angle sequence in the temporary CSV format."""
    output_dir = Path(__file__).resolve().parents[2] / "data_logs" / "angles"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_dir / f"motion_angles_{timestamp}.csv"
    headers = ["frame"]
    for joint in OUTPUT_JOINTS:
        headers.extend(f"{joint}_{axis.lower()}" for axis in ("Roll", "Pitch", "Yaw"))

    with output_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        for frame_index, frame_angles in enumerate(angle_sequence):
            writer.writerow([frame_index, *frame_angles.reshape(-1)])

    return output_path

with dai.Pipeline(device) as pipeline:
    cameraNode = pipeline.create(dai.node.Camera).build(sensorFps=fps)
    monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=fps)
    monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=fps)
    monoLeftOut = monoLeft.requestOutput((640, 400), fps=fps)
    monoRightOut = monoRight.requestOutput((640, 400), fps=fps)

    requiredCamCapabilities = dai.ImgFrameCapability()
    requiredCamCapabilities.fps.fixed(fps)
    requiredCamCapabilities.enableUndistortion = True
    
    detectionNetwork = pipeline.create(dai.node.DetectionNetwork).build(cameraNode, MODEL_NAME, requiredCamCapabilities)
    spatialCalculator = pipeline.create(dai.node.SpatialLocationCalculator)
    spatialCalculator.initialConfig.setCalculateSpatialKeypoints(True)
    detectionNetwork.out.link(spatialCalculator.inputDetections)

    if args.depthSource == "stereo":
        depth = pipeline.create(dai.node.StereoDepth).build(monoLeftOut, monoRightOut, presetMode=dai.node.StereoDepth.PresetMode.FAST_ACCURACY)
        if device.getPlatform() == dai.Platform.RVC2:
            detectionNetwork.passthrough.link(depth.inputAlignTo)
            depth.depth.link(spatialCalculator.inputDepth)
    elif args.depthSource == "neural":
        depth = pipeline.create(dai.node.NeuralDepth).build(monoLeftOut, monoRightOut, dai.DeviceModelZoo.NEURAL_DEPTH_MEDIUM)

    if device.getPlatform() == dai.Platform.RVC4:
        align = pipeline.create(dai.node.ImageAlign)
        depth.depth.link(align.input)
        detectionNetwork.passthrough.link(align.inputAlignTo)
        align.outputAligned.link(spatialCalculator.inputDepth)

    qRgb = detectionNetwork.passthrough.createOutputQueue(maxSize=4, blocking=False)
    qDet = spatialCalculator.outputDetections.createOutputQueue(maxSize=4, blocking=False)

    pipeline.start()
    
    last_frame_time = time.monotonic()
    fps_display = 0.0
    fps_smoothing_queue = deque(maxlen=10)

    frame = None
    detections = []

    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def process_and_display(frame, detections):
        global last_frame_time, fps_display

        current_time = time.monotonic()
        frame_interval = current_time - last_frame_time
        last_frame_time = current_time
        
        instant_fps = 1.0 / frame_interval if frame_interval > 0 else 0.0
        fps_smoothing_queue.append(instant_fps)
        fps_display = np.mean(fps_smoothing_queue)

        frame_black_skeleton = np.zeros_like(frame)
        current_frame_data = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
        
        main_detection = None
        min_depth = float('inf')
        
        for detection in detections:
            det_depth = detection.spatialCoordinates.z / 1000.0
            if 0 < det_depth < min_depth:
                min_depth = det_depth
                main_detection = detection

        if main_detection is not None:
            detection = main_detection
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.rectangle(frame_black_skeleton, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            keypoints = detection.getKeypoints()
            if len(keypoints) != 0:
                person_depth_m = min_depth

                smooth_joints = {}
                for j, keypoint in enumerate(keypoints):
                    if j >= NUM_JOINTS: break
                    kp_score = getattr(keypoint, 'score', getattr(keypoint, 'confidence', 1.0))
                    
                    if kp_score < 0.4:
                        smooth_joints[j] = (0.0, 0.0, 0.0)
                        continue

                    raw_x = keypoint.spatialCoordinates.x / 1000.0
                    raw_y = keypoint.spatialCoordinates.y / 1000.0
                    raw_z = keypoint.spatialCoordinates.z / 1000.0
                    if raw_z <= 0 and person_depth_m > 0: raw_z = person_depth_m

                    sm_x, sm_y, sm_z = joint_filter.update(j, raw_x, raw_y, raw_z)
                    smooth_joints[j] = (sm_x, sm_y, sm_z)

                base_x, base_y, base_z = 0.0, 0.0, 0.0
                if 11 in smooth_joints and 12 in smooth_joints:
                    if smooth_joints[11][2] > 0 and smooth_joints[12][2] > 0:
                        base_x = (smooth_joints[11][0] + smooth_joints[12][0]) / 2.0
                        base_y = (smooth_joints[11][1] + smooth_joints[12][1]) / 2.0
                        base_z = (smooth_joints[11][2] + smooth_joints[12][2]) / 2.0

                for j, keypoint in enumerate(keypoints):
                    if j >= NUM_JOINTS: break
                    keypoint_pos = frameNorm(frame, (keypoint.imageCoordinates.x, keypoint.imageCoordinates.y))
                    if j not in smooth_joints: continue
                    sm_x, sm_y, sm_z = smooth_joints[j]

                    if base_z > 0 and sm_z > 0:
                        norm_x = sm_x - base_x
                        norm_y = sm_y - base_y
                        norm_z = sm_z - base_z
                        current_frame_data[j] = [norm_x, norm_y, norm_z]
                    else:
                        current_frame_data[j] = [0.0, 0.0, 0.0]

                    cv2.circle(frame_black_skeleton, (keypoint_pos[0], keypoint_pos[1]), 3, (0, 255, 0), -1)

                for edge in YOLO26_EDGES:
                    if edge[0] < len(keypoints) and edge[1] < len(keypoints):
                        kp1_pos = frameNorm(frame, (keypoints[edge[0]].imageCoordinates.x, keypoints[edge[0]].imageCoordinates.y))
                        kp2_pos = frameNorm(frame, (keypoints[edge[1]].imageCoordinates.x, keypoints[edge[1]].imageCoordinates.y))
                        cv2.line(frame_black_skeleton, (kp1_pos[0], kp1_pos[1]), (kp2_pos[0], kp2_pos[1]), (0, 255, 0), 2)

        angle_energy = 0.0
        extracted_angle_sequence = None
        try:
            angle_frame = calculate_joint_angles(
                current_frame_data[np.newaxis, ...]
            )[0]
            angle_energy, extracted_angle_sequence = motion_trigger.process_frame(
                angle_frame
            )
        except (TypeError, ValueError):
            # 見切れや欠損で姿勢を構成できないフレームは状態を変更しない。
            pass

        if extracted_angle_sequence is not None:
            # TODO: 動作確認用の過渡的なCSV保存機能（エンコーダー実装時に削除予定）
            output_path = save_angle_sequence_csv(extracted_angle_sequence)
            print(
                f"動作データを保存しました: {output_path} "
                f"(フレーム数: {extracted_angle_sequence.shape[0]})"
            )
            # TODO: motion_encoder への角度配列連携は仕様確定後に実装する。

        angle_energy_history.append(angle_energy)

        # 画面描画
        graph_w, graph_h = 512, 200
        graph_img = np.zeros((graph_h, graph_w, 3), dtype=np.uint8) + 15
        
        state_str = "TRACKING (RECORDING)" if motion_trigger.current_state == 1 else "IDLE (WAITING)"
        state_color = (0, 0, 255) if motion_trigger.current_state == 1 else (0, 255, 0)
        cv2.putText(graph_img, f"STATUS: {state_str}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2)

        step_x = graph_w / SIGNAL_WINDOW_SIZE
        for i in range(1, len(angle_energy_history)):
            x1 = int((i - 1) * step_x)
            x2 = int(i * step_x)
            
            energy_y1 = int((graph_h - 20) - (angle_energy_history[i-1] * 20))
            energy_y2 = int((graph_h - 20) - (angle_energy_history[i] * 20))
            energy_y1, energy_y2 = np.clip(
                [energy_y1, energy_y2], 70, graph_h - 5
            )

            cv2.line(
                graph_img,
                (x1, energy_y1),
                (x2, energy_y2),
                (0, 255, 255),
                2,
            )

        cv2.putText(frame_black_skeleton, f"FPS: {fps_display:.1f}", (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        cv2.imshow("Skeleton Detection", frame_black_skeleton)
        cv2.imshow("Motion Trigger Monitor", graph_img)

    while pipeline.isRunning():
        inRgb = qRgb.tryGet()
        inDet = qDet.tryGet()

        if inRgb is not None: frame = inRgb.getCvFrame()
        if inDet is not None: detections = inDet.detections
            
        if frame is not None:
            process_and_display(frame, detections)
            frame = None 
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            pipeline.stop()
            break