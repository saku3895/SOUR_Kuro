#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import time
import argparse
import os
from datetime import datetime
from collections import deque

parser = argparse.ArgumentParser()
parser.add_argument("--depthSource", type=str, default="stereo", choices=["stereo", "neural"])
args = parser.parse_args()

device = dai.Device()
fps = 60
model_name = "luxonis/yolov8-nano-pose-estimation:coco-512x288"

# ✂️ 長いモデル名から「yolov8-nano」の部分を自動で抽出
raw_tag = model_name.split("/")[1]
model_tag = "-".join(raw_tag.split("-")[:2])

# 📁 ログ保存用フォルダの自動作成
LOG_DIR = "data_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 📋 コンパクトなCSVヘッダー
JOINT_LABELS = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow", "L_wrist", "R_wrist",
    "L_hip", "R_hip", "L_knee", "R_knee", "L_ankle", "R_ankle"
]

csv_header = ["timestamp", "frame_id", "fps", "latency_ms", "depth_m"]  
for i, label in enumerate(JOINT_LABELS):
    csv_header.extend([f"j{i:02d}_{label}_x", f"j{i:02d}_{label}_y", f"j{i:02d}_{label}_z"])

# メタデータとして1行目にモデル名を記録
CSV_HEADER_STR = f"# model_type: {model_tag}\n" + ",".join(csv_header) + "\n"


# ==========================================================
# 📈 移動平均フィルター用のデータ管理クラス
# ==========================================================
class JointFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = {i: deque(maxlen=window_size) for i in range(17)}
        self.last_valid = {i: (0.0, 0.0, 0.0) for i in range(17)}

    def update(self, joint_id, x, y, z):
        if z <= 0:
            if self.last_valid[joint_id][2] > 0:
                x, y, z = self.last_valid[joint_id]
            else:
                return x, y, z

        self.history[joint_id].append((x, y, z))
        self.last_valid[joint_id] = (x, y, z)

        arr = np.array(self.history[joint_id])
        mean_x = np.mean(arr[:, 0])
        mean_y = np.mean(arr[:, 1])
        mean_z = np.mean(arr[:, 2])

        return mean_x, mean_y, mean_z

joint_filter = JointFilter(window_size=5)
# ==========================================================

# Create pipeline
with dai.Pipeline(device) as pipeline:
    cameraNode = pipeline.create(dai.node.Camera).build(sensorFps=fps)
    
    monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=fps)
    monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=fps)
    monoLeftOut = monoLeft.requestOutput((640, 400), fps=fps)
    monoRightOut = monoRight.requestOutput((640, 400), fps=fps)

    requiredCamCapabilities = dai.ImgFrameCapability()
    requiredCamCapabilities.fps.fixed(fps)
    requiredCamCapabilities.enableUndistortion = True
    
    detectionNetwork = pipeline.create(dai.node.DetectionNetwork).build(cameraNode, model_name, requiredCamCapabilities)
    
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

    qRgb = detectionNetwork.passthrough.createOutputQueue(maxSize=1, blocking=False)
    qDet = spatialCalculator.outputDetections.createOutputQueue(maxSize=1, blocking=False)

    pipeline.start()

    frame = None
    detections = []
    
    # ⏱️ FPS計算用の時間管理
    last_frame_time = time.monotonic()
    fps_display = 0.0
    fps_smoothing_queue = deque(maxlen=10)

    # 💾 ロガー用の状態管理変数
    is_recording = False
    current_csv_file = None
    log_frame_id = 0
    prev_key = 0

    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    # 引数にラズパイ基準に同期された絶対時刻（device_epoch_time）を受け取る
    def displayFrame(name, frame, device_epoch_time=None):
        global is_recording, current_csv_file, log_frame_id, last_frame_time, fps_display

        # ⏱️ 現フレームの処理間隔から、このフレームの瞬間FPSを計算
        current_time = time.monotonic()
        frame_interval = current_time - last_frame_time
        last_frame_time = current_time
        
        instant_fps = 1.0 / frame_interval if frame_interval > 0 else 0.0
        fps_smoothing_queue.append(instant_fps)
        fps_display = np.mean(fps_smoothing_queue)

        frame_black_skeleton = np.zeros_like(frame)
        frame_raw_rgb = frame.copy()
        color = (255, 0, 0)
        
        for d_idx, detection in enumerate(detections):
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            
            cv2.putText(frame_black_skeleton, detection.labelName, (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame_black_skeleton, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            keypoints = detection.getKeypoints()
            if len(keypoints) == 0:
                continue

            person_depth_m = detection.spatialCoordinates.z / 1000.0

            # ⚙️ 1. 移動平均フィルターの適用
            smooth_joints = {}
            for j, keypoint in enumerate(keypoints):
                raw_x = keypoint.spatialCoordinates.x / 1000.0
                raw_y = keypoint.spatialCoordinates.y / 1000.0
                raw_z = keypoint.spatialCoordinates.z / 1000.0
                
                if raw_z <= 0 and person_depth_m > 0:
                    raw_z = person_depth_m

                sm_x, sm_y, sm_z = joint_filter.update(j, raw_x, raw_y, raw_z)
                smooth_joints[j] = (sm_x, sm_y, sm_z)

            # ⚙️ 2. 基準原点（腰の中心）の計算
            base_x, base_y, base_z = 0.0, 0.0, 0.0
            if 11 in smooth_joints and 12 in smooth_joints:
                if smooth_joints[11][2] > 0 and smooth_joints[12][2] > 0:
                    base_x = (smooth_joints[11][0] + smooth_joints[12][0]) / 2.0
                    base_y = (smooth_joints[11][1] + smooth_joints[12][1]) / 2.0
                    base_z = (smooth_joints[11][2] + smooth_joints[12][2]) / 2.0

            # ⚙️ 3. 正規化データの算術 ＆ 描画用ループ
            normalized_joint_data = {}
            for j, keypoint in enumerate(keypoints):
                keypoint_pos = frameNorm(frame, (keypoint.imageCoordinates.x, keypoint.imageCoordinates.y))
                sm_x, sm_y, sm_z = smooth_joints[j]

                if base_z > 0 and sm_z > 0:
                    norm_x = sm_x - base_x
                    norm_y = sm_y - base_y
                    norm_z = sm_z - base_z
                    normalized_joint_data[j] = (norm_x, norm_y, norm_z)
                else:
                    norm_z = 0.0
                    normalized_joint_data[j] = (0.0, 0.0, 0.0)

                cv2.circle(frame_black_skeleton, (keypoint_pos[0], keypoint_pos[1]), 3, (0, 255, 0), -1)
                rel_z_cm = int(norm_z * 100)
                label_text = f"({rel_z_cm:+.0f}cm)" if base_z > 0 and sm_z > 0 else "(0cm)"
                cv2.putText(frame_black_skeleton, label_text, (keypoint_pos[0] + 5, keypoint_pos[1] - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.35, (0, 255, 0))

            # 骨格線の描画
            for edge in detection.getEdges():
                if edge[0] < len(keypoints) and edge[1] < len(keypoints):
                    kp1 = keypoints[edge[0]]
                    kp2 = keypoints[edge[1]]
                    kp1_pos = frameNorm(frame, (kp1.imageCoordinates.x, kp1.imageCoordinates.y))
                    kp2_pos = frameNorm(frame, (kp2.imageCoordinates.x, kp2.imageCoordinates.y))
                    cv2.line(frame_black_skeleton, (kp1_pos[0], kp1_pos[1]), (kp2_pos[0], kp2_pos[1]), (0, 255, 0), 2)

            # 💾 4. CSVへのリアルタイム書き込み
            if is_recording and current_csv_file is not None:
                log_frame_id += 1
                
                # 🔥【基準を完全同期させた引き算！】
                # どちらも1970年基準の「秒単位のfloat」なので、引いて1000倍するだけで正確な遅延(ms)が出ます
                if device_epoch_time is not None:
                    latency_ms = (time.time() - device_epoch_time) * 1000
                else:
                    latency_ms = 0.0

                # 正常値の範囲（およそ20ms〜300ms程度）に収まった綺麗な遅延が直接書き込まれます
                row_data = [
                    f"{time.time():.3f}", 
                    str(log_frame_id), 
                    f"{instant_fps:.1f}", 
                    f"{latency_ms:.1f}",  # 🔥 ここに見やすい遅延(~ms)が入ります
                    f"{person_depth_m:.2f}"
                ]
                
                for j in range(17):
                    nx, ny, nz = normalized_joint_data.get(j, (0.0, 0.0, 0.0))
                    row_data.extend([f"{nx:+.3f}", f"{ny:+.3f}", f"{nz:+.3f}"])
                
                current_csv_file.write(",".join(row_data) + "\n")

        if is_recording:
            cv2.putText(frame_black_skeleton, f"● REC [{model_tag.upper()}] - PRESS SPACE TO STOP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame_raw_rgb, f"● REC [{model_tag.upper()}]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame_black_skeleton, f"STANDBY [{model_tag.upper()}] - PRESS SPACE TO START", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        fps_str = f"FPS: {fps_display:.1f}"
        cv2.putText(frame_black_skeleton, fps_str, (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame_raw_rgb, fps_str, (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.imshow("Detections", frame_black_skeleton)
        cv2.imshow("Raw_RGB", frame_raw_rgb)

    while pipeline.isRunning():
        inRgb = qRgb.tryGet()
        inDet = qDet.tryGet()

        device_epoch_time = None
        if inRgb is not None:
            frame = inRgb.getCvFrame()
            # 🔥【ここを変更！】 getTimestampDevice() を使うことで、
            # カメラの時間軸を自動的に「ラズパイのエポック秒（絶対時刻）」に変換して同期させます
            device_epoch_time = inRgb.getTimestampDevice().total_seconds()

        if inDet is not None:
            detections = inDet.detections

        if frame is not None:
            displayFrame("Detections", frame, device_epoch_time)
            
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32 and prev_key != 32:
            if not is_recording:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(LOG_DIR, f"motion_{model_tag}_{timestamp_str}.csv")
                current_csv_file = open(filename, "w", encoding="utf-8")
                current_csv_file.write(CSV_HEADER_STR)
                log_frame_id = 0
                is_recording = True
                print(f"🎬 ログ保存を開始しました: {filename}")
            else:
                is_recording = False
                if current_csv_file is not None:
                    current_csv_file.close()
                    current_csv_file = None
                print("🛑 ログ保存を終了しました。")
        
        prev_key = key

        if key == ord("q"):
            if current_csv_file is not None:
                current_csv_file.close()
            pipeline.stop()
            break