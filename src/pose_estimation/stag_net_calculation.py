import numpy as np
from collections import deque

class STAGNetDynamicsEngine:
    def __init__(self, 
                 threshold_rotation=0.015,     # ⚡ 閾値をアップ (0.002 → 0.015)
                 threshold_position=0.030,     # ⚡ 閾値をアップ (0.005 → 0.030)
                 min_trigger_frames=6,         # ⚡ 【新機能】動作開始の「溜め」判定 (約0.2秒間動きが続いたらスタート)
                 stability_frames=8):          # ⚡ 停止判定のフレーム数 (少し長めにして誤OFFを防ぐ)
        
        self.threshold_rotation = threshold_rotation
        self.threshold_position = threshold_position
        self.min_trigger_frames = min_trigger_frames
        self.stability_frames = stability_frames
        
        # ステートマシンの状態定義 (0: IDLE, 1: TRACKING)
        self.current_state = 0
        self.gesture_buffer = []
        
        # カウンター類
        self.above_threshold_counter = 0  # 動作開始用の溜めカウンター
        self.below_threshold_counter = 0  # 動作終了用の停止カウンター
        
        # 1フレーム前のデータ保持
        self.prev_bone_vectors = {}
        self.prev_joint_positions = None
        
        # ⚡ ノイズ除去用の移動平均バッファを 3 → 10 に拡大 (カメラのゆらぎを潰す)
        self.pos_energy_queue = deque(maxlen=10)
        self.rot_energy_queue = deque(maxlen=10)

    def calculate_bone_vector(self, p, c):
        d = c - p
        norm = np.linalg.norm(d)
        if norm < 1e-5:
            return np.zeros(3)
        return d / norm

    def process_frame(self, current_frame_data):
        if np.all(current_frame_data == 0):
            return 0.0

        # --- 1. 位置エネルギーの計算 (主要な腕関節) ---
        raw_position_energy = 0.0
        arm_joints = [5, 6, 7, 8, 9, 10]
        if self.prev_joint_positions is not None:
            diff = current_frame_data[arm_joints] - self.prev_joint_positions[arm_joints]
            raw_position_energy = float(np.sum(diff ** 2))
        self.prev_joint_positions = current_frame_data.copy()

        # --- 2. 角度・ねじれエネルギーの計算 ---
        raw_rotation_energy = 0.0
        bone_pairs = [(5, 7), (7, 9), (6, 8), (8, 10)]
        for parent, child in bone_pairs:
            p_coord = current_frame_data[parent]
            c_coord = current_frame_data[child]
            if np.all(p_coord == 0) or np.all(c_coord == 0):
                continue
            curr_vec = self.calculate_bone_vector(p_coord, c_coord)
            bone_key = (parent, child)
            if bone_key in self.prev_bone_vectors:
                prev_vec = self.prev_bone_vectors[bone_key]
                raw_rotation_energy += float(np.sum((curr_vec - prev_vec) ** 2))
            self.prev_bone_vectors[bone_key] = curr_vec

        # 直近10フレームの移動平均でノイズを滑らかにする
        self.pos_energy_queue.append(raw_position_energy)
        self.rot_energy_queue.append(raw_rotation_energy)
        
        smooth_position_energy = np.mean(self.pos_energy_queue)
        smooth_rotation_energy = np.mean(self.rot_energy_queue)

        # 動作中のヒステリシス
        if self.current_state == 1:
            th_rot = self.threshold_rotation * 1.2
            th_pos = self.threshold_position * 1.2
        else:
            th_rot = self.threshold_rotation
            th_pos = self.threshold_position

        # 「動き」があったかの判定
        is_moving = (smooth_rotation_energy > th_rot) or (smooth_position_energy > th_pos)
        
        # --- 3. ステートマシン判定 (溜め判定つき) ---
        if self.current_state == 0:  # STATE_IDLE (待機中)
            if is_moving:
                self.above_threshold_counter += 1
                # ⚡ 一定フレーム以上連続で動いた場合のみ TRIGGER ON
                if self.above_threshold_counter >= self.min_trigger_frames:
                    self.current_state = 1  # STATE_TRACKING
                    self.gesture_buffer = [current_frame_data.copy()]
                    self.below_threshold_counter = 0
                    print(f"\n⚡ [TRIGGER ON] 意図的な動作を検知! (Rot: {smooth_rotation_energy:.4f}, Pos: {smooth_position_energy:.4f})")
            else:
                self.above_threshold_counter = 0  # 途中で動きが切れたらリセット
                
        elif self.current_state == 1:  # STATE_TRACKING (記録中)
            self.gesture_buffer.append(current_frame_data.copy())
            
            if not is_moving:
                self.below_threshold_counter += 1
                if self.below_threshold_counter >= self.stability_frames:
                    print(f"🛑 [TRIGGER OFF] 動作終了")
                    packed_data = np.array(self.gesture_buffer)
                    
                    self._on_gesture_extracted(packed_data)
                    
                    self.current_state = 0
                    self.gesture_buffer = []
                    self.above_threshold_counter = 0
                    self.below_threshold_counter = 0
            else:
                self.below_threshold_counter = 0

        return smooth_rotation_energy

    def _on_gesture_extracted(self, data_package):
        frames, joints, coords = data_package.shape
        print(f"📦 【動作切り出し成功】 総フレーム数 : {frames} (約 {frames / 30.0:.2f} 秒)\n")