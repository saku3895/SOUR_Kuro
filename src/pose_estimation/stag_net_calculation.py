import numpy as np
from collections import deque

class STAGNetDynamicsEngine:
    def __init__(self, 
                 threshold_rotation=0.015,
                 threshold_position=0.030,
                 min_trigger_frames=6,
                 stability_frames=8):
        
        self.threshold_rotation = threshold_rotation
        self.threshold_position = threshold_position
        self.min_trigger_frames = min_trigger_frames
        self.stability_frames = stability_frames
        
        # 0: IDLE, 1: TRACKING
        self.current_state = 0
        self.gesture_buffer = []
        
        self.above_threshold_counter = 0
        self.below_threshold_counter = 0
        
        self.prev_bone_vectors = {}
        self.prev_joint_positions = None
        
        self.pos_energy_queue = deque(maxlen=10)
        self.rot_energy_queue = deque(maxlen=10)
        self.last_position_energy = 0.0
        self.last_rotation_energy = 0.0

    def calculate_bone_vector(self, p, c):
        d = c - p
        norm = np.linalg.norm(d)
        if norm < 1e-5:
            return np.zeros(3)
        return d / norm

    def process_frame(self, current_frame_data):
        """
        1フレーム分のデータを処理し、(rot_energy, extracted_gesture_data) を返す。
        動作完了時（TRIGGER OFF）のみ extracted_gesture_data にNumPy配列が入り、それ以外は None となる。
        """
        extracted_data = None

        if np.all(current_frame_data == 0):
            self.last_position_energy = 0.0
            self.last_rotation_energy = 0.0
            return 0.0, None

        # 1. 位置エネルギー
        raw_position_energy = 0.0
        arm_joints = [5, 6, 7, 8, 9, 10]
        if self.prev_joint_positions is not None:
            diff = current_frame_data[arm_joints] - self.prev_joint_positions[arm_joints]
            raw_position_energy = float(np.sum(diff ** 2))
        self.prev_joint_positions = current_frame_data.copy()

        # 2. 角度エネルギー
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

        self.pos_energy_queue.append(raw_position_energy)
        self.rot_energy_queue.append(raw_rotation_energy)
        
        smooth_position_energy = np.mean(self.pos_energy_queue)
        smooth_rotation_energy = np.mean(self.rot_energy_queue)
        self.last_position_energy = float(smooth_position_energy)
        self.last_rotation_energy = float(smooth_rotation_energy)

        # 閾値ヒステリシス
        if self.current_state == 1:
            th_rot = self.threshold_rotation * 1.2
            th_pos = self.threshold_position * 1.2
        else:
            th_rot = self.threshold_rotation
            th_pos = self.threshold_position

        is_moving = (smooth_rotation_energy > th_rot) or (smooth_position_energy > th_pos)
        
        # 3. ステートマシン判定
        if self.current_state == 0:  # IDLE
            if is_moving:
                self.above_threshold_counter += 1
                if self.above_threshold_counter >= self.min_trigger_frames:
                    self.current_state = 1  # TRACKING
                    self.gesture_buffer = [current_frame_data.copy()]
                    self.below_threshold_counter = 0
                    print(f"\n⚡ [TRIGGER ON] 記録開始")
            else:
                self.above_threshold_counter = 0
                
        elif self.current_state == 1:  # TRACKING
            self.gesture_buffer.append(current_frame_data.copy())
            
            if not is_moving:
                self.below_threshold_counter += 1
                if self.below_threshold_counter >= self.stability_frames:
                    print(f"🛑 [TRIGGER OFF] 記録終了")
                    extracted_data = np.array(self.gesture_buffer)
                    
                    self.current_state = 0
                    self.gesture_buffer = []
                    self.above_threshold_counter = 0
                    self.below_threshold_counter = 0
            else:
                self.below_threshold_counter = 0

        return smooth_rotation_energy, extracted_data