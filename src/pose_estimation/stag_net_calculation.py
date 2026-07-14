import numpy as np
from collections import deque
from robot_planning import RobotTaskPlanner

class STAGNetDynamicsEngine:
    def __init__(self, threshold_rotation=0.002, threshold_position=0.005, stability_frames=3):
        """
        STAG-Net思想に基づく軽量ダイナミクス計算 ＆ プルプル対策付き自動トリガーステートマシン
        """
        self.threshold_rotation = threshold_rotation
        self.threshold_position = threshold_position
        self.stability_frames = stability_frames  # スムージングを入れるため、短く設定可能に（8 → 3など）
        
        # ステートマシンの状態定義
        self.current_state = 0  # 0: IDLE, 1: TRACKING
        self.gesture_buffer = []
        self.below_threshold_counter = 0
        
        # 1フレーム前のデータ保持
        self.prev_bone_vectors = {}
        self.prev_joint_positions = None
        
        # 骨格プルプル（ジッター）対策用の移動平均バッファ (直近3フレーム)
        self.pos_energy_queue = deque(maxlen=3)
        self.rot_energy_queue = deque(maxlen=3)

    def calculate_bone_vector(self, p, c):
        d = c - p
        norm = np.linalg.norm(d)
        if norm < 1e-5:
            return np.zeros(3)
        return d / norm

    def process_frame(self, current_frame_data):
        if np.all(current_frame_data == 0):
            return 0.0

        # --- 1. 位置エネルギーの計算 ---
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

        # 🔥 【プルプル対策①】直近3フレームの移動平均をとってノイズの突発的なトゲを潰す
        self.pos_energy_queue.append(raw_position_energy)
        self.rot_energy_queue.append(raw_rotation_energy)
        
        smooth_position_energy = np.mean(self.pos_energy_queue)
        smooth_rotation_energy = np.mean(self.rot_energy_queue)

        # 🔥 【プルプル対策②】ヒステリシス（閾値の動的変更）
        # 動いている間(TRACKING)は、プルプルを「静止」と見なしやすくするために閾値を少し厳しく（高めに）する
        if self.current_state == 1:  # TRACKING中
            th_rot = self.threshold_rotation * 1.5  # 例: 0.002 → 0.003 に引き上げ
            th_pos = self.threshold_position * 1.5
        else:
            th_rot = self.threshold_rotation
            th_pos = self.threshold_position

        # 判定
        is_moving = (smooth_rotation_energy > th_rot) or (smooth_position_energy > th_pos)
        
        # --- 3. ステートマシン判定 ---
        if self.current_state == 0:  # STATE_IDLE
            if is_moving:
                self.current_state = 1  # STATE_TRACKING
                self.gesture_buffer = [current_frame_data.copy()]
                self.below_threshold_counter = 0
                print(f"⚡ [TRIGGER ON] 仕草検知開始! (Smooth Rot: {smooth_rotation_energy:.5f})")
                
        elif self.current_state == 1:  # STATE_TRACKING
            self.gesture_buffer.append(current_frame_data.copy())
            
            if not is_moving:
                self.below_threshold_counter += 1
                # スムージングが効いているので、ここを短いフレーム数（例: 3フレーム＝約0.05秒）にしても誤検知しにくい
                if self.below_threshold_counter >= self.stability_frames:
                    print(f"🛑 [TRIGGER OFF] 仕草終了 (総フレーム数: {len(self.gesture_buffer)})")
                    packed_data = np.array(self.gesture_buffer)
                    self._send_to_mllm_pipeline(packed_data)
                    
                    self.current_state = 0
                    self.gesture_buffer = []
                    self.below_threshold_counter = 0
            else:
                self.below_threshold_counter = 0

        # get_keypoint.pyのグラフ描画用には、滑らかにしたねじれ値を返す
        return smooth_rotation_energy

    def _send_to_mllm_pipeline(self, data_package):
        print(f"🚀 [MLLM Pipeline] {data_package.shape} の動作データをパッキングしました。")
        """
        時系列計算が終わった塊データを、新設した robot_planning.py へバトンタッチ
        """
        # ここで新設した計画モジュールを呼び出す
        planner = RobotTaskPlanner()
        planner.generate_action_plan(data_package)