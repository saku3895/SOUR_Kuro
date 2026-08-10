import numpy as np

class MotionLanguageEncoder:
    def __init__(self):
        self.target_joints = [
            (0, 'a'), (5, 'b'), (6, 'c'), (7, 'd'), (8, 'e'), (9, 'f'), (10, 'g')
        ]
        self.bounds_x = (-0.8, 0.8)
        self.bounds_y = (-0.6, 0.8)  # ⚡ 上がプラス(+0.8m)、下がマイナス(-0.6m)
        self.bounds_z = (-0.8, 0.8)

    def _quantize_value(self, val, bounds):
        min_v, max_v = bounds
        norm = (np.clip(val, min_v, max_v) - min_v) / (max_v - min_v)
        return int(norm * 9.99)

    def encode_frame(self, frame_data):
        tokens = []
        for j_idx, label in self.target_joints:
            x, y, z = frame_data[j_idx]
            if x == 0.0 and y == 0.0 and z == 0.0:
                qx, qy, qz = 5, 5, 5
            else:
                qx = self._quantize_value(x, self.bounds_x)
                
                # ⚡ カメラのY（下向き）を反転させて「上がプラス」の高さにする
                qy = self._quantize_value(-y, self.bounds_y)
                
                qz = self._quantize_value(z, self.bounds_z)
            tokens.append(f"{label}{qx}{qy}{qz}")
        return f"*{''.join(tokens)}#"

    def encode_sequence(self, data_package):
        num_frames = data_package.shape[0]
        motion_language_lines = []
        for f_idx in range(num_frames):
            frame_str = self.encode_frame(data_package[f_idx])
            motion_language_lines.append(f"t={f_idx+1:02d}: {frame_str}")
        return "\n".join(motion_language_lines)


# --- 🧪 動作確認テスト用コード ---
if __name__ == "__main__":
    # ダミーデータ生成: 10フレーム分、17関節、3次元座標
    dummy_data = np.random.uniform(-0.5, 0.5, size=(10, 17, 3))
    
    encoder = MotionLanguageEncoder()
    result_text = encoder.encode_sequence(dummy_data)
    
    print("=== 変換された動作言語データ（LLMに入力する文字列） ===")
    print(result_text)