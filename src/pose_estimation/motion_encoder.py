import numpy as np

class MotionLanguageEncoder:
    def __init__(self):
        # 1. 使用する7つの主要関節とIDの対応（YOLO Pose 17関節のインデックス）
        self.target_joints = [
            (0, 'a'),   # Nose (頭部/首の傾き)
            (5, 'b'),   # L_Shoulder (左肩)
            (6, 'c'),   # R_Shoulder (右肩)
            (7, 'd'),   # L_Elbow (左肘)
            (8, 'e'),   # R_Elbow (右肘)
            (9, 'f'),   # L_Wrist (左手首)
            (10, 'g')   # R_Wrist (右手首)
        ]
        
        # 2. 腰相対XYZ座標(メートル)を 0〜9 の10段階に変換するためのスケール定義
        # [min_val, max_val] の範囲外は 0 や 9 にクリップ（丸め処理）されます
        self.bounds_x = (-0.8, 0.8)   # X軸: 左右 (-0.8m:左端 〜 +0.8m:右端)
        self.bounds_y = (-0.2, 0.8)   # Y軸: 上下 (-0.2m:腰下 〜 +0.8m:頭上)
        self.bounds_z = (-0.8, 0.8)   # Z軸: 奥行 (-0.8m:手前 〜 +0.8m:奥)

    def _quantize_value(self, val, bounds):
        """実数値(float)を 0〜9 の整数(int)に量子化する関数"""
        min_v, max_v = bounds
        # 範囲内にクリップして 0.0 〜 1.0 に正規化
        norm = (np.clip(val, min_v, max_v) - min_v) / (max_v - min_v)
        # 0 〜 9 の整数値に変換
        quantized = int(norm * 9.99)
        return quantized

    def encode_frame(self, frame_data):
        """1フレーム分 (17, 3) の座標データを1コマの動作言語文字列に変換"""
        tokens = []
        for j_idx, label in self.target_joints:
            x, y, z = frame_data[j_idx]
            
            # 見切れ等でデータが完全な(0, 0, 0)の場合は 555 (中央) として埋める
            if x == 0.0 and y == 0.0 and z == 0.0:
                qx, qy, qz = 5, 5, 5
            else:
                qx = self._quantize_value(x, self.bounds_x)
                qy = self._quantize_value(y, self.bounds_y)
                qz = self._quantize_value(z, self.bounds_z)
            
            # 例: 'f' + '1' + '4' + '4' -> "f144"
            tokens.append(f"{label}{qx}{qy}{qz}")
            
        # 先行研究にならい '*' で始まり '#' で閉じる
        return f"*{''.join(tokens)}#"

    def encode_sequence(self, data_package):
        """
        切り出された時系列データ (Frames, 17, 3) 全体を
        LLMに入力する動作言語テキストに一括変換する
        """
        num_frames = data_package.shape[0]
        motion_language_lines = []
        
        for f_idx in range(num_frames):
            frame_str = self.encode_frame(data_package[f_idx])
            # 時系列のインデックスを付けて並べる
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