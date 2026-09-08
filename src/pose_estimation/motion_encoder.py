import numpy as np


class MotionLanguageEncoder:
    def __init__(self):
        self.angle_labels = (
            ("neck_angle_deg", "NK"),
            ("left_shoulder_angle_deg", "LS"),
            ("right_shoulder_angle_deg", "RS"),
            ("left_elbow_angle_deg", "LE"),
            ("right_elbow_angle_deg", "RE"),
            ("left_hip_angle_deg", "LH"),
            ("right_hip_angle_deg", "RH"),
            ("left_knee_angle_deg", "LK"),
            ("right_knee_angle_deg", "RK"),
            ("head_pitch_deg", "HP"),
            ("head_roll_deg", "HR"),
            ("head_yaw_deg", "HY"),
        )
        self.angle_bounds = (0.0, 180.0)
        self.orientation_bounds = (-180.0, 180.0)

    def _quantize_value(self, val, bounds):
        min_v, max_v = bounds
        if not np.isfinite(val):
            return "xx"
        normalized = (np.clip(val, min_v, max_v) - min_v) / (max_v - min_v)
        return f"{int(normalized * 99.99):02d}"

    def encode_frame(self, angle_frame):
        """Encode one frame from the angle-series dictionary."""
        tokens = []
        for name, label in self.angle_labels:
            value = angle_frame.get(name, np.nan)
            bounds = self.orientation_bounds if name.startswith("head_") else self.angle_bounds
            tokens.append(f"{label}{self._quantize_value(value, bounds)}")
        return f"*{''.join(tokens)}#"

    def encode_sequence(self, angle_series):
        """Encode a dict of one-dimensional angle arrays into motion language."""
        missing = [name for name, _ in self.angle_labels if name not in angle_series]
        if missing:
            raise ValueError(f"missing angle series: {', '.join(missing)}")
        lengths = {len(angle_series[name]) for name, _ in self.angle_labels}
        if len(lengths) != 1:
            raise ValueError("all angle series must have the same number of frames")
        num_frames = lengths.pop()
        motion_language_lines = []
        for f_idx in range(num_frames):
            angle_frame = {
                name: angle_series[name][f_idx]
                for name, _ in self.angle_labels
            }
            frame_str = self.encode_frame(angle_frame)
            motion_language_lines.append(f"t={f_idx+1:02d}: {frame_str}")
        return "\n".join(motion_language_lines)


# --- 🧪 動作確認テスト用コード ---
if __name__ == "__main__":
    dummy_angles = {
        name: np.random.uniform(0.0, 180.0, size=10)
        for name, _ in MotionLanguageEncoder().angle_labels
    }
    
    encoder = MotionLanguageEncoder()
    result_text = encoder.encode_sequence(dummy_angles)
    
    print("=== 変換された動作言語データ（LLMに入力する文字列） ===")
    print(result_text)