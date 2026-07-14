import numpy as np
import json
import requests

from ollama import Client

class RobotTaskPlanner:
    def __init__(self):
        """
        LLMを用いた意図推測および行動目標生成モジュール
        """
        self.model_name = 'llama3.1:8b'
        
        # 🔥 確認したPCのIPアドレス「172.28.192.187」をここに設定
        self.pc_ip = 'http://192.168.11.5:11434' 
        self.client = Client(host=self.pc_ip)

    def downsample_gesture_data(self, data_package, target_frames=15):
        """
        【データ軽量化】時系列データを均等に間引く
        """
        total_frames = len(data_package)
        if total_frames <= target_frames:
            return data_package
        indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        return data_package[indices]

    def generate_action_plan(self, raw_gesture_data):
        """
        時系列データからプロンプトを組み立て、外部サーバーのLLMへ送信するメイン関数
        """
        print("🧠 [Robot Planner] 骨格時系列データの計画・解析フェーズを開始します...")
        
        # 1. データの軽量化（12フレームに間引き）
        compressed_data = self.downsample_gesture_data(raw_gesture_data, target_frames=12)
        
        # 2. 腕関節（5,6,7,8,9,10）の抽出とJSONテキスト化
        arm_indices = [5, 6, 7, 8, 9, 10]
        simplified_motion = []
        for frame_idx, frame in enumerate(compressed_data):
            frame_dict = {"frame": frame_idx}
            for idx in arm_indices:
                frame_dict[f"joint_{idx}"] = np.round(frame[idx], 3).tolist()
            simplified_motion.append(frame_dict)
            
        motion_text = json.dumps(simplified_motion, indent=2)

        # 3. プロンプト（指示文）の組み立て
        prompt = f"""
あなたは猫型4脚ロボットの「意図推測・行動計画エンジン」です。
人間が行った動作の3次元骨格時系列データを解析し、その動作の潜在的な意図を推測して、ロボットが次に起こすべき行動目標を設定してください。

【入力データ】
- 腕関節の時系列変化:
{motion_text}

【出力フォーマット】
以下のJSON形式のみで回答してください。余計な挨拶や解説は一切含めないでください。
{{
  "estimated_intent": "推測される人間の潜在的意図",
  "action_goal": "ロボットが次に行うべき具体的な行動目標"
}}
"""

        # 4. 🔥 【先輩の方式を流用】 requests を使って外部のLLMサーバーへ送信
        # ここでは一般的なOllamaのAPI形式（Llama3等を想定）のデータ構造にしています
        payload = {
            "model": "llama3",       # 使用するモデル名（Ollamaに入っているもの）
            "prompt": prompt,
            "stream": False,         # 返答を細切れにせず、一括で受け取る設定
            "format": "json"         # 出力をJSONに固定するOllamaの機能
        }

        try:
            print(f"🌐 [Robot Planner] 外部PCのOllama ({self.pc_ip}) にデータを送信中...")
            
            # 🔥 self.client.chat を使うことで、ラズパイではなくPC側でLLMが実行されます！
            response = self.client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
            )
            
            llm_response_text = response.message.content
            print("\n--- [LLM 推測結果] ---")
            print(llm_response_text)
            print("----------------------\n")
            return llm_response_text
            
        except Exception as e:
            print(f"❌ PCのOllamaとの通信でエラーが発生しました: {e}")
            return None