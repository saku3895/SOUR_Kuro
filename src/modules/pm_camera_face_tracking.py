import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request
import time
import copy

from src.core.periodic_module import PeriodicModule

class PMCameraFaceTracking(PeriodicModule):
    def __init__(self, robot_properties, interval_ms=40, kp=0.05):
        super().__init__(interval_ms)
        self.robot_properties = robot_properties
        self.kp = kp  # 首振りの感度（ハンチングしたら小さく、遅すぎたら大きく調整）
        
        # 💡 SOURの初期位置設定からKuroの全関節（14個）の姿勢を安全にコピー
        self.base_positions = copy.deepcopy(robot_properties.servo_initial_positions)

        # 💡 首の現在のパルス位置を保持（初期値はKuroの初期位置: ID2=1500, ID3=1330）
        self.face_pan_pulse = self.base_positions[0]   # インデックス0番 = ID 2 (左右)
        self.face_tilt_pulse = self.base_positions[1]  # インデックス1番 = ID 3 (上下)

        self.detector = None
        self.cap = None
        self.is_initialized = False

    def _initialize_camera_and_ai(self):
        """ SOURの独立プロセス内で、安全にAIとカメラを初期化する """
        print("[*] SOUR独立プロセス内で AI & カメラの起動を開始します...")
        
        # 1. MediaPipeの準備
        model_path = 'blaze_face_short_range.tflite'
        if not os.path.exists(model_path):
            print("[*] MediaPipe AIモデルをダウンロード中...")
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            urllib.request.urlretrieve(url, model_path)

        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=python.BaseOptions.Delegate.CPU
        )
        options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
        self.detector = vision.FaceDetector.create_from_options(options)

        # 2. 💡【超重要】カメラがビジーから復帰するのを待つリトライループ
        # SOUR起動直後はLinuxのデバイスアクセスが非常に混み合うため、じっくり待ちます
        for retry in range(5):
            print(f"[*] USBカメラ(0番)へ接続を試みています... ({retry+1}/5)")
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            
            if self.cap.isOpened():
                success, _ = self.cap.read()
                if success:
                    print("[+] USBカメラ(0番)のキャプチャに成功しました！")
                    break
            
            # ダメなら一回手放して1秒待つ
            self.cap.release()
            time.sleep(1.0)

        # 3. 💡 0番がどうしてもダメだったときの保険（1番ポート）
        if not self.cap.isOpened():
            print("[⚠️警告] 0番ポートが全滅したため、1番ポートで試行します...")
            self.cap = cv2.VideoCapture(1, cv2.CAP_V4L2)

        # 最終チェック（どちらも開かなかったらログを出す）
        if not self.cap.isOpened():
            print("[❌致命的エラー] カメラデバイスを完全にロックされて開けませんでした。")
            print("一度『pkill -f python』を実行するか、USBカメラを抜き挿ししてください。")
            return  # is_initialized を True にせずここで止める

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.is_initialized = True
        print("[+] Kuro顔追従システムが完全に連動しました！")

    def execute_periodic_task(self, lock, data_dict):
        super().execute_periodic_task(lock, data_dict)
        
        # 終了処理
        if self.terminate:
            if self.cap: self.cap.release()
            cv2.destroyAllWindows()
            return

        # 初回初期化
        if not self.is_initialized:
            self._initialize_camera_and_ai()
            return

        # 1. カメラ画像から顔の座標を検出して、首の目標パルスを計算
        if self.cap and self.cap.isOpened():
            success, image = self.cap.read()
            if success:
                h, w, _ = image.shape
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                detection_result = self.detector.detect(mp_image)

                if detection_result.detections:
                    for detection in detection_result.detections:
                        bbox = detection.bounding_box
                        
                        # 顔の中心と画面中央からのズレを計算
                        center_x = bbox.origin_x + (bbox.width // 2)
                        center_y = bbox.origin_y + (bbox.height // 2)
                        error_x = center_x - (w // 2)
                        error_y = center_y - (h // 2)

                        # 💡 ズレを打ち消す方向に首のパルスをじわじわ更新
                        # カメラの反転特性に合わせて符号（-=, +=）を調整しています
                        self.face_pan_pulse -= error_x * self.kp
                        self.face_tilt_pulse += error_y * self.kp

                        # 💡 Kuroの可動範囲（Propertiesの設定値）を超えないように絶対ガード
                        self.face_pan_pulse = max(1000.0, min(2000.0, self.face_pan_pulse))
                        self.face_tilt_pulse = max(1010.0, min(1550.0, self.face_tilt_pulse))
                        
                        # デバッグ表示
                        # print(f"【追従制御中】Pan(ID2): {int(self.face_pan_pulse)} | Tilt(ID3): {int(self.face_tilt_pulse)}")
                        break

                # cv2.imshow('Kuro Face Tracking Lock', image)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     print("[*] 「q」キーが押されたため、安全な終了処理を開始します...")
                #     if self.cap:
                #         self.cap.release()
                #     cv2.destroyAllWindows()
                    
                #     # ゾンビ化を防ぐため、この子プロセス（自分自身）を確実に終了させる
                #     import os
                #     import signal
                #     os.kill(os.getpid(), signal.SIGKILL)

        # 2. 💡 SOURの共有メモリ（data_dict）を通してロボット全体に命令を送信
        if data_dict['servo_ready'] == True and data_dict['servo_params_updated'] == False:
            # 全身の初期姿勢をベースにする（手足は初期位置でピタッとロックされる）
            next_command_positions = copy.deepcopy(self.base_positions)
            
            # 💡 首の2つのモーターだけ、今計算した新しいパルス値に上書き！
            next_command_positions[0] = self.face_pan_pulse  # ID 2
            next_command_positions[1] = self.face_tilt_pulse # ID 3

            # 40msかけてその位置へ動かす
            # operation_times = [float(self.interval_ms)] * len(next_command_positions)
            operation_times = [20] * len(next_command_positions)
            
            # SOURの正規ルートで書き込み
            self.write_servo_positions(lock, data_dict, next_command_positions, operation_times)