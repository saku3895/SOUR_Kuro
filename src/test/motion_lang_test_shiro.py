from src.core.periodic_module import PeriodicModule
from src.communication.recieve_motion_lang import receive_gmlp
from src.communication.lobot_serial_communication import LobotServoController
from src.control.motion_lang2motor import parse_motion_commands

import sys, os, time
import random


class MotionLangTestShiro(PeriodicModule):
    def __init__(self, robot_properties, interval_ms=1000):
        super().__init__(interval_ms)
        self.robot_properties = robot_properties
        self.motion_step = 0
        self.motor_commands = []
        self.is_initialized = False
        self.current_motion_data = None  # 現在のモーションデータを保存

    def initialize_motion_commands(self):
        """モーションコマンドを初期化（初回のみ）"""
        initial_commands = ('1', ['*a6b5c4d8e0f4g6h4i0j4k4l5m1n5o5p8q2r5s5t9#'])
        motor_initial = parse_motion_commands(initial_commands)
        
        # 初期ポーズのみを設定
        self.motor_commands = motor_initial
        self.is_initialized = True
        print(f"初期ポーズ設定完了: {len(self.motor_commands)}個のコマンド")

    def generate_new_motion(self):
        """新しいモーションを生成"""
        print("=== 新しいモーション生成中 ===")
        
        # 初回は固定文字列、2回目以降は前回のmotion_dataを使用
        if self.current_motion_data is None:
            input_string = "*a6b5c4d8e0f4g6h4i0j4k4l5m1n5o5p8q2r5s5t9#"
        else:
            # motion_dataのlistを文字列に結合
            input_string = ''.join(self.current_motion_data[1])
            print(f"前回のモーションデータを使用: {input_string}")
        
        motion_data = receive_gmlp(input_string)
        self.current_motion_data = motion_data  # 次回のために保存
        new_motor_commands = parse_motion_commands(motion_data)
        
        # 新しいモーションコマンドを設定
        self.motor_commands = new_motor_commands
        print(f"新しいモーション生成完了: {len(self.motor_commands)}個のコマンド")

    def execute_periodic_task(self, lock, data_dict):
        #if all servos are not ready, return
        if data_dict['servo_ready'] == False:
            return

        # 初回実行時にモーションコマンドを初期化
        if not self.is_initialized:
            self.initialize_motion_commands()
            return

        # モーションコマンドがある場合は実行
        if self.motion_step < len(self.motor_commands):
            cmd = self.motor_commands[self.motion_step]
            if cmd:
                print(f"モーション実行: ステップ{self.motion_step + 1} - {len(cmd)}個のサーボ")
                
                # コマンドをSOURシステムの形式に変換
                servo_positions = list(self.robot_properties.servo_initial_positions)
                servo_operation_times = [1000.0] * self.robot_properties.num_servos
                
                # cmdの各サーボ情報を適用
                for servo_id, position in cmd:
                    if servo_id in self.robot_properties.servo_ids:
                        index = self.robot_properties.servo_ids.index(servo_id)
                        servo_positions[index] = float(position)
                
                self.write_servo_positions(lock, data_dict, servo_positions, servo_operation_times)
                self.motion_step += 1
            else:
                self.motion_step += 1
        else:
            # 全てのモーションが完了したら新しいモーションを生成
            self.motion_step = 0
            print("モーションシーケンス完了 - 新しいモーション生成開始")
            self.generate_new_motion()
