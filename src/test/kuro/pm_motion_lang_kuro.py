from src.control.motion_lang2motor import parse_motion_commands
from src.core.periodic_module import PeriodicModule

import queue
import threading


class PMMotionLanguageKuro(PeriodicModule):
    """1行のKuro動作言語を1つのモータ姿勢として実行する周期モジュール。"""

    def __init__(self, robot_properties, motion_language_line=None, interval_ms=1000):
        super().__init__(interval_ms)
        self.robot_properties = robot_properties
        self.motion_language_line = motion_language_line
        self.servo_positions = None
        self.has_written = True
        self.input_queue = queue.Queue()
        self.input_thread_started = False

    @staticmethod
    def _normalize_motion_language(motion_language_line):
        if not isinstance(motion_language_line, str):
            raise TypeError("motion_language_line must be a string")

        line = motion_language_line.strip()
        command_start = line.find("*")
        command_end = line.find("#", command_start)
        if command_start < 0 or command_end < 0:
            raise ValueError("動作言語は * で始まり # で終わる1行を指定してください")

        command = line[command_start:command_end + 1]
        if line[command_end + 1:].strip():
            raise ValueError("動作言語は1行のコマンドだけを指定してください")
        return command

    def _parse_motion_language(self):
        motion_language_line = self._normalize_motion_language(
            self.motion_language_line
        )
        parsed_commands = parse_motion_commands(
            ("1", [motion_language_line])
        )
        if len(parsed_commands) != 1 or not parsed_commands[0]:
            raise ValueError("動作言語から有効なモータ指令を生成できません")

        servo_positions = list(self.robot_properties.servo_initial_positions)
        if len(servo_positions) != self.robot_properties.num_servos:
            raise ValueError("servo_initial_positions の要素数が num_servos と一致しません")

        for servo_id, position in parsed_commands[0]:
            if servo_id in self.robot_properties.servo_ids:
                servo_index = self.robot_properties.servo_ids.index(servo_id)
                servo_positions[servo_index] = float(position)

        return servo_positions

    def _read_motion_language(self):
        print(
            "動作言語を入力してください（例: *a4d6e4q6i6m1n1f3r3h4l4j2p4t3#）",
            flush=True,
        )
        while not self.terminate:
            try:
                motion_language_line = input("> ")
            except EOFError:
                return
            except KeyboardInterrupt:
                return

            if motion_language_line.strip():
                self.input_queue.put(motion_language_line)
                print("動作言語を受信しました。サーボの準備完了を待っています。", flush=True)

    def _start_input_thread(self):
        if self.input_thread_started:
            return

        self.input_thread_started = True
        threading.Thread(target=self._read_motion_language, daemon=True).start()

    def _load_next_motion(self):
        try:
            self.motion_language_line = self.input_queue.get_nowait()
            self.servo_positions = self._parse_motion_language()
            self.has_written = False
        except queue.Empty:
            return
        except (TypeError, ValueError) as error:
            print(f"動作言語を受け付けられません: {error}")

    def execute_periodic_task(self, lock, data_dict):
        super().execute_periodic_task(lock, data_dict)
        self._start_input_thread()
        self._load_next_motion()

        if self.has_written or not data_dict["servo_ready"]:
            return
        if data_dict["servo_params_updated"]:
            return

        servo_operation_times = [1000.0] * self.robot_properties.num_servos
        self.write_servo_positions(
            lock,
            data_dict,
            self.servo_positions,
            servo_operation_times,
        )
        self.has_written = True
        print(f"動作言語を実行: {self.motion_language_line}", flush=True)