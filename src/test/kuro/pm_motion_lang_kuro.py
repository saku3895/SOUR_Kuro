from src.control.motion_lang2motor import parse_motion_commands
from src.core.periodic_module import PeriodicModule


class PMMotionLanguageKuro(PeriodicModule):
    """1行の動作言語または複数姿勢列を、姿勢1→姿勢2→...として実行する。"""

    def __init__(
        self,
        robot_properties,
        motion_language_line="*a4d6e4q6i6m1n1f3r3h4l4j2p4t3#",
        interval_ms=1000,
        pose_hold_ms=2000,
        default_pose=None,
    ):
        super().__init__(interval_ms)
        self.robot_properties = robot_properties
        self.pose_hold_ms = max(0, int(pose_hold_ms))
        self.pose_hold_remaining_ms = 0

        self.motion_languages = self._prepare_motion_languages(motion_language_line)
        if default_pose is not None:
            self.motion_languages = [self._normalize_motion_language(default_pose)] + self.motion_languages

        self.pose_index = 0
        self.current_motion_language = self.motion_languages[0]
        self.servo_positions = self._parse_motion_language(self.current_motion_language)
        self.has_written = False

    def _prepare_motion_languages(self, motion_language_line):
        if isinstance(motion_language_line, str):
            return [self._normalize_motion_language(motion_language_line)]
        if isinstance(motion_language_line, (list, tuple)) and len(motion_language_line) > 0:
            return [self._normalize_motion_language(m) for m in motion_language_line]
        raise ValueError("motion_language_line は文字列または動作言語リストを指定してください")

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

    def _parse_motion_language(self, motion_language_line):
        normalized = self._normalize_motion_language(motion_language_line)
        parsed_commands = parse_motion_commands(("1", [normalized]))
        if len(parsed_commands) != 1 or not parsed_commands[0]:
            raise ValueError(f"動作言語から有効なモータ指令を生成できません: {normalized}")

        servo_positions = list(self.robot_properties.servo_initial_positions)
        if len(servo_positions) != self.robot_properties.num_servos:
            raise ValueError("servo_initial_positions の要素数が num_servos と一致しません")

        for servo_id, position in parsed_commands[0]:
            if servo_id in self.robot_properties.servo_ids:
                servo_index = self.robot_properties.servo_ids.index(servo_id)
                servo_positions[servo_index] = float(position)

        return servo_positions

    def _next_pose(self):
        if self.pose_index >= len(self.motion_languages):
            return None

        motion_language = self.motion_languages[self.pose_index]
        self.current_motion_language = motion_language
        self.servo_positions = self._parse_motion_language(motion_language)
        self.pose_index += 1
        return motion_language

    def execute_periodic_task(self, lock, data_dict):
        super().execute_periodic_task(lock, data_dict)

        if not data_dict["servo_ready"]:
            return
        if data_dict["servo_params_updated"]:
            return

        if self.has_written:
            if self.pose_hold_remaining_ms > 0:
                self.pose_hold_remaining_ms = max(0, self.pose_hold_remaining_ms - self.interval_ms)
                return
            self.has_written = False

        next_motion = self._next_pose()
        if next_motion is None:
            print("動作言語の列を完了しました", flush=True)
            self.pose_hold_remaining_ms = 0
            return

        servo_operation_times = [1000.0] * self.robot_properties.num_servos
        self.write_servo_positions(
            lock,
            data_dict,
            self.servo_positions,
            servo_operation_times,
        )
        self.has_written = True
        self.pose_hold_remaining_ms = self.pose_hold_ms
        print(f"動作言語を実行: {next_motion}", flush=True)

    def reset(self):
        self.pose_index = 0
        self.has_written = False
        self.pose_hold_remaining_ms = 0
        self.current_motion_language = self.motion_languages[0]
        self.servo_positions = self._parse_motion_language(self.current_motion_language)