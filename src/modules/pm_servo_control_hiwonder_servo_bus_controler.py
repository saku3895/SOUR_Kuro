from src.modules.pm_servo_control import PMServoControl
from src.communication.lobot_serial_communication import LobotServoController

class PMServoControlHiwonderServoBusControler(PMServoControl):
    def __init__(self, robot_properties, interval_ms=1000):
        super().__init__(robot_properties, interval_ms)
        self.lsc = LobotServoController(port=robot_properties.servo_controller_port)
        if not self.lsc.connect():
            raise ConnectionError("Failed to connect to servo controller")

    def send_servo_commands(self, next_positions):
        #print("command hiwonder:" + str(next_positions))
        command = []
        for i in range(len(self.servo_ids)):
            command.append((self.servo_ids[i], int(next_positions[i])))
        # print(command)
        self.lsc.moveServos(command, 10)
