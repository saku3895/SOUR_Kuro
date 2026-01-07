from src.modules.pm_servo_control import PMServoControl
from src.communication.lobot_serial_communication import LobotServoController

#Servo Controller Class for Hiwonder Serial Bus Servo Controller
#https://www.hiwonder.com/products/serial-bus-servo-controller
class PMServoControlHiwonderSerialBusServoController(PMServoControl):
    def __init__(self, robot_properties, interval_ms=1000):
        super().__init__(robot_properties, interval_ms)
        self.lsc = LobotServoController(port=robot_properties.servo_controller_port)
        if not self.lsc.connect():
            raise ConnectionError("Failed to connect to servo controller")

    def send_servo_commands(self, next_positions):

        """Send servo commands to Hiwonder Serial Bus Servo Controller.

        Args:
            next_positions (list): List of servo positions to send to the controller.
        """
        command = []
        for i in range(len(self.servo_ids)):
            command.append((self.servo_ids[i], int(next_positions[i])))
        # print(command)
        self.lsc.moveServos(command, 10)
