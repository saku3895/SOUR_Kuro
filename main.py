from src.core.robot_core import RobotCore
from src.core.robot_properties import RobotProperties, ServoEasingFunctions

from src.modules.pm_servo_control import PMServoControl
from src.modules.pm_servo_control_hiwonder_serial_bus_servo_controller import PMServoControlHiwonderSerialBusServoController
from src.modules.pm_unitv2 import PMUnitV2

from src.test.pm_test_shiro import PMTestShiro
from src.test.motion_lang_test_shiro import MotionLangTestShiro
from src.test.pm_demo_shiro import PMDemoShiro
from src.test.pm_demo_betty import PMDemoBetty


def main():

    debug_mode = False

    print("SOUR: Simple Operation for Ubiquitous Robotics")

    robot_properties = RobotProperties()

    ## properties for shiro
    """
    robot_properties.num_servos = 14
    robot_properties.servo_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    robot_properties.servo_min_positions = [100.0, 0.0, 500.0, 450.0, 100.0, 130.0, 150.0, 100.0, 90.0, 200.0, 400.0, 100.0, 450.0, 150.0]
    robot_properties.servo_max_positions = [850.0, 550.0, 850.0, 850.0, 500.0, 500.0, 900.0, 900.0, 850.0, 600.0, 850.0, 850.0, 850.0, 650.0]
    #robot_properties.servo_initial_positions = [500.0, 350.0, 450.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 370.0, 600.0, 500.0, 700.0, 300.0]
    robot_properties.servo_initial_positions = [500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0]
    robot_properties.servo_shifts = [0.0, -150.0, -50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 130.0, -300.0, 0.0, 200.0, -200.0]
    """
    ## properties for betty
    robot_properties.num_servos = 4
    robot_properties.servo_ids = [0, 1, 2, 3]
    robot_properties.servo_min_positions = [500.0, 500.0, 500.0, 500.0]
    robot_properties.servo_max_positions = [2500.0, 2500.0, 2500.0, 2500.0]
    robot_properties.servo_initial_positions = [1500.0, 1500.0, 1500.0, 1500.0]
    robot_properties.servo_shifts = [0.0, 0.0, 0.0, 0.0]
    robot_properties.servo_easing_function = ServoEasingFunctions.EASE_IN_OUT_CUBIC
    robot_properties.servo_controller_port = '/dev/ttyAMA2'
    robot_properties.camera_port = '/dev/ttyAMA4'


    r_core = RobotCore(robot_properties)

    #register periodic modules
    #r_core.register_module(PMTestShiro(robot_properties, 3000))
    #r_core.register_module(MotionLangTestShiro(robot_properties, 3000)) gMLPで動作させるとき
    #r_core.register_module(PMDemoShiro(robot_properties, 100)) #demo for shiro
    r_core.register_module(PMDemoBetty(robot_properties, 80))

    r_core.register_module(PMUnitV2(robot_properties, 500))

    if debug_mode:
        r_core.register_module(PMServoControl(robot_properties, 100))
    else:
        r_core.register_module(PMServoControlHiwonderSerialBusServoController(robot_properties, 100))

    r_core.run()


if __name__ == "__main__":

    main()