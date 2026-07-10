from src.core.robot_core import RobotCore
from src.core.robot_properties import RobotProperties, ServoEasingFunctions

from src.modules.pm_servo_control import PMServoControl
from src.modules.pm_servo_control_hiwonder_serial_bus_servo_controller import PMServoControlHiwonderSerialBusServoController
from src.modules.pm_unitv2 import PMUnitV2
from src.modules.pm_mpu6886_imu import PMMPU6886IMU
from src.modules.pm_camera_face_tracking import PMCameraFaceTracking

from src.test.pm_test_shiro import PMTestShiro
from src.test.motion_lang_test_shiro import MotionLangTestShiro
from src.test.motion_lang_test_kuro import MotionLangTestKuro
from src.test.pm_demo_shiro import PMDemoShiro
from src.test.pm_demo_betty import PMDemoBetty
from src.test.pm_demo_kuro import PMDemoKuro


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
    """
    robot_properties.num_servos = 4
    robot_properties.servo_ids = [0, 1, 2, 3]
    robot_properties.servo_min_positions = [500.0, 500.0, 500.0, 500.0]
    robot_properties.servo_max_positions = [2500.0, 2500.0, 2500.0, 2500.0]
    robot_properties.servo_initial_positions = [1500.0, 1500.0, 1500.0, 1500.0]
    robot_properties.servo_shifts = [0.0, 0.0, 0.0, 0.0]
    robot_properties.servo_easing_function = ServoEasingFunctions.EASE_IN_OUT_CUBIC
    robot_properties.servo_controller_port = '/dev/ttyAMA2'
    robot_properties.camera_port = '/dev/ttyAMA4'
    """
    ## properties for kuro

    robot_properties.num_servos = 14
    robot_properties.servo_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    robot_properties.servo_min_positions = [800.0, 1010.0, 1300.0, 800.0, 1100.0, 1350.0, 1250.0, 1100.0, 1000.0, 540.0, 1100.0, 800.0, 600.0, 1100.0]
    robot_properties.servo_max_positions = [2110.0, 1550.0, 2000.0, 1800.0, 1700.0, 2000.0, 2000.0, 2100.0, 1800.0, 1750.0, 2500.0, 1700.0, 1900.0, 2300.0]
    robot_properties.servo_initial_positions = [1500.0, 1330.0, 1800.0, 1550.0, 1250.0, 1850.0, 1350.0, 1450.0, 1550.0, 1200.0, 1800.0, 1500.0, 1250.0, 1800.0]
    robot_properties.servo_shifts = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    robot_properties.servo_easing_function = ServoEasingFunctions.EASE_IN_OUT_CUBIC
    robot_properties.servo_controller_port = '/dev/ttyS0'

    r_core = RobotCore(robot_properties)

    #register periodic modules
    #r_core.register_module(PMTestShiro(robot_properties, 3000))
    #r_core.register_module(MotionLangTestShiro(robot_properties, 3000)) gMLPで動作させるとき
    #r_core.register_module(PMDemoShiro(robot_properties, 100)) #demo for shiro
    # r_core.register_module(PMMPU6886IMU(robot_properties, 10))

    ## for betty
    #r_core.register_module(PMDemoBetty(robot_properties, 80))
    #r_core.register_module(PMUnitV2(robot_properties, 500))

    #for kuro
    # r_core.register_module(PMDemoKuro(robot_properties, 100))
    # r_core.register_module(MotionLangTestKuro(robot_properties, 3000))
    # r_core.register_module(PMCameraMediaPipe(robot_properties, 40, kp=0.08))
    # for kuro
    # r_core.register_module(PMDemoKuro(robot_properties, 100))
    # r_core.register_module(MotionLangTestKuro(robot_properties, 3000))
    # r_core.register_module(PMCameraMediaPipe(robot_properties, 40, kp=0.08))
    
    r_core.register_module(PMCameraFaceTracking(robot_properties, 20))

    if debug_mode:
        r_core.register_module(PMServoControl(robot_properties, 10))
    else:
        r_core.register_module(PMServoControlHiwonderSerialBusServoController(robot_properties, 100))

    r_core.run()


if __name__ == "__main__":

    main()