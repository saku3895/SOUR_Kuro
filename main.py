from src.core.robot_core import RobotCore
from src.core.robot_properties import RobotProperties, ServoEasingFunctions
from src.test.pm_test1 import PeriodicModuleTest1
from src.test.pm_test2 import PeriodicModuleTest2
from src.test.pm_test3 import PeriodicModuleTest3
from src.test.pm_test_shiro import PMTestShiro
from src.modules.pm_servo_control import PMServoControl
from src.modules.pm_servo_control_hiwonder_servo_bus_controler import PMServoControlHiwonderServoBusControler


def main():
    print("SOUR: Simple Operation for Ubiquitous Robotics")

    robot_properties = RobotProperties()
    robot_properties.num_servos = 14
    robot_properties.servo_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    robot_properties.servo_min_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    robot_properties.servo_max_positions = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
    robot_properties.servo_initial_positions = [500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0]

    robot_properties.servo_easing_function = ServoEasingFunctions.LINEAR


    r_core = RobotCore(robot_properties)
    #r_core.register_module(PeriodicModuleTest1(300))
    r_core.register_module(PMTestShiro(robot_properties, 3000))
    #r_core.register_module(PMServoControl(robot_properties, 100))
    r_core.register_module(PMServoControlHiwonderServoBusControler(robot_properties, 100))
    #r_core.register_module(PeriodicModuleTest2(5000))
    #r_core.register_module(PeriodicModuleTest3(15000))
    r_core.run()


if __name__ == "__main__":

    main()