from src.core.robot_core import RobotCore
from src.test.pm_test1 import PeriodicModuleTest1
from src.test.pm_test2 import PeriodicModuleTest2
from src.test.pm_test3 import PeriodicModuleTest3


def main():
    print("SOUR: Simple Operation for Ubiquitous Robotics")
    r_core = RobotCore()
    r_core.register_module(PeriodicModuleTest1(1000))
    r_core.register_module(PeriodicModuleTest2(5000))
    r_core.register_module(PeriodicModuleTest3(15000))
    r_core.run()


if __name__ == "__main__":

    main()