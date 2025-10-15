from src.core.periodic_module import PeriodicModule
from src.test.motion import Motion

import random
import copy

class BettyDemoMotionData():

    ## motion for 4 DoF
    demo_motion_default_position = [
        [1500.0, 1500.0, 1500.0, 1500.0]
    ]

    demo_motion_default_time = [
        [1000.0, 1000.0, 1000.0, 1000.0]
    ]

    demo_motion_default_interval = [
        [1000.0]
    ]


    demo_motion_raise_hands_position = [
        [1500.0, 1500.0, 1500.0, 1500.0]
    ]

    demo_motion_raise_hands_time = [
        [1000.0, 1000.0, 1000.0, 1000.0]
    ]

    demo_motion_raise_hands_interval = [
        [1000.0]
    ]

    def __init__(self):
        pass

    def get_demo_motion_default(self):
        return Motion(self.demo_motion_default_position, self.demo_motion_default_time, self.demo_motion_default_interval, loop_time=0, fluc_type=Motion.FLUCTUATION_TYPE_UNIFORM, fluc_amp_motion=50, fluc_amp_time=100, fluc_amp_interval=1000)

    def get_demo_motion_raise_hands(self):
        return Motion(self.demo_motion_raise_hands_position, self.demo_motion_raise_hands_time, self.demo_motion_raise_hands_interval, loop_time=1, fluc_type=Motion.FLUCTUATION_TYPE_UNIFORM, fluc_amp_motion=50, fluc_amp_time=100, fluc_amp_interval=1000)

class PMDemoBetty(PeriodicModule):
    def __init__(self, robot_properties, interval_ms=1000):
        super().__init__(interval_ms)
        self.robot_properties = robot_properties
        self.demo_motion_default = BettyDemoMotionData().get_demo_motion_default()
        self.demo_motion_raise_hands = BettyDemoMotionData().get_demo_motion_raise_hands()
        self.next_motion_interval = 0
        self.current_motion = self.demo_motion_default

    def execute_periodic_task(self, lock, data_dict):

        super().execute_periodic_task(lock, data_dict)

        if data_dict['servo_ready'] == True and data_dict['servo_params_updated'] == False:

            if self.next_motion_interval > 0:#interval time
                self.next_motion_interval -= self.interval_ms
            else:
                if not self.current_motion.is_motion_finished():
                    position, time, interval = self.current_motion.get_next_motion()

                    self.write_servo_positions(lock, data_dict, position, time)

                    self.next_motion_interval = interval

                else:
                    # state transition
                    pass

