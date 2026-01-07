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

    demo_motion_default_interval = [1000.0]
#########

    demo_motion_raise_hands_position = [
        [1500.0, 1500.0, 900.0, 2100.0]
    ]

    demo_motion_raise_hands_time = [
        [1000.0, 1000.0, 2000.0, 2000.0]
    ]

    demo_motion_raise_hands_interval = [1000.0]
##########
    demo_motion_wave_hand_position = [
        [1500.0, 1500.0, 1500.0, 2100.0],
        [1500.0, 1500.0, 1500.0, 1800.0],
        [1500.0, 1500.0, 1500.0, 2100.0],
        [1500.0, 1500.0, 1500.0, 1800.0],
        [1500.0, 1500.0, 1500.0, 2100.0],
        [1500.0, 1500.0, 1500.0, 1800.0]
    ]

    demo_motion_wave_hand_time = [
        [300.0, 300.0, 300.0, 300.0],
        [300.0, 300.0, 300.0, 300.0],
        [300.0, 300.0, 300.0, 300.0],
        [300.0, 300.0, 300.0, 300.0],
        [300.0, 300.0, 300.0, 300.0],
        [300.0, 300.0, 300.0, 300.0]
    ]
    demo_motion_wave_hand_interval = [10.0,10.0,10.0,10.0,10.0,1000.0]
#############

    demo_motion_nod_position = [
        [1500.0, 1500.0, 1500.0, 1500.0],
        [1500.0, 1600.0, 1500.0, 1500.0],
        [1500.0, 1500.0, 1500.0, 1500.0],
        [1500.0, 1600.0, 1500.0, 1500.0]
    ]

    demo_motion_nod_time = [
        [200.0, 200.0, 200.0, 200.0],
        [200.0, 200.0, 200.0, 200.0],
        [200.0, 200.0, 200.0, 200.0],
        [200.0, 200.0, 200.0, 200.0]
    ]

    demo_motion_nod_interval = [10.0,10.0,10.0,1000.0]

#############

    demo_motion_shake_head_position = [
        [1300.0, 1500.0, 1500.0, 1500.0],
        [1700.0, 1500.0, 1500.0, 1500.0],
        [1300.0, 1500.0, 1500.0, 1500.0],
        [1700.0, 1500.0, 1500.0, 1500.0]
    ]

    demo_motion_shake_head_time = [
        [200.0, 200.0, 200.0, 200.0],
        [200.0, 200.0, 200.0, 200.0],
        [200.0, 200.0, 200.0, 200.0],
        [200.0, 200.0, 200.0, 200.0]
    ]

    demo_motion_shake_head_interval = [10.0,10.0,10.0,1000.0]
    

    def __init__(self):
        pass

    def get_demo_motion_default(self):
        return Motion(self.demo_motion_default_position, self.demo_motion_default_time, self.demo_motion_default_interval, loop_time=3, fluc_type=Motion.FLUCTUATION_TYPE_UNIFORM, fluc_amp_motion=150, fluc_amp_time=1000, fluc_amp_interval=1000)

    def get_demo_motion_raise_hands(self):
        return Motion(self.demo_motion_raise_hands_position, self.demo_motion_raise_hands_time, self.demo_motion_raise_hands_interval, loop_time=1, fluc_type=Motion.FLUCTUATION_TYPE_UNIFORM, fluc_amp_motion=50, fluc_amp_time=100, fluc_amp_interval=1000)

    def get_demo_motion_wave_hand(self):
        return Motion(self.demo_motion_wave_hand_position, self.demo_motion_wave_hand_time, self.demo_motion_wave_hand_interval, loop_time=1, fluc_type=Motion.FLUCTUATION_TYPE_UNIFORM)

    def get_demo_motion_nod(self):
        return Motion(self.demo_motion_nod_position, self.demo_motion_nod_time, self.demo_motion_nod_interval, loop_time=1, fluc_type=Motion.FLUCTUATION_TYPE_UNIFORM)

    def get_demo_motion_shake_head(self):
        return Motion(self.demo_motion_shake_head_position, self.demo_motion_shake_head_time, self.demo_motion_shake_head_interval, loop_time=1, fluc_type=Motion.FLUCTUATION_TYPE_UNIFORM)

class PMDemoBetty(PeriodicModule):
    def __init__(self, robot_properties, interval_ms=1000):
        super().__init__(interval_ms)
        self.robot_properties = robot_properties
        self.demo_motion_default = BettyDemoMotionData().get_demo_motion_default()
        self.demo_motion_raise_hands = BettyDemoMotionData().get_demo_motion_raise_hands()
        self.demo_motion_wave_hand = BettyDemoMotionData().get_demo_motion_wave_hand()
        self.demo_motion_nod = BettyDemoMotionData().get_demo_motion_nod()
        self.demo_motion_shake_head = BettyDemoMotionData().get_demo_motion_shake_head()
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
                    if self.current_motion == self.demo_motion_default:
                        # transition to demo_motion_raise_hands in probability of 10%
                        if random.random() < 0.05:
                            self.current_motion = self.demo_motion_raise_hands
                            print("demo_motion_raise_hands")
                        elif random.random() < 0.1:
                            self.current_motion = self.demo_motion_wave_hand
                            print("demo_motion_wave_hand")
                        elif random.random() < 0.15:
                            self.current_motion = self.demo_motion_nod
                            print("demo_motion_nod")
                        elif random.random() < 0.2:
                            self.current_motion = self.demo_motion_shake_head
                            print("demo_motion_shake_head")

                    elif self.current_motion == self.demo_motion_raise_hands:
                        self.current_motion = self.demo_motion_default
                        print("demo_motion_default")
                    elif self.current_motion == self.demo_motion_wave_hand:
                        self.current_motion = self.demo_motion_default
                        print("demo_motion_default")
                    elif self.current_motion == self.demo_motion_nod:
                        self.current_motion = self.demo_motion_default
                        print("demo_motion_default")
                    elif self.current_motion == self.demo_motion_shake_head:
                        self.current_motion = self.demo_motion_default
                        print("demo_motion_default")

                    self.current_motion.reset()
