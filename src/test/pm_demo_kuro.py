from src.core.periodic_module import PeriodicModule
from src.test.motion import Motion

import random
import copy

class KuroDemoMotionData():

    # motion for 14 DoF
    # order: [ID2, ID3, ID4, ID5, ID6, ID7, ID8, ID9, ID10, ID11, ID12, ID13, ID14, ID15]
    # key:   [a,   d,   e,   q,   i,   m,   n,   f,   r,    h,    l,    j,    p,    t]

    # motion 1
    demo_motion1_position = [
        [1500.0, 1330.0, 1800.0, 1550.0, 1250.0, 1850.0, 1350.0, 1450.0, 1550.0, 1200.0, 1800.0, 1500.0, 1250.0, 1800.0]
    ]
    demo_motion1_time = [
        [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
    ]
    demo_motion1_interval = [1000.0]

    # motion 2
    demo_motion2_position = [
        # 基本姿勢
        [1500.0, 1330.0, 1800.0, 1550.0, 1250.0, 1850.0, 1350.0, 1450.0, 1550.0, 1200.0, 1800.0, 1500.0, 1250.0, 1800.0],
        [1500.0, 1330.0, 1800.0, 1550.0, 1250.0, 2000.0, 1350.0, 1450.0, 1550.0, 1200.0, 1800.0, 1500.0, 1250.0, 1800.0],
        [1800.0, 1330.0, 1800.0, 1550.0, 1250.0, 2000.0, 1350.0, 1450.0, 1550.0, 1200.0, 1800.0, 1300.0, 1250.0, 1800.0],
        [1800.0, 1500.0, 1850.0, 1550.0, 1250.0, 2000.0, 1350.0, 1450.0, 1550.0, 550.0, 1800.0, 1300.0, 1250.0, 1800.0]
    ]
    demo_motion2_time = [
        [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
        [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
        [1500.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0, 1500.0],
        [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
    ]
    demo_motion2_interval = [1000.0, 1000.0, 1000.0, 5000.0]

    # motion 3
    # motion 3: left front leg step test
    demo_motion3_position = [
        # 基本姿勢
        [1500.0, 1330.0, 1800.0, 1550.0, 1250.0, 1850.0, 1350.0, 1450.0, 1550.0, 1200.0, 1800.0, 1500.0, 1250.0, 1800.0],
        # 左前脚だけ少し曲げる
        [1500.0, 1330.0, 1750.0, 1550.0, 1250.0, 1850.0, 1350.0, 1500.0, 1550.0, 1300.0, 1800.0, 1500.0, 1250.0, 1800.0],
        # 基本姿勢に戻す
        [1500.0, 1330.0, 1800.0, 1550.0, 1250.0, 1850.0, 1350.0, 1450.0, 1550.0, 1200.0, 1800.0, 1500.0, 1250.0, 1800.0]
    ]

    demo_motion3_time = [
        [2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0],
        [2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0],
        [2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0]
    ]

    demo_motion3_interval = [500.0, 500.0, 1000.0]

    #############
        
    def __init__(self):
            pass

    def get_demo_motion1(self):
        motion1 = Motion(
            self.demo_motion1_position,
            self.demo_motion1_time,
            self.demo_motion1_interval,
            loop_time=1,
            fluc_type=Motion.FLUCTUATION_TYPE_UNIFORM,
            fluc_amp_motion=0,
            fluc_amp_time=0,
            fluc_amp_interval=0
        )

        # 首ランダム
        # index 0 = ID2 = NECK_PHI1
        # index 1 = ID3 = NECK_PSI2
        motion1.set_randomizer(0, 0, 1000, 2000)
        motion1.set_randomizer(0, 1, 1010, 1550)

        return motion1

    def get_demo_motion2(self):
        return Motion(
            self.demo_motion2_position,
            self.demo_motion2_time,
            self.demo_motion2_interval,
            loop_time=10,
            fluc_type=Motion.FLUCTUATION_TYPE_UNIFORM,
            fluc_amp_motion=0,
            fluc_amp_time=0,
            fluc_amp_interval=0
        )

    def get_demo_motion3(self):
        return Motion(
            self.demo_motion3_position,
            self.demo_motion3_time,
            self.demo_motion3_interval,
            loop_time=1,
            fluc_type=Motion.FLUCTUATION_TYPE_UNIFORM,
            fluc_amp_motion=0,
            fluc_amp_time=0,
            fluc_amp_interval=0
        )


class PMDemoKuro(PeriodicModule):
    def __init__(self, robot_properties, interval_ms=1000):
        super().__init__(interval_ms)
        self.robot_properties = robot_properties
        self.demo_motion1 = KuroDemoMotionData().get_demo_motion1()
        self.demo_motion2 = KuroDemoMotionData().get_demo_motion2()
        self.demo_motion3 = KuroDemoMotionData().get_demo_motion3()
        self.next_motion_interval = 0

        self.current_motion = self.demo_motion1

    def execute_periodic_task(self, lock, data_dict):

        super().execute_periodic_task(lock, data_dict)

        if data_dict['servo_ready'] == True and data_dict['servo_params_updated'] == False:

            if self.next_motion_interval > 0:  # interval time
                self.next_motion_interval -= self.interval_ms
            else:
                if not self.current_motion.is_motion_finished():
                    position, time, interval = self.current_motion.get_next_motion()

                    self.write_servo_positions(lock, data_dict, position, time)

                    self.next_motion_interval = interval
                else:

                    # # state transition among demo motions
                    # if self.current_motion == self.demo_motion1:
                    #     self.current_motion = self.demo_motion2
                    #     print("demo_motion2")
                    # elif self.current_motion == self.demo_motion3:
                    #     self.current_motion = self.demo_motion2
                    #     print("demo_motion2")
                    # elif self.current_motion == self.demo_motion2:
                    #     # transition to demo_motion1 in probability of 1/3,
                    #     # demo_motion3 in probability of 1/3,
                    #     # demo_motion2 in probability of 1/3
                    #     if random.randint(0, 2) == 0:
                    #         self.current_motion = self.demo_motion1
                    #         print("demo_motion1")
                    #     elif random.randint(0, 2) == 1:
                    #         self.current_motion = self.demo_motion3
                    #         print("demo_motion3")
                    #     else:
                    #         self.current_motion = self.demo_motion2
                    #         print("demo_motion2")

                    # self.current_motion.reset()
                    # repeat motion1 only

                    self.current_motion = self.demo_motion2
                    print("demo_motion2")


                    self.current_motion.reset()