from src.core.periodic_module import PeriodicModule
from src.test.motion import Motion

import random
import copy

class KuroDemoMotionData():

    # Shiroの demo_motion1 の動きをKuroの数値に変換
    demo_motion1_position = [
        [1350.0, 1430.0, 1650.0, 1400.0, 1550.0, 1570.0, 890.0, 1520.0, 1580.0, 1500.0, 1450.0, 1500.0, 1450.0, 1600.0]
    ]
    demo_motion1_time = [
        [3000.0, 3000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
    ]
    demo_motion1_interval = [1000.0]

    # Shiroの demo_motion2 の動き（ALL 500.0）をKuroの数値に変換
    demo_motion2_position = [
         [1335.0, 1375.0, 1810.0, 1375.0, 1290.0, 1630.0, 790.0, 1535.0, 1580.0, 1500.0, 1480.0, 1490.0, 1450.0, 1335.0]
            ]
    demo_motion2_time = [
        [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
    ]
    demo_motion2_interval = [10.0]

    # Shiroの demo_motion3 の動きをKuroの数値に変換
    demo_motion3_position = [
        [1450.0, 1977.3, 1100.0, 943.8, 2100.0, 2100.0, 733.3, 1800.0, 1877.6, 1812.5, 1061.1, 1246.7, 925.0, 1880.0],
        [1450.0, 1977.3, 1500.0, 943.8, 1750.0, 2100.0, 666.7, 1650.0, 1969.7, 2487.5, 1683.3, 1420.0, 1625.0, 1320.0],
        [1450.0, 1977.3, 1100.0, 943.8, 2100.0, 2100.0, 733.3, 1800.0, 1877.6, 1812.5, 1061.1, 1246.7, 925.0, 1880.0]
    ]
    demo_motion3_time = [
        [3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0],
        [3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0],
        [3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0]
    ]
    demo_motion3_interval = [10.0, 3000.0, 10.0]


    def __init__(self):
        pass

    # sitting and look around (Shiroの設定を完全移植)
    def get_demo_motion1(self):
        motion1 = Motion(self.demo_motion1_position, self.demo_motion1_time, self.demo_motion1_interval, 
                         loop_time=1, fluc_type=Motion.FLUCTUATION_TYPE_UNIFORM, 
                         fluc_amp_motion=50, fluc_amp_time=100, fluc_amp_interval=1000)
        
        # 首のランダム値 (Shiro: 300~650, 400~550) をKuroのスケールに変換
        motion1.set_randomizer(0, 0, 1250.0, 1600.0)
        motion1.set_randomizer(0, 1, 1731.8, 2100.0)
        return motion1

    def get_demo_motion2(self):
        return Motion(self.demo_motion2_position, self.demo_motion2_time, self.demo_motion2_interval, 
                      loop_time=10, fluc_type=Motion.FLUCTUATION_TYPE_UNIFORM, 
                      fluc_amp_motion=50, fluc_amp_time=10, fluc_amp_interval=1000)

    def get_demo_motion3(self):
        return Motion(self.demo_motion3_position, self.demo_motion3_time, self.demo_motion3_interval, 
                      loop_time=1, fluc_type=Motion.FLUCTUATION_TYPE_UNIFORM, 
                      fluc_amp_motion=50, fluc_amp_time=10, fluc_amp_interval=1000)


class PMDemoKuro(PeriodicModule):
    def __init__(self, robot_properties, interval_ms=1000):
        super().__init__(interval_ms)
        self.robot_properties = robot_properties
        self.demo_motion1 = KuroDemoMotionData().get_demo_motion1()
        self.demo_motion2 = KuroDemoMotionData().get_demo_motion2()
        self.demo_motion3 = KuroDemoMotionData().get_demo_motion3()
        self.next_motion_interval = 0

        # 初期モーションもShiroに合わせてdemo_motion2スタートに変更
        self.current_motion = self.demo_motion1

    def execute_periodic_task(self, lock, data_dict):
            super().execute_periodic_task(lock, data_dict)
    
            #If servo_params are checked and all servos are ready, it means that previous motion is finished.
            #If servo_ready and servo_params_updated are both True, it means that servo control has not checked servo parameters yet.
            if data_dict['servo_ready'] == True and data_dict['servo_params_updated'] == False:
                
                if self.next_motion_interval > 0:#interval time
                    self.next_motion_interval -= self.interval_ms
                else:
                    if not self.current_motion.is_motion_finished():
                        position, time, interval = self.current_motion.get_next_motion()
    
                        self.write_servo_positions(lock, data_dict, position, time)
    
                        self.next_motion_interval = interval
                    else:
    
                    # state transition among demo motions
                        if self.current_motion == self.demo_motion1:
                            self.current_motion = self.demo_motion2
                            print("demo_motion2")
                        elif self.current_motion == self.demo_motion3:
                            self.current_motion = self.demo_motion2
                            print("demo_motion2")
                        elif self.current_motion == self.demo_motion2:
                            # transition to demo_motion1 in probability of 1/3, demo_motion3 in probability of 1/3, demo_motion2 in probability of 1/3
                            if random.randint(0, 2) == 0:
                                self.current_motion = self.demo_motion1
                                print("demo_motion1")
                            elif random.randint(0, 2) == 1:
                                self.current_motion = self.demo_motion3
                                print("demo_motion3")
                            else:
                                self.current_motion = self.demo_motion2
                                print("demo_motion2")
    
                        self.current_motion.reset()