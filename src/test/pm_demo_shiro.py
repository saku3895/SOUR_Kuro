from src.core.periodic_module import PeriodicModule
from src.test.motion import Motion

import random
import copy

class ShiroDemoMotionData():

    #motion for 14 DoF

    #motion 1
    demo_motion1_position = [
        [500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 400.0, 600.0, 600.0, 700.0, 700.0, 400.0, 700.0, 300.0]
    ]

    demo_motion1_time = [
        [3000.0, 3000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
    ]

    demo_motion1_interval = [1000.0]

    #motion 2
    demo_motion2_position = [
        [500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0]
    ]

    demo_motion2_time = [
        [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
    ]

    demo_motion2_interval = [10.0]

    #motion 3
    demo_motion3_position = [
        [500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0],
        [500.0, 500.0, 700.0, 500.0, 300.0, 500.0, 400.0, 300.0, 600.0, 700.0, 700.0, 700.0, 700.0, 300.0],
        [500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0]
    ]

    demo_motion3_time = [
        [3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0],
        [3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0],
        [3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0]
    ]

    demo_motion3_interval = [10.0,3000.0,10.0]



    def __init__(self):
        pass

    #sitting and look around
    def get_demo_motion1(self):
        motion1 = Motion(self.demo_motion1_position, self.demo_motion1_time, self.demo_motion1_interval, loop_time=1, fluc_type=Motion.FLUCTUATION_TYPE_UNIFORM, fluc_amp_motion=50, fluc_amp_time=100, fluc_amp_interval=1000)
        motion1.set_randomizer(0, 0, 300, 650)
        motion1.set_randomizer(0, 1, 400, 550)
        return motion1

    def get_demo_motion2(self):
        return Motion(self.demo_motion2_position, self.demo_motion2_time, self.demo_motion2_interval, loop_time=10, fluc_type=Motion.FLUCTUATION_TYPE_UNIFORM, fluc_amp_motion=50, fluc_amp_time=10, fluc_amp_interval=1000)

    def get_demo_motion3(self):
        return Motion(self.demo_motion3_position, self.demo_motion3_time, self.demo_motion3_interval, loop_time=1, fluc_type=Motion.FLUCTUATION_TYPE_UNIFORM, fluc_amp_motion=50, fluc_amp_time=10, fluc_amp_interval=1000)



class PMDemoShiro(PeriodicModule):
    def __init__(self, robot_properties, interval_ms=1000):
        super().__init__(interval_ms)
        self.robot_properties = robot_properties
        self.demo_motion1 = ShiroDemoMotionData().get_demo_motion1()
        self.demo_motion2 = ShiroDemoMotionData().get_demo_motion2()
        self.demo_motion3 = ShiroDemoMotionData().get_demo_motion3()
        self.next_motion_interval = 0

        self.current_motion = self.demo_motion2

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


                # if not self.demo_motion1.is_motion_finished():

                #     position, time, interval = self.demo_motion1.get_next_motion()

                #     self.write_servo_positions(lock, data_dict, position, time)

                #     self.next_motion_interval = interval    

                # else:
                #     self.terminate_all(lock, data_dict)

                # if not self.demo_motion2.is_motion_finished():

                #     position, time, interval = self.demo_motion2.get_next_motion()

                #     self.write_servo_positions(lock, data_dict, position, time)

                #     self.next_motion_interval = interval

                # else:
                #     self.terminate_all(lock, data_dict)

                    # if not self.demo_motion3.is_motion_finished():

                    #     position, time, interval = self.demo_motion3.get_next_motion()

                    #     self.write_servo_positions(lock, data_dict, position, time)

                    #     self.next_motion_interval = interval

                    # else:
                    #     self.terminate_all(lock, data_dict)


        
        
