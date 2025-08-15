from src.core.periodic_module import PeriodicModule

import random
import copy

class Motion():

    FLUCTUATION_TYPE_NONE = 0
    FLUCTUATION_TYPE_UNIFORM = 1

    def __init__(self, position, time, interval, loop_time=1, fluc_type=FLUCTUATION_TYPE_NONE, fluc_amp_motion=0.0, fluc_amp_time=0.0, fluc_amp_interval=0.0):
        assert len(position) == len(time) and len(time) == len(interval)

        self.position = position
        self.time = time
        self.interval = interval
        self.current_index = 0
        self.loop_count = 0
        self.loop_time = loop_time

        self.fluc_type = fluc_type
        self.fluc_amp_motion = fluc_amp_motion
        self.fluc_amp_time = fluc_amp_time
        self.fluc_amp_interval = fluc_amp_interval

        #for randomizer, prepare 2 dimensional array
        self.randomizer = [[None for j in range(len(self.position[0]))] for i in range(len(self.position))]

        self.motion_finished = False


    def get_next_motion(self):
        
        if self.motion_finished:
            return None, None, None

        next_index = self.current_index

        self.current_index = (self.current_index + 1) % len(self.position)
        
        if next_index == len(self.position)-1:
            self.loop_count += 1
            if self.loop_time > 0 and self.loop_count >= self.loop_time:#loop_time == 0 is infinite loop
                self.motion_finished = True

        next_position = copy.deepcopy(self.position[next_index])
        next_time = copy.deepcopy(self.time[next_index])
        next_interval = copy.deepcopy(self.interval[next_index])

        #set random value
        for i in range(len(next_position)):

            if self.randomizer[next_index][i] is not None:
                min_pos, max_pos = self.randomizer[next_index][i]
                next_position[i] = random.uniform(min_pos, max_pos)

        #set fluctuation
        for i in range(len(next_position)):

            if self.fluc_type == Motion.FLUCTUATION_TYPE_UNIFORM:
                next_position[i] += random.uniform(-self.fluc_amp_motion, self.fluc_amp_motion)
                next_time[i] += random.uniform(-self.fluc_amp_time, self.fluc_amp_time)
                next_interval += random.uniform(-self.fluc_amp_interval, self.fluc_amp_interval)

        return next_position, next_time, next_interval

    def get_motion_length(self):
        return len(self.position)

    def is_motion_finished(self):
        return self.motion_finished

    def set_randomizer(self, index_i, index_j, min_pos, max_pos):
        self.randomizer[index_i][index_j] = (min_pos, max_pos)
        

    

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


    def __init__(self):
        pass

    #sitting and look around
    def get_demo_motion1(self):
        motion1 = Motion(self.demo_motion1_position, self.demo_motion1_time, self.demo_motion1_interval, loop_time=15, fluc_type=Motion.FLUCTUATION_TYPE_UNIFORM, fluc_amp_motion=50, fluc_amp_time=100, fluc_amp_interval=1000)
        motion1.set_randomizer(0, 0, 300, 650)
        motion1.set_randomizer(0, 1, 400, 550)
        return motion1

    def get_demo_motion2(self):
        return Motion(self.demo_motion2_position, self.demo_motion2_time, self.demo_motion2_interval, loop_time=10, fluc_type=Motion.FLUCTUATION_TYPE_UNIFORM, fluc_amp_motion=50, fluc_amp_time=10, fluc_amp_interval=1000)

class PMDemoShiro(PeriodicModule):
    def __init__(self, robot_properties, interval_ms=1000):
        super().__init__(interval_ms)
        self.robot_properties = robot_properties
        self.demo_motion1 = ShiroDemoMotionData().get_demo_motion1()
        self.demo_motion2 = ShiroDemoMotionData().get_demo_motion2()
        self.next_motion_interval = 0

    def execute_periodic_task(self, lock, data_dict):
        super().execute_periodic_task(lock, data_dict)

        #If servo_params are checked and all servos are ready, it means that previous motion is finished.
        #If servo_ready and servo_params_updated are both True, it means that servo control has not checked servo parameters yet.
        if data_dict['servo_ready'] == True and data_dict['servo_params_updated'] == False:
            
            if self.next_motion_interval > 0:#interval time
                self.next_motion_interval -= self.interval_ms
            else:

                if not self.demo_motion1.is_motion_finished():

                    position, time, interval = self.demo_motion1.get_next_motion()

                    self.write_servo_positions(lock, data_dict, position, time)

                    self.next_motion_interval = interval    

                else:
                    self.terminate_all(lock, data_dict)

                # if not self.demo_motion2.is_motion_finished():

                #     position, time, interval = self.demo_motion2.get_next_motion()

                #     self.write_servo_positions(lock, data_dict, position, time)

                #     self.next_motion_interval = interval

                # else:
                #     self.terminate_all(lock, data_dict)

        
        
