from src.core.periodic_module import PeriodicModule

import random

class Motion():
    def __init__(self, position, time, interval, loop_time=1):
        assert len(position) == len(time) and len(time) == len(interval)

        self.position = position
        self.time = time
        self.interval = interval
        self.current_index = 0
        self.loop_count = 0
        self.loop_time = loop_time
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

        return self.position[next_index], self.time[next_index], self.interval[next_index]

    def get_motion_length(self):
        return len(self.position)

    def is_motion_finished(self):
        return self.motion_finished

    

class ShiroDemoMotionData():

    #motion for 14 DoF
    demo_motion1_position = [
        [500.0, 350.0, 450.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 370.0, 600.0, 500.0, 700.0, 300.0],
        [700.0, 350.0, 450.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 370.0, 600.0, 500.0, 700.0, 300.0],
        [300.0, 350.0, 450.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 370.0, 600.0, 500.0, 700.0, 300.0]
    ]

    demo_motion1_time = [
        [1000.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        [10000.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        [200.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    ]

    demo_motion1_interval = [1000, 2000, 3000]

    def __init__(self):
        pass

    def get_demo_motion1(self):
        return Motion(self.demo_motion1_position, self.demo_motion1_time, self.demo_motion1_interval, loop_time=3)

class PMDemoShiro(PeriodicModule):
    def __init__(self, robot_properties, interval_ms=1000):
        super().__init__(interval_ms)
        self.robot_properties = robot_properties
        self.demo_motion1 = ShiroDemoMotionData().get_demo_motion1()
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

        
        
