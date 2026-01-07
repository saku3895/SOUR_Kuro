from src.core.periodic_module import PeriodicModule

import random

class PMTestShiro(PeriodicModule):
    def __init__(self, robot_properties, interval_ms=1000):
        super().__init__(interval_ms)
        self.robot_properties = robot_properties

    def execute_periodic_task(self, lock, data_dict):
        
        #if all servos are not ready, return
        if data_dict['servo_ready'] == False:
            return

        servo_positions = self.robot_properties.servo_initial_positions
        servo_operation_times = [300.0] * self.robot_properties.num_servos

        random_pos = random.uniform(300.0, 700.0)
        servo_positions[0] = random_pos

        random_op_time = random.uniform(100.0, 2000.0)
        servo_operation_times[0] = random_op_time

        self.write_servo_positions(lock, data_dict, servo_positions, servo_operation_times)

