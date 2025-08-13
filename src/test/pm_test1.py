from src.core.periodic_module import PeriodicModule

import random

class PeriodicModuleTest1(PeriodicModule):
    def __init__(self, interval_ms=1000):
        super().__init__(interval_ms)

    def execute_periodic_task(self, lock, data_dict):

        if data_dict['servo_ready'] == False:
            return

        #make servo positions list with random values
        servo_positions = [random.uniform(0.0, 1000) for x in range(6)]

        #make servo operation times list with random values
        servo_operation_times = [random.uniform(0.0, 5000) for x in range(6)]
        
        self.write_servo_positions(lock, data_dict, servo_positions, servo_operation_times)

        print("PeriodicModuleTest1")
        print(data_dict)