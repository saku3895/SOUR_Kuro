from src.core.periodic_module import PeriodicModule
from src.core.robot_properties import ServoEasingFunctions

import time
import copy

class PMServoControl(PeriodicModule):
    def __init__(self, robot_properties, interval_ms=1000):
        super().__init__(interval_ms)

        #servo information
        self.servo_ids = copy.deepcopy(robot_properties.servo_ids)
        self.servo_max_positions = copy.deepcopy(robot_properties.servo_max_positions)
        self.servo_min_positions = copy.deepcopy(robot_properties.servo_min_positions)
        self.servo_shifts = copy.deepcopy(robot_properties.servo_shifts)

        self.servo_easing_function = robot_properties.servo_easing_function

        #servo control parameters
        self.all_servos_operated = True
        self.servo_operation_start_time = 0

        self.servo_target_positions = [0.0] * len(self.servo_ids)
        self.servo_operation_times = [0.0] * len(self.servo_ids)

        #set servo prev position to middle
        self.servo_prev_positions = copy.deepcopy(robot_properties.servo_initial_positions)
        self.servo_next_positions = copy.deepcopy(robot_properties.servo_initial_positions)
            
        

    #override this method to send servo commands to each device
    def send_servo_commands(self, next_positions):
        print("command:" + str(next_positions))


    def execute_periodic_task(self, lock, data_dict):
        super().execute_periodic_task(lock, data_dict)

        #check if the servo parameters have been updated
        servo_params_updated = data_dict['servo_params_updated']
        if servo_params_updated == True:
            
            self.all_servos_operated = False
            self.servo_operation_start_time = time.perf_counter()

            self.servo_prev_positions = copy.deepcopy(self.servo_next_positions)

            self.servo_target_positions = copy.deepcopy(data_dict['servo_target_positions'])
            self.servo_operation_times = copy.deepcopy(data_dict['servo_operation_times'])

            with lock:
                data_dict['servo_params_updated'] = False
                data_dict['servo_ready'] = False

        #calculate the servo positions
        if self.all_servos_operated == False:
            current_time = time.perf_counter()
            op_elapsed_time = (current_time - self.servo_operation_start_time) * 1000 #milliseconds
            for i in range(len(self.servo_ids)):

                #print(self.servo_target_positions)
                servo_target_position = self.servo_target_positions[i]
                servo_operation_time = self.servo_operation_times[i]
                servo_prev_position = self.servo_prev_positions[i]
                servo_max_position = self.servo_max_positions[i]
                servo_min_position = self.servo_min_positions[i]
                servo_shift = self.servo_shifts[i]

                servo_target_position += servo_shift

                #calculate the operation ratio. Do not allow values to be greater than 1.0 or less than 0.0.
                op_ratio = min(max(0.0, op_elapsed_time / servo_operation_time), 1.0)

                #servo easing function
                if self.servo_easing_function == ServoEasingFunctions.LINEAR:
                    op_ratio = ServoEasingFunctions.linear(op_ratio)
                elif self.servo_easing_function == ServoEasingFunctions.EASE_IN_OUT_CUBIC:
                    op_ratio = ServoEasingFunctions.ease_in_out_cubic(op_ratio)

                #calculate the servo position
                servo_position = servo_prev_position + (servo_target_position - servo_prev_position) * op_ratio
                
                servo_position = min(max(servo_position, servo_min_position), servo_max_position)

                #set the servo position
                self.servo_next_positions[i] = servo_position

            #send servo commands
            self.send_servo_commands(self.servo_next_positions)

            #check if all servos have been operated
            if op_elapsed_time >= max(self.servo_operation_times):
                self.all_servos_operated = True

                with lock:
                    data_dict['servo_ready'] = True


            
        

