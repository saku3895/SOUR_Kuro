from src.core.periodic_module import PeriodicModule
from src.core.robot_properties import ServoEasingFunctions

class PMServoControl(PeriodicModule):
    def __init__(self, servo_ids, servo_max_positions, servo_min_positions, interval_ms=1000):
        super().__init__(interval_ms)

        #servo information
        self.servo_ids = servo_ids
        self.servo_max_positions = servo_max_positions
        self.servo_min_positions = servo_min_positions

        #servo control parameters
        self.all_servos_operated = True
        self.servo_operation_start_time = 0

        self.servo_next_positions = [0.0] * len(self.servo_ids)

    def send_servo_commands(self, next_positions):
        pass


    def execute_periodic_task(self, lock, data_dict):

        #check if the servo parameters have been updated
        servo_params_updated = data_dict['servo_params_updated']
        if servo_params_updated == True:
            
            self.all_servos_operated = False
            self.servo_operation_start_time = time.perf_counter()
            servo_target_positions = data_dict['servo_target_positions']
            servo_operation_times = data_dict['servo_operation_times']

            with lock:
                data_dict['servo_params_updated'] = False

        #calculate the servo positions
        if self.all_servo_operated == False:
            current_time = time.perf_counter()
            op_elapsed_time = (current_time - self.servo_operation_start_time) * 1000 #milliseconds
            for i in range(len(self.servo_ids)):
                servo_id = self.servo_ids[i]
                servo_max_position = self.servo_max_positions[i]
                servo_min_position = self.servo_min_positions[i]
                servo_target_position = servo_target_positions[i]
                servo_operation_time = servo_operation_times[i]

                #calculate the operation ratio. Do not allow values to be greater than 1.0 or less than 0.0.
                op_ratio = min(max(0.0, op_elapsed_time / servo_operation_time), 1.0)

                #servo easing function
                if self.servo_easing_function == ServoEasingFunctions.LINEAR:
                    op_ratio = ServoEasingFunctions.linear(op_ratio)
                elif self.servo_easing_function == ServoEasingFunctions.EASE_IN_OUT_CUBIC:
                    op_ratio = ServoEasingFunctions.ease_in_out_cubic(op_ratio)

                #calculate the servo position
                servo_position = servo_min_position + (servo_max_position - servo_min_position) * op_ratio

                #set the servo position
                self.servo_next_positions[i] = servo_position

            #send servo commands
            self.send_servo_commands(self.servo_next_positions)

            #check if all servos have been operated
            if op_elapsed_time >= max(servo_operation_times):
                self.all_servos_operated = True


            
        

