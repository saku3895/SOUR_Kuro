import time

# the base class for periodic modules. It will be executed periodically with the specified interval
class PeriodicModule:
    def __init__(self, interval_ms=1000):
        self.interval_ms = interval_ms


    def run(self, lock, data_dict):
        #check the interval and execute the periodic task
        time0 = time.perf_counter()
        while True:

            time1 = time.perf_counter()
            if (time1 - time0)*1000 > self.interval_ms:
                time0 = time1

                self.execute_periodic_task(lock, data_dict)
        

    def execute_periodic_task(self, data_dict):
        pass

    def write_servo_positions(self, lock, data_dict, servo_positions, servo_operation_times):
        with lock:
            data_dict['servo_target_positions'] = servo_positions
            data_dict['servo_operation_times'] = servo_operation_times
            data_dict['servo_params_updated'] = True