import time

# the base class for periodic modules. It will be executed periodically with the specified interval
class PeriodicModule:
    def __init__(self, interval_ms=1000):
        self.interval_ms = interval_ms
        self.terminate = False


    def run(self, lock, data_dict):
        #check the interval and execute the periodic task
        time0 = time.perf_counter()
        while not self.terminate:

            time1 = time.perf_counter()
            if (time1 - time0)*1000 > self.interval_ms:
                time0 = time1

                self.execute_periodic_task(lock, data_dict)
        

    def execute_periodic_task(self, lock, data_dict):
        """
        The function to be executed periodically.

        Parameters
        ----------
        lock : multiprocessing.Lock
            The lock for accessing the shared memory.
        data_dict : multiprocessing.Manager().dict()
            The dictionary for shared memory.

        Returns
        -------
        None
        """
        #check if the program should be terminated
        if data_dict['terminate'] == True:
            self.terminate = True

    #write the servo positions to the shared memory
    def write_servo_positions(self, lock, data_dict, servo_positions, servo_operation_times):

        with lock:
            data_dict['servo_target_positions'] = servo_positions
            data_dict['servo_operation_times'] = servo_operation_times
            data_dict['servo_params_updated'] = True

    def terminate_all(self, lock, data_dict):
        with lock:
            data_dict['terminate'] = True