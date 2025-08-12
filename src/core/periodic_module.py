import time

class PeriodicModule:
    def __init__(self, interval_ms=1000):
        self.interval_ms = interval_ms


    def run(self, lock, data_dict):
        time0 = time.perf_counter()
        while True:

            time1 = time.perf_counter()
            if (time1 - time0)*1000 > self.interval_ms:
                time0 = time1

                self.execute_periodic_task(lock, data_dict)

        

    def execute_periodic_task(self, data_dict):
        pass