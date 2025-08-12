from src.core.periodic_module import PeriodicModule

class PeriodicModuleTest1(PeriodicModule):
    def __init__(self, interval_ms=1000):
        super().__init__(interval_ms)

    def execute_periodic_task(self, lock, data_dict):

        with lock:
            data_dict['counter'] += 1

        print("PeriodicModuleTest1")

        print(data_dict)