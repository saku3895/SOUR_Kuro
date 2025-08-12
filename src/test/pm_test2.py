from src.core.periodic_module import PeriodicModule
class PeriodicModuleTest2(PeriodicModule):
    def __init__(self, interval_ms=1000):
        super().__init__(interval_ms)

    def execute_periodic_task(self, lock, data_dict):
        with lock:
            data_dict['counter'] += 100
            data_dict['servo_angles'] = [10.0,20.0,30.0,40.0,50.0,60.0]
        print("PeriodicModuleTest2")