from src.core.periodic_module import PeriodicModule

class PeriodicModuleTest3(PeriodicModule):
    def __init__(self, interval_ms=1000):
        super().__init__(interval_ms)

    def execute_periodic_task(self, lock, data_dict):
        with lock:
            data_dict['counter'] += 10000
            data_dict['servo_angles'] = [100.0,200.0,300.0,400.0,500.0,600.0]
        print("PeriodicModuleTest3")