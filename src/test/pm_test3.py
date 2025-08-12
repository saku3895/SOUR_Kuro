from src.core.periodic_module import PeriodicModule

class PeriodicModuleTest3(PeriodicModule):
    def __init__(self, interval_ms=1000):
        super().__init__(interval_ms)

    def execute_periodic_task(self, counter):
        counter.value += 10000
        print("PeriodicModuleTest3")