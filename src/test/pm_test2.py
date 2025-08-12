from src.core.periodic_module import PeriodicModule
class PeriodicModuleTest2(PeriodicModule):
    def __init__(self, interval_ms=1000):
        super().__init__(interval_ms)

    def execute_periodic_task(self):
        print("PeriodicModuleTest2")