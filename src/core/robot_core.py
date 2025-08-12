import multiprocessing

from src.core.periodic_module import PeriodicModule
from src.core.async_module import AsyncModule

class RobotCore:

    def __init__(self):
        self.modules = []
        self.async_modules = []

        self.manager = multiprocessing.Manager()
        self.counter = self.manager.Value('i', 0)


    def register_module(self, module):
        #accept periodic module only
        assert isinstance(module, PeriodicModule)
        self.modules.append(module)

    def register_async_module(self, module):
        #accept async module only
        assert isinstance(module, AsyncModule)
        self.async_modules.append(module)

    def run(self):
    
        processes = []
        #run all periodic modules
        for module in self.modules:
            process = multiprocessing.Process(target=module.run, args=[self.counter]) #add shared memory here
            processes.append(process)
            process.start()

        #wait until all periodic modules are finished
        for process in processes:
            process.join()


        




