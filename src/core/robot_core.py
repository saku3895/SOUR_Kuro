import multiprocessing

from src.core.periodic_module import PeriodicModule
from src.core.async_module import AsyncModule

class RobotCore:
 
    def __init__(self):
        self.modules = []
        self.async_modules = []
        self.periodic_handler_events = []

    def register_module(self, module):
        #accept periodic module only
        assert isinstance(module, PeriodicModule)
        self.modules.append(module)

    def register_async_module(self, module):
        #accept async module only
        assert isinstance(module, AsyncModule)
        self.async_modules.append(module)

    def run(self):
        
        #prepare shared memory here

        #run all periodic modules
        for module in self.modules:
            process = multiprocessing.Process(target=module.run) #add shared memory here
            process.start()






