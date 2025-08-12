import multiprocessing

class RobotCore:
 
    def __init__(self):
        self.modules = []
        self.async_modules = []

    def register_module(self, module):
        #accept periodic module only
        assert isinstance(module, PeriodicModule)
        self.modules.append(module)

    def register_async_module(self, module):
        #accept async module only
        assert isinstance(module, AsyncModule)
        self.async_modules.append(module)

    def run(self):
        pass



