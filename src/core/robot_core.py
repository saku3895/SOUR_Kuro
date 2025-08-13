import multiprocessing

from src.core.periodic_module import PeriodicModule
from src.core.async_module import AsyncModule

class RobotCore:

    def __init__(self, robot_properties):
        self.modules = []
        self.async_modules = []

        #shared memory
        self.manager = multiprocessing.Manager()

        #exclusive lock
        self.lock = multiprocessing.Lock()
    
        #data dictionary for shared memory
        self.data_dict = self.manager.dict()

        num_servos = robot_properties.num_servos

        #servo positions and operation parameters
        self.data_dict['servo_current_positions'] = robot_properties.servo_initial_positions
        self.data_dict['servo_params_updated'] = True




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
            process = multiprocessing.Process(target=module.run, args=(self.lock, self.data_dict,)) #add shared memory here
            processes.append(process)
            process.start()

        #wait until all periodic modules are finished
        for process in processes:
            process.join()


        




