import multiprocessing

from src.core.periodic_module import PeriodicModule
from src.core.async_module import AsyncModule

class RobotCore:

    def __init__(self, robot_properties):
        """
        Initialize RobotCore object.

        Parameters
        ----------
        robot_properties : RobotProperties
            RobotProperties object which contains the properties of the robot.

        Attributes
        ----------
        modules : list
            List of periodic modules registered to the robot core.
        async_modules : list
            List of async modules registered to the robot core.
        manager : multiprocessing.Manager
            Manager object for shared memory.
        lock : multiprocessing.Lock
            Exclusive lock for accessing the shared memory.
        data_dict : multiprocessing.Manager().dict()
            Dictionary for shared memory.
        """
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
        self.data_dict['servo_target_positions'] = [0.0] * num_servos
        self.data_dict['servo_operation_times'] = [0.0] * num_servos
        self.data_dict['servo_params_updated'] = False
        self.data_dict['servo_ready'] = True

        #program termination flag
        self.data_dict['terminate'] = False

    def register_module(self, module):
        """
        Register a periodic module to the robot core.

        Parameters
        ----------
        module : PeriodicModule
            The periodic module to be registered.

        Raises
        ------
        AssertionError
            If the module is not an instance of PeriodicModule.
        """
        #accept periodic module only
        assert isinstance(module, PeriodicModule)
        self.modules.append(module)

    def register_async_module(self, module):
        """
        Register an async module to the robot core.

        Parameters
        ----------
        module : AsyncModule
            The async module to be registered.

        Raises
        ------
        AssertionError
            If the module is not an instance of AsyncModule.
        """
        #accept async module only
        assert isinstance(module, AsyncModule)
        self.async_modules.append(module)

    def run(self):
    
        """
        Run all periodic modules.

        This function runs all periodic modules registered to the robot core in parallel.

        It creates a separate process for each periodic module and waits until all processes are finished.

        """

        processes = []
        #run all periodic modules
        for module in self.modules:
            process = multiprocessing.Process(target=module.run, args=(self.lock, self.data_dict,)) #add shared memory here
            processes.append(process)
            process.start()

        #wait until all periodic modules are finished
        for process in processes:
            process.join()


        




