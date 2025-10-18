from src.core.periodic_module import PeriodicModule
from src.communication.unitv2_serial_communication import UnitV2SerialCommunication

import time
import copy

class PMUnitV2(PeriodicModule):
    def __init__(self, robot_properties, interval_ms=1000):
        super().__init__(interval_ms)

        self.unitv2 = UnitV2SerialCommunication(port=robot_properties.camera_port)

        if not self.unitv2.connect():
            raise ConnectionError("Failed to connect to unitv2")
        
        self.json_message = ""

    def read_data(self):
        self.json_message = self.unitv2.receive_data()
        print(self.json_message)

    def execute_periodic_task(self, lock, data_dict):
        super().execute_periodic_task(lock, data_dict)

        self.read_data()





    