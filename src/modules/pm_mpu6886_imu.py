from src.core.periodic_module import PeriodicModule
from src.communication.mpu6886 import MPU6886

class PMMPU6886IMU(PeriodicModule):
    def __init__(self, robot_properties,interval_ms=1000):
        super().__init__(interval_ms)
        self.mpu6886 = MPU6886()
        self.mpu6886.initialize()


    def execute_periodic_task(self, lock, data_dict):
        super().execute_periodic_task(lock, data_dict)

        gx, gy, gz = self.mpu6886.get_gyro_data()
        ax, ay, az = self.mpu6886.get_accel_data()
        temp = self.mpu6886.get_temp_data()

        print(f"gx: {gx}, gy: {gy}, gz: {gz}, ax: {ax}, ay: {ay}, az: {az}, temp: {temp}")

        #data_dict['gx'] = gx
        #data_dict['gy'] = gy
        #data_dict['gz'] = gz

        #data_dict['ax'] = ax
        #data_dict['ay'] = ay
        #data_dict['az'] = az

        #data_dict['temp'] = temp