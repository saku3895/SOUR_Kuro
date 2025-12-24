from src.core.periodic_module import PeriodicModule
from src.communication.mpu6886 import MPU6886

import imufusion
import numpy

class PMMPU6886IMU(PeriodicModule):
    def __init__(self, robot_properties,interval_ms=1000):
        super().__init__(interval_ms)
        self.mpu6886 = MPU6886()
        self.mpu6886.initialize()
        self.ahrs = imufusion.Ahrs()


    def execute_periodic_task(self, lock, data_dict):
        super().execute_periodic_task(lock, data_dict)

        gyro_data = self.mpu6886.get_gyro_data()
        accel_data = self.mpu6886.get_accel_data()

        #convert gyro_data and accel_data to numpy.array
        na_gyro_data = numpy.array(gyro_data)
        na_accel_data = numpy.array(accel_data)

        temp = self.mpu6886.get_temp_data()

        self.ahrs.update_no_magnetometer(na_gyro_data, na_accel_data, 0.01)
        euler_angles = self.ahrs.quaternion.to_euler()

        print(f"euler angles:  pitch: {euler_angles[0]}, roll: {euler_angles[1]}, yaw: {euler_angles[2]}")

        #print(f"gx: {gx}, gy: {gy}, gz: {gz}, ax: {ax}, ay: {ay}, az: {az}, temp: {temp}")

        #data_dict['gx'] = gx
        #data_dict['gy'] = gy
        #data_dict['gz'] = gz

        #data_dict['ax'] = ax
        #data_dict['ay'] = ay
        #data_dict['az'] = az

        #data_dict['temp'] = temp