from src.core.periodic_module import PeriodicModule
from src.communication.mpu6886 import MPU6886

import imufusion

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
        temp = self.mpu6886.get_temp_data()

        ahrs.update_no_magnetometer(gyro_data, accel_data, 0.1)
        euler_angles = ahrs.quaternion.to_euler()

        print(f"euler angles: roll: {euler_angles[0]}, pitch: {euler_angles[1]}, yaw: {euler_angles[2]}")

        #print(f"gx: {gx}, gy: {gy}, gz: {gz}, ax: {ax}, ay: {ay}, az: {az}, temp: {temp}")

        #data_dict['gx'] = gx
        #data_dict['gy'] = gy
        #data_dict['gz'] = gz

        #data_dict['ax'] = ax
        #data_dict['ay'] = ay
        #data_dict['az'] = az

        #data_dict['temp'] = temp