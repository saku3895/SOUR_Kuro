from enum import Enum
from smbus2 import SMBus
import time
import numpy

class MPU6886Constants(Enum):
    IMU_6886_ADDRESS = 0x68
    IMU_6886_WHOAMI = 0x75
    IMU_6886_ACCEL_INTEL_CTRL = 0x69
    IMU_6886_SMPLRT_DIV = 0x19
    IMU_6886_INT_PIN_CFG = 0x37
    IMU_6886_INT_ENABLE = 0x38
    IMU_6886_ACCEL_XOUT_H = 0x3B
    IMU_6886_ACCEL_XOUT_L = 0x3C
    IMU_6886_ACCEL_YOUT_H = 0x3D
    IMU_6886_ACCEL_YOUT_L = 0x3E
    IMU_6886_ACCEL_ZOUT_H = 0x3F
    IMU_6886_ACCEL_ZOUT_L = 0x40

    IMU_6886_TEMP_OUT_H = 0x41
    IMU_6886_TEMP_OUT_L = 0x42

    IMU_6886_GYRO_XOUT_H = 0x43
    IMU_6886_GYRO_XOUT_L = 0x44
    IMU_6886_GYRO_YOUT_H = 0x45
    IMU_6886_GYRO_YOUT_L = 0x46
    IMU_6886_GYRO_ZOUT_H = 0x47
    IMU_6886_GYRO_ZOUT_L = 0x48

    IMU_6886_USER_CTRL = 0x6A
    IMU_6886_PWR_MGMT_1 = 0x6B
    IMU_6886_PWR_MGMT_2 = 0x6C
    IMU_6886_CONFIG = 0x1A
    IMU_6886_GYRO_CONFIG = 0x1B
    IMU_6886_ACCEL_CONFIG = 0x1C
    IMU_6886_ACCEL_CONFIG2 = 0x1D
    IMU_6886_FIFO_EN = 0x23

    IMU_6886_FIFO_ENABLE = 0x23
    IMU_6886_FIFO_COUNT = 0x72
    IMU_6886_FIFO_R_W = 0x74
    IMU_6886_GYRO_OFFSET = 0x13

    RtA = 57.324841
    AtR = 0.0174533	
    Gyro_Gr = 0.0010653

    ASCALE_AFS_2G = 0
    ASCALE_AFS_4G = 1
    ASCALE_AFS_8G = 2
    ASCALE_AFS_16G = 3

    GSCALE_GFS_250DPS = 0
    GSCALE_GFS_500DPS = 1
    GSCALE_GFS_1000DPS = 2
    GSCALE_GFS_2000DPS = 3


class MPU6886:

    def __init__(self):
        bus_num = 1
        self.bus = SMBus(bus_num)

        self.a_res = 0.0
        self.g_res = 0.0
        self.imu_ID = -1
        self.gy_scale = -1
        self.ac_scale = -1

    def initialize(self):
        self.gyscale = MPU6886Constants.GSCALE_GFS_2000DPS.value
        self.acscale = MPU6886Constants.ASCALE_AFS_8G.value

        self.imu_ID = self.bus.read_i2c_block_data(MPU6886Constants.IMU_6886_ADDRESS.value, MPU6886Constants.IMU_6886_WHOAMI.value, 1)

        time.sleep(0.001)

        #initialize regdata as numpy uint16
        regdata = numpy.uint8(0x00)

        self.bus.write_i2c_block_data(MPU6886Constants.IMU_6886_ADDRESS.value, MPU6886Constants.IMU_6886_PWR_MGMT_1.value, [regdata])
        time.sleep(0.01)

        regdata = 0x01 << 7
        self.bus.write_i2c_block_data(MPU6886Constants.IMU_6886_ADDRESS.value, MPU6886Constants.IMU_6886_PWR_MGMT_1.value, [regdata])
        time.sleep(0.01)

        regdata = 0x01 << 0
        self.bus.write_i2c_block_data(MPU6886Constants.IMU_6886_ADDRESS.value, MPU6886Constants.IMU_6886_PWR_MGMT_1.value, [regdata])
        time.sleep(0.01)

        ##8g
        regdata = 0x10
        self.bus.write_i2c_block_data(MPU6886Constants.IMU_6886_ADDRESS.value, MPU6886Constants.IMU_6886_ACCEL_CONFIG.value, [regdata])
        time.sleep(0.001)

        ##2000dps
        regdata = 0x18
        self.bus.write_i2c_block_data(MPU6886Constants.IMU_6886_ADDRESS.value, MPU6886Constants.IMU_6886_GYRO_CONFIG.value, [regdata])
        time.sleep(0.001)

        ##1khz output
        regdata = 0x01
        self.bus.write_i2c_block_data(MPU6886Constants.IMU_6886_ADDRESS.value, MPU6886Constants.IMU_6886_CONFIG.value, [regdata])
        time.sleep(0.001)

        ##2 div, FIFO 500hz out
        regdata = 0x01
        self.bus.write_i2c_block_data(MPU6886Constants.IMU_6886_ADDRESS.value, MPU6886Constants.IMU_6886_SMPLRT_DIV.value, [regdata])
        time.sleep(0.001)

        regdata = 0x00
        self.bus.write_i2c_block_data(MPU6886Constants.IMU_6886_ADDRESS.value, MPU6886Constants.IMU_6886_INT_ENABLE.value, [regdata])
        time.sleep(0.001)

        regdata = 0x00
        self.bus.write_i2c_block_data(MPU6886Constants.IMU_6886_ADDRESS.value, MPU6886Constants.IMU_6886_ACCEL_CONFIG2.value, [regdata])
        time.sleep(0.001)

        regdata = 0x00
        self.bus.write_i2c_block_data(MPU6886Constants.IMU_6886_ADDRESS.value, MPU6886Constants.IMU_6886_USER_CTRL.value, [regdata])
        time.sleep(0.001)

        regdata = 0x00
        self.bus.write_i2c_block_data(MPU6886Constants.IMU_6886_ADDRESS.value, MPU6886Constants.IMU_6886_FIFO_EN.value, [regdata])
        time.sleep(0.001)

        regdata = 0x22
        self.bus.write_i2c_block_data(MPU6886Constants.IMU_6886_ADDRESS.value, MPU6886Constants.IMU_6886_INT_PIN_CFG.value, [regdata])
        time.sleep(0.001)

        regdata = 0x01
        self.bus.write_i2c_block_data(MPU6886Constants.IMU_6886_ADDRESS.value, MPU6886Constants.IMU_6886_INT_ENABLE.value, [regdata])
        time.sleep(0.01)

        #self.update_gres()
        #self.update_ares()
        self.set_gyro_fsr(self.gyscale)
        self.set_accel_fsr(self.acscale)

    def get_accel_adc(self):
        buf = self.bus.read_i2c_block_data(MPU6886Constants.IMU_6886_ADDRESS.value, MPU6886Constants.IMU_6886_ACCEL_XOUT_H.value, 6)


        ax = ((buf[0] << 8) | buf[1])
        ay = ((buf[2] << 8) | buf[3])
        az = ((buf[4] << 8) | buf[5])

        return ax, ay, az

    def get_gyro_adc(self):
        buf = self.bus.read_i2c_block_data(MPU6886Constants.IMU_6886_ADDRESS.value, MPU6886Constants.IMU_6886_GYRO_XOUT_H.value, 6)

        gx = ((buf[0] << 8) | buf[1])
        gy = ((buf[2] << 8) | buf[3])
        gz = ((buf[4] << 8) | buf[5])

        return gx, gy, gz

    def get_temp_adc(self):
        buf = self.bus.read_i2c_block_data(MPU6886Constants.IMU_6886_ADDRESS.value, MPU6886Constants.IMU_6886_TEMP_OUT_H.value, 2)

        temp = ((buf[0] << 8) | buf[1])

        return temp


    def set_gyro_fsr(self, g_scale):
        regdata = (g_scale << 3)
        self.bus.write_i2c_block_data(MPU6886Constants.IMU_6886_ADDRESS.value, MPU6886Constants.IMU_6886_GYRO_CONFIG.value, [regdata])
        time.sleep(0.01)
        self.gy_scale = g_scale
        self.update_gres()

    def set_accel_fsr(self, a_scale):
        regdata = (a_scale << 3)
        self.bus.write_i2c_block_data(MPU6886Constants.IMU_6886_ADDRESS.value, MPU6886Constants.IMU_6886_ACCEL_CONFIG.value, [regdata])
        time.sleep(0.01)
        self.ac_scale = a_scale
        self.update_ares()


    def update_gres(self):
        if self.gy_scale == MPU6886Constants.GSCALE_GFS_250DPS.value:
            self.g_res = 250.0/32768.0
        elif self.gy_scale == MPU6886Constants.GSCALE_GFS_500DPS.value:
            self.g_res = 500.0/32768.0
        elif self.gy_scale == MPU6886Constants.GSCALE_GFS_1000DPS.value:
            self.g_res = 1000.0/32768.0
        elif self.gy_scale == MPU6886Constants.GSCALE_GFS_2000DPS.value:
            self.g_res = 2000.0/32768.0

    def update_ares(self):
        if self.ac_scale == MPU6886Constants.ASCALE_AFS_2G.value:
            self.a_res = 2.0/32768.0
        elif self.ac_scale == MPU6886Constants.ASCALE_AFS_4G.value:
            self.a_res = 4.0/32768.0
        elif self.ac_scale == MPU6886Constants.ASCALE_AFS_8G.value:
            self.a_res = 8.0/32768.0
        elif self.ac_scale == MPU6886Constants.ASCALE_AFS_16G.value:
            self.a_res = 16.0/32768.0


    def get_accel_data(self):
        ax, ay, az = self.get_accel_adc()

        ax *= self.a_res
        ay *= self.a_res
        az *= self.a_res

        return ax, ay, az

    def get_gyro_data(self):
        gx, gy, gz = self.get_gyro_adc()
        #print(f"get_gyro_data:gx: {gx}, gy: {gy}, gz: {gz}, self.g_res: {self.g_res}")

        gx *= self.g_res
        gy *= self.g_res
        gz *= self.g_res

        return gx, gy, gz

    def get_temp_data(self):
        temp = self.get_temp_adc()

        temp = temp/326.8 + 25.0

        return temp

    def set_gyro_offset(self, x, y, z):
        ##prepare numpy list of uint8
        buf = numpy.zeros(6, dtype=numpy.uint8)
        buf[0] = x >> 8
        buf[1] = x & 0xFF
        buf[2] = y >> 8
        buf[3] = y & 0xFF
        buf[4] = z >> 8
        buf[5] = z & 0xFF

        self.bus.write_i2c_block_data(MPU6886Constants.IMU_6886_ADDRESS.value, MPU6886Constants.IMU_6886_GYRO_OFFSET.value, buf)

    
