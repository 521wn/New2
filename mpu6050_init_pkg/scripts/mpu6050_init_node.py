#!/usr/bin/env python3
import rospy
import smbus
import math
import time
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3

# 硬件配置
PCA9548A_ADDRESS = 0x70       # 多路复用器I2C地址
MPU6050_BASE_ADDRESS = 0x68   # MPU6050默认地址
I2C_BUS = 1                   # I2C总线号
NUM_SENSORS = 3                # 传感器数量

# MPU6050寄存器定义
PWR_MGMT_1 = 0x6B
ACCEL_CONFIG = 0x1C
GYRO_CONFIG = 0x1B
ACCEL_XOUT_H = 0x3B

# 量程配置
ACCEL_FS_SEL = 0x00  # ±2g (16384 LSB/g)
GYRO_FS_SEL = 0x00   # ±250 °/s (131 LSB/°/s)

# 转换系数
ACCEL_SCALE = 16384.0    # ±2g
GYRO_SCALE = 131.0       # ±250°/s
RAD_TO_DEG = 180/math.pi

# 滤波器参数
ALPHA = 0.98   # 互补滤波系数
DT = 0.1       # 10Hz采样周期

class MPU6050_IMU:
    def __init__(self, bus, address):
        self.bus = bus
        self.address = address
        self.init_mpu6050()
        
    def init_mpu6050(self):
        """初始化MPU6050传感器"""
        try:
            self.bus.write_byte_data(self.address, PWR_MGMT_1, 0x00)
            self.bus.write_byte_data(self.address, ACCEL_CONFIG, ACCEL_FS_SEL)
            self.bus.write_byte_data(self.address, GYRO_CONFIG, GYRO_FS_SEL)
            time.sleep(0.1)
        except Exception as e:
            rospy.logerr(f"MPU6050初始化失败: {e}")

    def read_raw_data(self):
        """读取原始传感器数据"""
        try:
            data = self.bus.read_i2c_block_data(self.address, ACCEL_XOUT_H, 14)
            
            raw_accel = [
                twos_complement((data[0] << 8) | data[1]),
                twos_complement((data[2] << 8) | data[3]),
                twos_complement((data[4] << 8) | data[5])
            ]
            
            raw_gyro = [
                twos_complement((data[8] << 8) | data[9]),
                twos_complement((data[10] << 8) | data[11]),
                twos_complement((data[12] << 8) | data[13])
            ]
            
            return raw_accel, raw_gyro
        except Exception as e:
            rospy.logwarn(f"传感器读取失败: {e}")
            return None, None

def twos_complement(val, bits=16):
    return val - (1 << bits) if val >= (1 << (bits - 1)) else val

class IMU_Publisher:
    def __init__(self):
        self.bus = smbus.SMBus(I2C_BUS)
        self.sensors = []
        self.orientations = [[0.0, 0.0, 0.0] for _ in range(NUM_SENSORS)]
        
        # 修改发布者：只创建三个imu_rpy发布者，编号从1开始
        self.pubs = []
        for i in range(1, NUM_SENSORS+1):  # 生成1,2,3
            pub = rospy.Publisher(f'/sensor{i}/imu_rpy', Vector3, queue_size=10)
            self.pubs.append(pub)
        
        self.select_channel(0)
        time.sleep(0.1)
        
    def select_channel(self, channel):
        try:
            if 0 <= channel < 8:
                self.bus.write_byte(PCA9548A_ADDRESS, 1 << channel)
                time.sleep(0.01)
        except Exception as e:
            rospy.logerr(f"多路复用器通信失败: {e}")

    def process_sensor(self, channel):
        self.select_channel(channel)
        
        if len(self.sensors) <= channel:
            sensor = MPU6050_IMU(self.bus, MPU6050_BASE_ADDRESS)
            self.sensors.append(sensor)
        
        raw_accel, raw_gyro = self.sensors[channel].read_raw_data()
        if raw_accel is None or raw_gyro is None:
            return
        
        # 物理单位转换保持不变
        accel = [x/ACCEL_SCALE * 9.81 for x in raw_accel]
        gyro = [x/GYRO_SCALE * math.pi/180 for x in raw_gyro]
        
        # 姿态计算保持不变
        accel_roll = math.atan2(accel[1], math.sqrt(accel[0]**2 + accel[2]**2))
        accel_pitch = math.atan2(-accel[0], math.sqrt(accel[1]**2 + accel[2]**2))
        
        self.orientations[channel][0] = ALPHA * (self.orientations[channel][0] + gyro[0]*DT) + (1-ALPHA)*accel_roll
        self.orientations[channel][1] = ALPHA * (self.orientations[channel][1] + gyro[1]*DT) + (1-ALPHA)*accel_pitch
        self.orientations[channel][2] += gyro[2] * DT
        
        # 修改发布逻辑：只发布RPY数据
        rpy_msg = Vector3()
        rpy = [x * RAD_TO_DEG for x in self.orientations[channel]]
        rpy_msg.x = rpy[0]
        rpy_msg.y = rpy[1]
        rpy_msg.z = rpy[2]
        
        # 注意索引调整：通道0对应传感器1
        self.pubs[channel].publish(rpy_msg)
        
        # 打印信息保持不变
        print(f"\n=== Sensor {channel+1} ===")  # 显示编号从1开始
        print(f"Accel [m/s²]: X={accel[0]:.2f}, Y={accel[1]:.2f}, Z={accel[2]:.2f}")
        print(f"Gyro [rad/s]: X={gyro[0]:.2f}, Y={gyro[1]:.2f}, Z={gyro[2]:.2f}")
        print(f"RPY [deg]: Roll={rpy[0]:.1f}, Pitch={rpy[1]:.1f}, Yaw={rpy[2]:.1f}")

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            for channel in range(NUM_SENSORS):
                self.process_sensor(channel)
            rate.sleep()

if __name__ == '__main__':
    try:
        rospy.init_node('mpu6050_imu_array')
        publisher = IMU_Publisher()
        publisher.run()
    except rospy.ROSInterruptException:
        pass