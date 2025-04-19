#!/usr/bin/env python3
import rospy
import smbus
import math
import time
import numpy as np
from std_msgs.msg import Float32
from geometry_msgs.msg import Quaternion
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from filterpy.kalman import KalmanFilter

# 参数配置
KALMAN_Q = 0.001   # 过程噪声
KALMAN_R = 0.1     # 观测噪声
HUMAN_JOINT_LIMITS = {
    'shoulder_yaw': (-90, 90),   # 大臂Yaw角限制
    'elbow_yaw': (-120, 120)     # 小臂Yaw角限制
}

# 硬件配置
PCA9548A_ADDRESS = 0x70
MPU6050_BASE_ADDRESS = 0x68
I2C_BUS = 1
NUM_SENSORS = 3

# MPU6050寄存器定义
PWR_MGMT_1 = 0x6B
ACCEL_CONFIG = 0x1C
GYRO_CONFIG = 0x1B
ACCEL_XOUT_H = 0x3B

# 量程配置
ACCEL_FS_SEL = 0x00  # ±2g
GYRO_FS_SEL = 0x00   # ±250°/s

# 转换系数
ACCEL_SCALE = 16384.0
GYRO_SCALE = 131.0
RAD_TO_DEG = 180/math.pi

# 滤波器参数
ALPHA = 0.98
DT = 0.1  # 10Hz

class MPU6050_IMU:
    def __init__(self, bus, address):
        self.bus = bus
        self.address = address
        self.init_mpu6050()
        
    def init_mpu6050(self):
        """初始化传感器"""
        try:
            self.bus.write_byte_data(self.address, PWR_MGMT_1, 0x00)
            self.bus.write_byte_data(self.address, ACCEL_CONFIG, ACCEL_FS_SEL)
            self.bus.write_byte_data(self.address, GYRO_CONFIG, GYRO_FS_SEL)
            time.sleep(0.05)
        except Exception as e:
            rospy.logerr(f"MPU6050初始化失败: {e}")

    def read_raw_data(self):
        """读取原始数据"""
        try:
            data = self.bus.read_i2c_block_data(self.address, ACCEL_XOUT_H, 14)
            return (
                [self.twos_complement((data[i] << 8) | data[i+1]) for i in range(0,6,2)],   # 加速度
                [self.twos_complement((data[i] << 8) | data[i+1]) for i in range(8,14,2)]  # 陀螺仪
            )
        except Exception as e:
            rospy.logwarn(f"传感器读取失败: {e}")
            return None, None

    @staticmethod
    def twos_complement(val, bits=16):
        """二进制补码转换"""
        return val - (1 << bits) if val >= (1 << (bits-1)) else val

class SensorFusion:
    def __init__(self):
        self.kf = [self.create_kalman() for _ in range(NUM_SENSORS)]
        self.covariances = [np.eye(4) for _ in range(NUM_SENSORS)]
        
    # 修改SensorFusion类中的create_kalman方法
    def create_kalman(self):
        kf = KalmanFilter(dim_x=4, dim_z=6)
        kf.F = np.eye(4)  # 状态转移矩阵保持4x4
        kf.H = np.zeros((6,4))  # 新的观测矩阵维度
        kf.H[:3, :3] = np.eye(3)  # 加速度观测
        kf.H[3:, 3] = 1.0         # 陀螺仪z轴观测
        kf.Q = KALMAN_Q * np.eye(4)
        kf.R = KALMAN_R * np.eye(6)
        return kf

    def dynamic_fusion(self):
        """带形状检查的动态融合"""
        fused_q = np.zeros((4,1))
        total_weight = 0
        for i in range(NUM_SENSORS):
            if self.kf[i].x.shape == (4,1):
                weight = 1/np.trace(self.covariances[i])
                fused_q += weight * self.kf[i].x
                total_weight += weight
        return fused_q / total_weight

class ArmIMUNode:
    def __init__(self):
        self.bus = smbus.SMBus(I2C_BUS)
        self.sensors = []
        self.orientations = [[0.0, 0.0, 0.0] for _ in range(NUM_SENSORS)]
        self.current_angles = [[0.0, 0.0, 0.0] for _ in range(NUM_SENSORS)]
        
        # 初始化发布者
        self.pub_joint1 = rospy.Publisher('/joint1/angles', Float32, queue_size=5)
        self.pub_joint2 = rospy.Publisher('/joint2/angles', Float32, queue_size=5)
        self.pub_joint3 = rospy.Publisher('/joint3/angles', Float32, queue_size=5)
        
        # 新增成员
        self.quaternions = [np.array([1.0,0,0,0]) for _ in range(NUM_SENSORS)]
        self.fusion = SensorFusion()
        self.tf_broadcaster = TransformBroadcaster()
        self.joint_angles_filtered = [0.0, 0.0, 0.0]

        # 初始化多路复用器通道
        self.select_channel(0)
        time.sleep(0.1)

    def select_channel(self, channel):
        """切换I2C多路复用器通道"""
        try:
            if 0 <= channel < 8:
                self.bus.write_byte(PCA9548A_ADDRESS, 1 << channel)
                time.sleep(0.01)
        except Exception as e:
            rospy.logerr(f"多路复用器错误: {e}")

    def update_quaternion(self, channel, gyro, accel):
        """使用Mahony算法更新四元数"""
        q = self.quaternions[channel]
        if np.linalg.norm(accel) == 0:
            return q
        accel = accel / np.linalg.norm(accel)
        
        # Mahony互补滤波
        v = np.array([2*(q[1]*q[3] - q[0]*q[2]),
                      2*(q[0]*q[1] + q[2]*q[3]),
                      q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2])
        e = np.cross(accel, v)
        gyro += 0.5 * e
        
        # 四元数积分
        q_dot = 0.5 * np.array([
            -q[1]*gyro[0] - q[2]*gyro[1] - q[3]*gyro[2],
            q[0]*gyro[0] + q[2]*gyro[2] - q[3]*gyro[1],
            q[0]*gyro[1] - q[1]*gyro[2] + q[3]*gyro[0], 
            q[0]*gyro[2] + q[1]*gyro[1] - q[2]*gyro[0]
        ])
        q = q + q_dot * DT
        self.quaternions[channel] = q / np.linalg.norm(q)
        return q

    def publish_tf(self, q, sensor_id):
        """发布TF坐标系"""
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "base_link"
        t.child_frame_id = f"sensor_{sensor_id}"
        t.transform.rotation = Quaternion(*q)
        self.tf_broadcaster.sendTransform(t)

    def apply_joint_limits(self, angles):
        """应用人体关节限制"""
        angles[1] = np.clip(angles[1], *HUMAN_JOINT_LIMITS['shoulder_yaw'])
        angles[2] = np.clip(angles[2], *HUMAN_JOINT_LIMITS['elbow_yaw'])
        return angles

    def quaternion_to_euler(self, q):
        """四元数转欧拉角"""
        q = q / np.linalg.norm(q)
        q0, q1, q2, q3 = q
        
        sin_p = 2*(q0*q2 - q3*q1)
        pitch = math.asin(np.clip(sin_p, -1.0, 1.0))
        
        sin_r = 2*(q0*q1 + q2*q3)
        cos_r = 1 - 2*(q1**2 + q2**2)
        roll = math.atan2(sin_r, cos_r)
        
        return roll, pitch, 0.0

    def differential_filter(self, value, joint_id):
        """差分滤波"""
        alpha = 0.9
        self.joint_angles_filtered[joint_id] = alpha * self.joint_angles_filtered[joint_id] + (1-alpha)*value
        return self.joint_angles_filtered[joint_id]

    def process_sensor(self, channel):
        """处理传感器数据"""
        self.select_channel(channel)
        
        if len(self.sensors) <= channel:
            self.sensors.append(MPU6050_IMU(self.bus, MPU6050_BASE_ADDRESS))
        
        raw_accel, raw_gyro = self.sensors[channel].read_raw_data()
        if raw_accel is None or raw_gyro is None:
            return
            
        # 单位转换
        accel = [x/ACCEL_SCALE * 9.81 for x in raw_accel]
        gyro = [x/GYRO_SCALE * math.pi/180 for x in raw_gyro]
        
        # 更新四元数
        q = self.update_quaternion(channel, np.array(gyro), np.array(accel))
        
        # 卡尔曼滤波
        self.fusion.kf[channel].predict()
        self.fusion.kf[channel].update(np.hstack([accel, gyro]))
        self.fusion.covariances[channel] = self.fusion.kf[channel].P
        
        self.current_angles[channel] = self.quaternion_to_euler(q)
        self.publish_tf(q, channel)

    def calculate_joint_angles(self):
        """计算关节角度"""
        # 从四元数获取各传感器姿态
        sensor0_roll, sensor0_pitch, _ = self.quaternion_to_euler(self.quaternions[0])
        sensor1_roll, sensor1_pitch, _ = self.quaternion_to_euler(self.quaternions[1])
        sensor2_roll, sensor2_pitch, _ = self.quaternion_to_euler(self.quaternions[2])
        
        # 关节角度计算
        joint1 = sensor2_pitch  # 使用sensor2的俯仰角
        joint2 = self.differential_filter(sensor1_roll - sensor0_roll, 1)
        joint3 = self.differential_filter(sensor2_roll - sensor1_roll, 2)
        
        # 应用限制
        limited = self.apply_joint_limits([joint1, joint2, joint3])
        return [ang * RAD_TO_DEG for ang in limited]

    def run(self):
        """主循环"""
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            for channel in range(NUM_SENSORS):
                self.process_sensor(channel)
            
            j1, j2, j3 = self.calculate_joint_angles()
            self.pub_joint1.publish(Float32(j1))
            self.pub_joint2.publish(Float32(j2))
            self.pub_joint3.publish(Float32(j3))
            
            #rospy.loginfo(f"J1: {j1:.2f}° | J2: {j2:.2f}° | J3: {j3:.2f}°")
            rate.sleep()

if __name__ == '__main__':
    try:
        rospy.init_node('nmpu6050_arm_node')
        node = ArmIMUNode()
        node.run()
    except rospy.ROSInterruptException:
        pass