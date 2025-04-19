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
        
    def create_kalman(self):
        """修正后的卡尔曼滤波器配置"""
        kf = KalmanFilter(dim_x=4, dim_z=6)
        kf.x = np.array([[1.0], [0.0], [0.0], [0.0]])  # 初始状态（四元数）
        kf.F = np.eye(4)  # 状态转移矩阵
        kf.H = np.zeros((6,4))
        kf.H[:3, :3] = np.eye(3)  # 加速度观测
        kf.H[3:, 3] = 1.0         # 陀螺仪观测
        kf.P = np.eye(4) * 0.1    # 初始协方差
        kf.Q = KALMAN_Q * np.eye(4)
        kf.R = KALMAN_R * np.eye(6)
        return kf

    def dynamic_fusion(self):
        """动态加权融合"""
        valid_kf = [k for k in self.kf if np.linalg.norm(k.x) > 0.9]
        if not valid_kf:
            return np.array([1.0, 0.0, 0.0, 0.0])
        
        weights = [1/np.trace(c) for c in self.covariances]
        total = sum(weights)
        fused_q = np.zeros(4)
        for i in range(len(valid_kf)):
            fused_q += weights[i]/total * valid_kf[i].x.flatten()
        return fused_q / np.linalg.norm(fused_q)

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
        """使用改进的Mahony算法更新四元数"""
        q = self.quaternions[channel].copy()
        
        # 加速度计数据校验
        accel_norm = np.linalg.norm(accel)
        if accel_norm < 1e-6 or np.any(np.isnan(accel)):
            rospy.logwarn_once(f"传感器{channel}加速度数据异常")
            return q
        
        accel = accel / accel_norm
        
        # Mahony互补滤波
        v = np.array([
            2*(q[1]*q[3] - q[0]*q[2]),
            2*(q[0]*q[1] + q[2]*q[3]),
            q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2
        ])
        
        # 误差计算（增加数值保护）
        e = np.cross(accel, v)
        e = np.nan_to_num(e)
        
        # 误差积分
        gyro_corrected = gyro + 0.5 * e
        
        # 四元数微分方程（增加数值稳定性）
        q_dot = 0.5 * np.array([
            -q[1]*gyro_corrected[0] - q[2]*gyro_corrected[1] - q[3]*gyro_corrected[2],
             q[0]*gyro_corrected[0] + q[2]*gyro_corrected[2] - q[3]*gyro_corrected[1],
             q[0]*gyro_corrected[1] - q[1]*gyro_corrected[2] + q[3]*gyro_corrected[0],
             q[0]*gyro_corrected[2] + q[1]*gyro_corrected[1] - q[2]*gyro_corrected[0]
        ])
        
        # 积分并标准化
        q = q + q_dot * DT
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-6 or np.any(np.isnan(q)):
            rospy.logwarn(f"传感器{channel}四元数异常，重置")
            return np.array([1.0, 0.0, 0.0, 0.0])
        
        q_normalized = q / q_norm
        return q_normalized

    def publish_tf(self, q, sensor_id):
        """安全发布TF坐标系"""
        # 强制标准化并检查有效性
        q_normalized = q / np.linalg.norm(q)
        if (abs(np.linalg.norm(q_normalized) - 1.0) > 0.01) or np.any(np.isnan(q_normalized)):
            rospy.logwarn_throttle(1, f"传感器{sensor_id}四元数无效: {q_normalized}")
            return
        
        try:
            t = TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "base_link"
            t.child_frame_id = f"sensor_{sensor_id}"
            t.transform.rotation = Quaternion(*q_normalized)
            self.tf_broadcaster.sendTransform(t)
        except Exception as e:
            rospy.logerr(f"TF发布失败: {e}")

    def apply_joint_limits(self, angles):
        """应用人体关节限制"""
        angles[1] = np.clip(angles[1], *HUMAN_JOINT_LIMITS['shoulder_yaw'])
        angles[2] = np.clip(angles[2], *HUMAN_JOINT_LIMITS['elbow_yaw'])
        return angles

    def quaternion_to_euler(self, q):
        """安全的四元数转欧拉角"""
        q = np.nan_to_num(q)
        q_norm = q / np.linalg.norm(q)
        q0, q1, q2, q3 = q_norm
        

        # 计算俯仰角（pitch）
        sin_p = 2*(q0*q2 - q3*q1)
        sin_p = np.clip(sin_p, -1.0, 1.0)
        pitch = math.asin(sin_p)
        
        # 计算横滚角（roll）
        sin_r = 2*(q0*q1 + q2*q3)
        cos_r = 1 - 2*(q1**2 + q2**2)
        cos_r = np.clip(cos_r, -1.0, 1.0)
        roll = math.atan2(sin_r, cos_r)
        
        return roll, pitch, 0.0

    def differential_filter(self, value, joint_id):
        """改进的差分滤波器"""
        alpha = 0.9
        if np.isnan(value):
            return self.joint_angles_filtered[joint_id]
        self.joint_angles_filtered[joint_id] = alpha * self.joint_angles_filtered[joint_id] + (1-alpha)*value
        return self.joint_angles_filtered[joint_id]

    def process_sensor(self, channel):
        """传感器数据处理流程"""
        self.select_channel(channel)
        
        # 延迟初始化传感器对象
        if len(self.sensors) <= channel:
            self.sensors.append(MPU6050_IMU(self.bus, MPU6050_BASE_ADDRESS))
        
        # 读取原始数据
        raw_accel, raw_gyro = self.sensors[channel].read_raw_data()
        if raw_accel is None or raw_gyro is None:
            return
            
        try:
            # 单位转换
            accel = np.array([x/ACCEL_SCALE * 9.81 for x in raw_accel])
            gyro = np.array([x/GYRO_SCALE * math.pi/180 for x in raw_gyro])
            
            # 更新四元数
            self.quaternions[channel] = self.update_quaternion(channel, gyro, accel)
            
            # 卡尔曼滤波处理
            measurement = np.hstack([accel, gyro]).reshape(-1,1)
            self.fusion.kf[channel].predict()
            self.fusion.kf[channel].update(measurement)
            self.fusion.covariances[channel] = self.fusion.kf[channel].P
            
            # 更新当前角度
            self.current_angles[channel] = self.quaternion_to_euler(self.quaternions[channel])
            
            # 发布TF
            self.publish_tf(self.quaternions[channel], channel)
        except Exception as e:
            rospy.logerr_throttle(1, f"传感器{channel}处理异常: {e}")

    def calculate_joint_angles(self):
        """改进的关节角度计算"""
        sensor0_roll, sensor0_pitch, _ = self.quaternion_to_euler(self.quaternions[0])
        sensor1_roll, sensor1_pitch, _ = self.quaternion_to_euler(self.quaternions[1])
        sensor2_roll, sensor2_pitch, _ = self.quaternion_to_euler(self.quaternions[2])
        
        # 关节角度计算（可配置差分顺序）
        JOINT2_SIGN = 1  # 尝试1或-1
        
        joint1 = sensor2_pitch
        joint2 = self.differential_filter(
            (sensor1_roll - sensor0_roll) * JOINT2_SIGN, 1
        )
        joint3 = self.differential_filter(
            (sensor2_roll - sensor1_roll) * JOINT2_SIGN, 2
        )
        
        # 应用限制
        limited = self.apply_joint_limits([joint1, joint2, joint3])
        return [ang * RAD_TO_DEG for ang in limited]

    def run(self):
        """主循环（增加异常处理）"""
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            try:
                for channel in range(NUM_SENSORS):
                    self.process_sensor(channel)
                
                j1, j2, j3 = self.calculate_joint_angles()
                self.pub_joint1.publish(Float32(j1))
                self.pub_joint2.publish(Float32(j2))
                self.pub_joint3.publish(Float32(j3))


                
                rate.sleep()
            except rospy.ROSInterruptException:
                break
            except Exception as e:
                rospy.logerr_throttle(1, f"主循环异常: {e}")
                time.sleep(1)

if __name__ == '__main__':
    try:
        rospy.init_node('nmpu6050_arm_node')
        node = ArmIMUNode()
        node.run()
    except rospy.ROSInterruptException:
        pass