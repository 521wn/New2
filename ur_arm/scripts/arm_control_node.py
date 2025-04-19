#!/usr/bin/env python3
import rospy
import math
from std_msgs.msg import Float32
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from message_filters import ApproximateTimeSynchronizer, Subscriber

class ArmControlNode:
    def __init__(self):
        rospy.init_node('arm_control_node', log_level=rospy.INFO)
        
        # 关节参数配置（新增两个关节）
        self.JOINT_LIMITS = {
            'joint1': (-math.pi/2, math.pi/2),     # ±90度
            'joint2': (-math.pi*2/3, math.pi*2/3), # ±120度
            'joint3': (-math.pi*2/3, math.pi*2/3),
            'joint4_right': (0.0, 0.0),            # 新增固定关节
            'joint4_left': (0.0, 0.0)              # 始终为0
        }

        # 初始化低通滤波器（为新增关节创建占位过滤器）
        self.filters = {
            'joint1': LowPassFilter(alpha=0.3),
            'joint2': LowPassFilter(alpha=0.3),
            'joint3': LowPassFilter(alpha=0.3),
            'joint4_right': DummyFilter(),        # 新增虚拟过滤器
            'joint4_left': DummyFilter()           # 直接返回0
        }

        # 创建发布器
        self.control_pub = rospy.Publisher('/arm_controller/command',
                                         JointTrajectory,
                                         queue_size=5)
        self.joint_state_pub = rospy.Publisher('/joint_states',
                                             JointState,
                                             queue_size=10)

        # 仅订阅需要控制的三个关节（原逻辑不变）
        joint_subs = [
            Subscriber('/joint1/angles', Float32),
            Subscriber('/joint2/angles', Float32),
            Subscriber('/joint3/angles', Float32)
        ]
        
        self.ts = ApproximateTimeSynchronizer(
            joint_subs, 
            queue_size=5,
            slop=0.01,
            allow_headerless=True
        )
        self.ts.registerCallback(self.joint_angles_callback)

        rospy.loginfo("节点初始化完成，等待关节角度数据...")

    def joint_angles_callback(self, joint1_msg, joint2_msg, joint3_msg):
        try:
            # 原始三个关节处理逻辑保持不变
            raw_angles = [
                math.radians(joint1_msg.data),
                math.radians(joint2_msg.data),
                math.radians(joint3_msg.data),
                # 新增关节的固定值
                0.0,  # joint4_right
                0.0   # joint4_left
            ]

            # 滤波处理（仅前三个关节需要滤波）
            filtered_angles = [
                self.filters['joint1'].update(raw_angles[0]),
                self.filters['joint2'].update(raw_angles[1]),
                self.filters['joint3'].update(raw_angles[2]),
                self.filters['joint4_right'].update(raw_angles[3]),
                self.filters['joint4_left'].update(raw_angles[4])
            ]

            # 安全限位处理
            safe_angles = [
                self.clamp_angle('joint1', filtered_angles[0]),
                self.clamp_angle('joint2', filtered_angles[1]),
                self.clamp_angle('joint3', filtered_angles[2]),
                self.clamp_angle('joint4_right', filtered_angles[3]),
                self.clamp_angle('joint4_left', filtered_angles[4])
            ]

            # 发布控制指令
            self.publish_joint_command(safe_angles)
            self.publish_joint_states(safe_angles)

            rospy.logdebug(f"""
            关节目标角度 (deg):
              Joint1: {math.degrees(safe_angles[0]):.1f}
              Joint2: {math.degrees(safe_angles[1]):.1f}
              Joint3: {math.degrees(safe_angles[2]):.1f}
              Joint4_R: {math.degrees(safe_angles[3]):.1f}
              Joint4_L: {math.degrees(safe_angles[4]):.1f}
            """)

        except Exception as e:
            rospy.logerr(f"关节角度处理异常: {str(e)}")

    def publish_joint_command(self, angles):
        """发布包含新增关节的控制指令"""
        trajectory = JointTrajectory()
        trajectory.header.stamp = rospy.Time.now()
        trajectory.joint_names = ["joint1", "joint2", "joint3", "joint4_right", "joint4_left"]
        
        point = JointTrajectoryPoint()
        point.positions = angles
        point.velocities = [0.0] * 5    # 所有关节速度设为0
        point.time_from_start = rospy.Duration(0.1)
        
        trajectory.points.append(point)
        self.control_pub.publish(trajectory)

    def publish_joint_states(self, angles):
        """发布包含新增关节的状态"""
        js = JointState()
        js.header.stamp = rospy.Time.now()
        js.name = ["joint1", "joint2", "joint3", "joint4_right", "joint4_left"]
        js.position = angles
        self.joint_state_pub.publish(js)

    def clamp_angle(self, joint_name, angle):
        """新增关节的强制归零"""
        if "joint4" in joint_name:
            return 0.0  # 强制固定为0
        return max(self.JOINT_LIMITS[joint_name][0],
                 min(angle, self.JOINT_LIMITS[joint_name][1]))

class LowPassFilter:
    """原有滤波器实现不变"""
    def __init__(self, alpha=0.5, beta=0.2):
        self.alpha = alpha
        self.beta = beta
        self.prev_value = 0.0
        self.filtered_value = 0.0

    def update(self, new_value):
        stage1 = self.alpha * new_value + (1 - self.alpha) * self.prev_value
        self.filtered_value = self.beta * stage1 + (1 - self.beta) * self.filtered_value
        self.prev_value = new_value
        return self.filtered_value

class DummyFilter:
    """新增虚拟过滤器用于固定关节"""
    def __init__(self):
        pass
    
    def update(self, new_value):
        return 0.0  # 始终返回0

if __name__ == '__main__':
    try:
        node = ArmControlNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("节点已安全关闭")