#!/usr/bin/env python3
import rospy
import serial
from std_msgs.msg import Float32
class HTS16L_Controller:
    def __init__(self):
        # 初始化串口
        self.ser = serial.Serial(
            port='/dev/ttyS0',
            baudrate=115200,
            timeout=0.1
        )
        
        # 舵机参数
        self.servo_id = 0x01  # 默认舵机ID
        self.min_angle = -120  # 机械限位
        self.max_angle = 120
        
        # ROS订阅
        rospy.Subscriber("/servo_angle", Float32, self.angle_callback)
        rospy.loginfo("HTS-16L控制器已就绪")

    def _build_cmd(self, angle_deg):
        """构建控制指令"""
        # 角度限幅和单位转换
        angle = int(max(self.min_angle, min(angle_deg, self.max_angle)))/ 0.36
        angle_val = int(angle) & 0xFFFF
        
        cmd = [
            0x55, 0x55,       # Header
            0x08,             # Length
            self.servo_id,    # Servo ID
            0x03,             # Write Command
            0x2A,             # Position Register
            (angle_val >> 8) & 0xFF,  # Position High
            angle_val & 0xFF,         # Position Low
            0x00, 0x00        # Speed (max)
        ]
        checksum = sum(cmd[2:]) & 0xFF
        return bytes(cmd + [checksum])

    def angle_callback(self, msg):
        """角度指令回调"""
        rospy.loginfo(f"目标角度: {msg.data}°")
        cmd = self._build_cmd(msg.data)
        self.ser.write(cmd)
        
        # 读取响应（可选）
        resp = self.ser.read(8)
        if resp:
            rospy.logdebug(f"响应: {resp.hex()}")

    def cleanup(self):
        self.ser.close()

if __name__ == '__main__':
    rospy.init_node('hts16l_controller')
    ctrl = None  # 初始化变量
    try:
        ctrl = HTS16L_Controller()
        rospy.spin()
    except Exception as e:
        rospy.logerr(f"初始化失败: {str(e)}")
    finally:
        if ctrl:  # 确保变量存在
            ctrl.cleanup()