import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import math
import time

class VLNActionExecutor(Node):
    def __init__(self):
        super().__init__('vln_action_executor')
        
        # 发布与订阅
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        # 状态变量
        self.curr_pose = None
        self.collision_detected = False
        self.collision_count = 0
        self.is_running = False
        
        # 配置参数
        self.LINEAR_SPEED = 0.1  # 前进速度 (m/s)
        self.ANGULAR_SPEED = 0.3 # 旋转速度 (rad/s)
        self.COLLISION_THRESHOLD = 0.3 # 碰撞阈值 (米)

    def odom_callback(self, msg):
        self.curr_pose = msg.pose.pose

    def scan_callback(self, msg):
        # 提取正前方 60 度范围内的最小距离
        # 假设 lidar 0度是正前，通常消息是从 -PI 到 PI
        ranges = msg.ranges
        num_points = len(ranges)
        mid = num_points // 2
        # 取中间 1/6 的范围 (约 60度)
        front_view = ranges[mid-30 : mid+30] 
        valid_ranges = [r for r in front_view if r > msg.range_min and r < msg.range_max]
        
        if valid_ranges and min(valid_ranges) < self.COLLISION_THRESHOLD:
            if self.is_running and not self.collision_detected:
                self.collision_detected = True
                self.collision_count += 1
                self.get_logger().error(f"！！！检测到碰撞停止！！！ 总次数: {self.collision_count}")

    def get_yaw(self):
        """从四元数中提取偏航角(Yaw)"""
        q = self.curr_pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def execute_forward_25cm(self):
        self.get_logger().info("执行动作：前进 25cm")
        if self.curr_pose is None: return
        
        start_x = self.curr_pose.position.x
        start_y = self.curr_pose.position.y
        self.is_running = True
        self.collision_detected = False
        
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            dist = math.sqrt((self.curr_pose.position.x - start_x)**2 + 
                             (self.curr_pose.position.y - start_y)**2)
            
            if self.collision_detected or dist >= 0.25:
                break
            
            msg = Twist()
            msg.linear.x = self.LINEAR_SPEED
            self.cmd_pub.publish(msg)
            
        self.stop_robot()

    def execute_rotate_15deg(self, direction="left"):
        angle_rad = 0.2618 if direction == "left" else -0.2618
        self.get_logger().info(f"执行动作：{direction}转 15度")
        if self.curr_pose is None: return
        
        start_yaw = self.get_yaw()
        target_yaw = start_yaw + angle_rad
        
        # 角度标准化到 -PI 到 PI
        target_yaw = math.atan2(math.sin(target_yaw), math.cos(target_yaw))
        
        self.is_running = True
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            current_yaw = self.get_yaw()
            
            # 计算剩余角度差
            diff = math.atan2(math.sin(target_yaw - current_yaw), math.cos(target_yaw - current_yaw))
            
            if abs(diff) < 0.02: # 容差
                break
            
            msg = Twist()
            msg.angular.z = self.ANGULAR_SPEED if diff > 0 else -self.ANGULAR_SPEED
            self.cmd_pub.publish(msg)
            
        self.stop_robot()

    def stop_robot(self):
        self.cmd_pub.publish(Twist())
        self.is_running = False
        time.sleep(0.5) # 停稳

def main():
    rclpy.init()
    node = VLNActionExecutor()
    
    print("--- VLN 动作执行器已就绪 ---")
    print("输入 1:前进25cm, 2:左转15°, 3:右转15°, q:退出")
    
    try:
        while rclpy.ok():
            # 这里先用手动输入模拟 VLN 模型的动作序列
            user_input = input("等待指令: ")
            if user_input == '1':
                node.execute_forward_25cm()
            elif user_input == '2':
                node.execute_rotate_15deg("left")
            elif user_input == '3':
                node.execute_rotate_15deg("right")
            elif user_input == 'q':
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
