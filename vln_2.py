#!/usr/bin/env python3

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
        self.LINEAR_SPEED = 0.1      # 前进速度 (m/s)
        self.ANGULAR_SPEED = 0.3     # 旋转速度 (rad/s)
        self.COLLISION_THRESHOLD = 0.3  # 碰撞阈值 (米)

    def odom_callback(self, msg):
        self.curr_pose = msg.pose.pose

    def scan_callback(self, msg):
        ranges = msg.ranges
        if not ranges:
            return
            
        num_points = len(ranges)
        mid = num_points // 2
        # 取正前方约60度范围（±30个点，假设分辨率足够）
        start_idx = max(0, mid - 30)
        end_idx = min(num_points, mid + 30)
        front_view = ranges[start_idx:end_idx]
        
        valid_ranges = [
            r for r in front_view 
            if msg.range_min < r < msg.range_max
        ]
        
        if valid_ranges and min(valid_ranges) < self.COLLISION_THRESHOLD:
            if self.is_running and not self.collision_detected:
                self.collision_detected = True
                self.collision_count += 1
                self.get_logger().error(f"！！！检测到碰撞停止！！！ 总次数: {self.collision_count}")

    def get_yaw(self):
        """从四元数中提取偏航角(Yaw)，范围 [-π, π]"""
        if self.curr_pose is None:
            return 0.0
        q = self.curr_pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def execute_forward_25cm(self):
        if self.curr_pose is None:
            self.get_logger().error("无法前进：尚未收到 /odom 数据！")
            return
            
        self.get_logger().info("执行动作：前进 25cm")
        start_x = self.curr_pose.position.x
        start_y = self.curr_pose.position.y
        self.is_running = True
        self.collision_detected = False
        
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0)  # 非阻塞处理回调
            
            # 检查是否到达目标或碰撞
            dist = math.sqrt(
                (self.curr_pose.position.x - start_x) ** 2 +
                (self.curr_pose.position.y - start_y) ** 2
            )
            
            if self.collision_detected or dist >= 0.25:
                break
                
            # 发布前进指令
            msg = Twist()
            msg.linear.x = self.LINEAR_SPEED
            self.cmd_pub.publish(msg)
            
            time.sleep(0.05)  # 控制频率 ≈20 Hz
        
        self.stop_robot()

    def execute_rotate_15deg(self, direction="left"):
        if self.curr_pose is None:
            self.get_logger().error("无法旋转：尚未收到 /odom 数据！")
            return
            
        angle_rad = math.radians(15) if direction == "left" else -math.radians(15)
        self.get_logger().info(f"执行动作：{direction}转 15度")
        
        start_yaw = self.get_yaw()
        target_yaw = start_yaw + angle_rad
        # 标准化到 [-π, π]
        target_yaw = math.atan2(math.sin(target_yaw), math.cos(target_yaw))
        
        self.is_running = True
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0)
            current_yaw = self.get_yaw()
            
            # 计算最短角度差
            diff = target_yaw - current_yaw
            diff = math.atan2(math.sin(diff), math.cos(diff))  # 归一化到 [-π, π]
            
            if abs(diff) < 0.02:  # 误差 < 1.15度
                break
                
            msg = Twist()
            msg.angular.z = self.ANGULAR_SPEED if diff > 0 else -self.ANGULAR_SPEED
            self.cmd_pub.publish(msg)
            
            time.sleep(0.05)  # 控制频率 ≈20 Hz
        
        self.stop_robot()

    def stop_robot(self):
        """发送零速度指令并重置状态"""
        self.cmd_pub.publish(Twist())  # 所有字段默认为 0.0
        self.is_running = False


def main():
    rclpy.init()
    node = VLNActionExecutor()
    
    print("\n--- VLN 动作执行器已就绪 ---")
    print("输入 1: 前进25cm, 2: 左转15°, 3: 右转15°, q: 退出\n")
    
    try:
        while rclpy.ok():
            user_input = input("等待指令: ").strip()
            if user_input == '1':
                node.execute_forward_25cm()
            elif user_input == '2':
                node.execute_rotate_15deg("left")
            elif user_input == '3':
                node.execute_rotate_15deg("right")
            elif user_input == 'q':
                break
            else:
                print("无效输入，请输入 1 / 2 / 3 / q")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
