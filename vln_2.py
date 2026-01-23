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
        if self.curr_pose is None:
            return 0.0
        q = self.curr_pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    # ========================
    # 【依赖 /odom 的精确动作】
    # ========================
    def execute_forward_25cm(self):
        if self.curr_pose is None:
            self.get_logger().warn("未收到 /odom 数据，跳过精确前进。可使用 'f' 进行定时前进测试。")
            return
            
        self.get_logger().info("执行动作：前进 25cm（基于 odom）")
        start_x = self.curr_pose.position.x
        start_y = self.curr_pose.position.y
        self.is_running = True
        self.collision_detected = False
        
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0)
            
            dist = math.sqrt(
                (self.curr_pose.position.x - start_x) ** 2 +
                (self.curr_pose.position.y - start_y) ** 2
            )
            
            if self.collision_detected or dist >= 0.25:
                break
                
            msg = Twist()
            msg.linear.x = self.LINEAR_SPEED
            self.cmd_pub.publish(msg)
            time.sleep(0.05)
        
        self.stop_robot()

    def execute_rotate_15deg(self, direction="left"):
        if self.curr_pose is None:
            self.get_logger().warn("未收到 /odom 数据，跳过精确旋转。可使用 'l' 或 'r' 进行定时旋转测试。")
            return
            
        angle_rad = math.radians(15) if direction == "left" else -math.radians(15)
        self.get_logger().info(f"执行动作：{direction}转 15度（基于 odom）")
        
        start_yaw = self.get_yaw()
        target_yaw = start_yaw + angle_rad
        target_yaw = math.atan2(math.sin(target_yaw), math.cos(target_yaw))
        
        self.is_running = True
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0)
            current_yaw = self.get_yaw()
            diff = target_yaw - current_yaw
            diff = math.atan2(math.sin(diff), math.cos(diff))
            
            if abs(diff) < 0.02:
                break
                
            msg = Twist()
            msg.angular.z = self.ANGULAR_SPEED if diff > 0 else -self.ANGULAR_SPEED
            self.cmd_pub.publish(msg)
            time.sleep(0.05)
        
        self.stop_robot()

    # ========================
    # 【不依赖 /odom 的定时动作】
    # ========================
    def move_timed(self, linear_x=0.0, angular_z=0.0, duration_sec=1.0):
        """按固定时间发送速度指令，无需 odom"""
        self.get_logger().info(f"执行定时移动: linear.x={linear_x}, angular.z={angular_z}, 持续 {duration_sec}s")
        self.is_running = True
        self.collision_detected = False
        
        start_time = time.time()
        while rclpy.ok() and (time.time() - start_time) < duration_sec:
            # 即使没有 odom，也检查激光雷达（如果可用）
            rclpy.spin_once(self, timeout_sec=0)
            
            # 如果在定时过程中检测到碰撞，提前停止
            if self.collision_detected:
                self.get_logger().warn("定时移动因碰撞提前终止")
                break
                
            msg = Twist()
            msg.linear.x = linear_x
            msg.angular.z = angular_z
            self.cmd_pub.publish(msg)
            time.sleep(0.05)
        
        self.stop_robot()

    def stop_robot(self):
        self.cmd_pub.publish(Twist())
        self.is_running = False


def main():
    rclpy.init()
    node = VLNActionExecutor()
    
    print("\n" + "="*50)
    print("🚀 VLN 动作执行器已启动")
    print("- 若有 /odom：使用 1/2/3 执行精确动作")
    print("- 若无 /odom：使用 f/b/l/r 执行定时动作")
    print("="*50)
    print("指令说明:")
    print("  1 : 前进 25cm (需 /odom)")
    print("  2 : 左转 15°   (需 /odom)")
    print("  3 : 右转 15°   (需 /odom)")
    print("  f : 前进 1秒    (无需 /odom)")
    print("  b : 后退 1秒    (无需 /odom)")
    print("  l : 左转 1秒    (无需 /odom)")
    print("  r : 右转 1秒    (无需 /odom)")
    print("  q : 退出程序")
    print("-"*50)

    try:
        while rclpy.ok():
            user_input = input("请输入指令: ").strip().lower()
            if user_input == '1':
                node.execute_forward_25cm()
            elif user_input == '2':
                node.execute_rotate_15deg("left")
            elif user_input == '3':
                node.execute_rotate_15deg("right")
            elif user_input == 'f':
                node.move_timed(linear_x=0.2, duration_sec=1.0)
            elif user_input == 'b':
                node.move_timed(linear_x=-0.2, duration_sec=1.0)
            elif user_input == 'l':
                node.move_timed(angular_z=0.6, duration_sec=1.0)
            elif user_input == 'r':
                node.move_timed(angular_z=-0.6, duration_sec=1.0)
            elif user_input == 'q':
                break
            else:
                print("⚠️ 无效指令，请重新输入")
    except KeyboardInterrupt:
        print("\n🛑 收到中断信号，正在退出...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
