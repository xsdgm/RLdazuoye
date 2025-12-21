import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Pose
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
import numpy as np
import math
import os
import time
import random
import threading
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class GazeboEnv(Node):
    def __init__(self):
        super().__init__('rl_navigation_env')
        
        # Constants
        self.GOAL_DIST_THRESHOLD = 0.2
        self.COLLISION_DIST = 0.18
        self.MAX_STEPS = 500
        
        # Action Limits (TurtleBot3 Burger)
        self.MAX_V = 0.22
        self.MAX_W = 2.84
        
        # State variables
        self.scan_data = None
        self.imu_data = None
        self.odom_data = None
        self.current_goal = None
        self.last_action = np.array([0.0, 0.0])
        self.step_count = 0
        self.robot_name = 'burger'  # Entity name in Gazebo
        
        # QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos)
        self.create_subscription(Imu, '/imu', self.imu_callback, qos)
        self.create_subscription(Odometry, '/odom', self.odom_callback, qos)
        
        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Service Clients
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_client = self.create_client(DeleteEntity, '/delete_entity')
        self._wait_for_spawn_delete_services()

        # Load robot SDF once for respawn
        model_path = os.path.join(
            get_package_share_directory('turtlebot3_gazebo'),
            'models',
            'turtlebot3_burger',
            'model.sdf'
        )
        with open(model_path, 'r') as f:
            self.robot_sdf = f.read()
        
        # Wait for service
        # Note: We don't block here indefinitely because sometimes the service takes time to appear
        # or we might want to fallback. But for now we follow user instructions.
        
        # Define house boundaries (approximate for TurtleBot3 House)
        # The house layout is roughly within these bounds
        self.house_bounds = {
            'x_min': -3.0,
            'x_max': 3.0,
            'y_min': -2.5,
            'y_max': 2.5
        }
        
        # Define wall-free zones for spawning and goals (approximate room areas)
        # Format: (x_min, x_max, y_min, y_max)
        self.safe_zones = [
            # Living room / Center area
            (-2.5, -0.5, -1.0, 1.5),
            # Right side rooms
            (0.3, 2.5, -1.5, 1.5),
            # Back area
            (-1.5, 1.5, 1.2, 2.2),
            # Bottom area
            (-2.0, 1.5, -2.0, -0.8)
        ]
        
        # Margin from walls for safety
        self.spawn_margin = 0.3
        self.goal_margin = 0.2
        
        self.lock = threading.Lock()
        
    def scan_callback(self, msg):
        with self.lock:
            self.scan_data = msg
            
    def imu_callback(self, msg):
        with self.lock:
            self.imu_data = msg
            
    def odom_callback(self, msg):
        with self.lock:
            self.odom_data = msg
            
    def get_state(self):
        # Wait for data
        while self.scan_data is None or self.imu_data is None or self.odom_data is None:
            time.sleep(0.01)
            
        with self.lock:
            # 1. LiDAR (360)
            # Normalize and handle inf
            ranges = np.array(self.scan_data.ranges)
            ranges = np.nan_to_num(ranges, nan=3.5, posinf=3.5, neginf=0.0)
            ranges = np.clip(ranges, 0.0, 3.5)
            lidar_state = ranges / 3.5
            
            # 2. IMU (ax, ay, wz)
            ax = self.imu_data.linear_acceleration.x
            ay = self.imu_data.linear_acceleration.y
            wz = self.imu_data.angular_velocity.z
            
            # 3. Goal (rho, psi)
            px = self.odom_data.pose.pose.position.x
            py = self.odom_data.pose.pose.position.y
            
            # Quaternion to Euler (Yaw)
            q = self.odom_data.pose.pose.orientation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            gx, gy = self.current_goal
            
            dist = math.sqrt((gx - px)**2 + (gy - py)**2)
            target_angle = math.atan2(gy - py, gx - px)
            heading_error = target_angle - yaw
            
            # Normalize angle to [-pi, pi]
            while heading_error > math.pi: heading_error -= 2 * math.pi
            while heading_error < -math.pi: heading_error += 2 * math.pi
            
            # 4. Last Action
            lv, lw = self.last_action
            
            vector_state = np.array([ax, ay, wz, dist, heading_error, lv, lw])
            
            return lidar_state, vector_state, dist, heading_error
            
    def _wait_for_spawn_delete_services(self):
        for _ in range(10):
            ok_spawn = self.spawn_client.wait_for_service(timeout_sec=0.5)
            ok_delete = self.delete_client.wait_for_service(timeout_sec=0.5)
            if ok_spawn and ok_delete:
                return True
            self.get_logger().warn('Waiting for /spawn_entity and /delete_entity services...')
        self.get_logger().error('Spawn/Delete services not available; respawn may fail.')
        return False

    def respawn_robot(self, x, y, yaw):
        # Delete existing entity (best-effort)
        if self.delete_client.service_is_ready():
            del_req = DeleteEntity.Request()
            del_req.name = self.robot_name
            self.delete_client.call_async(del_req)

        if not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('spawn_entity service unavailable, skip respawn')
            return False

        spawn_req = SpawnEntity.Request()
        spawn_req.name = self.robot_name
        spawn_req.xml = self.robot_sdf
        spawn_req.robot_namespace = ''
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = 0.01
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        pose.orientation.w = cy
        pose.orientation.z = sy
        spawn_req.initial_pose = pose

        future = self.spawn_client.call_async(spawn_req)
        start = time.time()
        while not future.done():
            if time.time() - start > 5.0:
                self.get_logger().error('spawn_entity call timed out')
                return False
            time.sleep(0.05)

        result = future.result()
        if result is None or not getattr(result, 'success', True):
            self.get_logger().warn('spawn_entity failed; robot may stay at old pose')
            return False
        return True
        
    def generate_random_spawn_pose(self):
        """Generate a random spawn position within safe zones of the house."""
        # Randomly select a safe zone
        zone = random.choice(self.safe_zones)
        x_min, x_max, y_min, y_max = zone
        
        # Add margin to avoid spawning too close to walls
        x_min += self.spawn_margin
        x_max -= self.spawn_margin
        y_min += self.spawn_margin
        y_max -= self.spawn_margin
        
        # Generate random position within the zone
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        z = 0.01
        
        # Generate random yaw (orientation) in [0, 2*pi]
        yaw = random.uniform(0, 2 * math.pi)
        
        return x, y, z, yaw
    
    def generate_random_goal(self):
        """Generate a random goal position within safe zones of the house."""
        # Randomly select a safe zone
        zone = random.choice(self.safe_zones)
        x_min, x_max, y_min, y_max = zone
        
        # Add margin to avoid goals too close to walls
        x_min += self.goal_margin
        x_max -= self.goal_margin
        y_min += self.goal_margin
        y_max -= self.goal_margin
        
        # Generate random position within the zone
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        
        return x, y
        
    def reset(self):
        # Stop robot
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        
        # Generate random spawn pose within house
        sx, sy, sz, syaw = self.generate_random_spawn_pose()
        
        # Respawn robot at random pose
        success = self.respawn_robot(sx, sy, syaw)
        if not success:
            self.get_logger().warn('Respawn failed; robot may stay at previous location')
        
        # Generate random goal position within house
        self.current_goal = self.generate_random_goal()
        
        self.step_count = 0
        self.last_action = np.array([0.0, 0.0])
        time.sleep(0.5)


        
        lidar, state, _, _ = self.get_state()
        return lidar, state
        
    def step(self, action):
        # Action: [v_scale, w_scale] in [-1, 1]
        # Map to [0, MAX_V] and [-MAX_W, MAX_W]
        
        # v: [-1, 1] -> [0, MAX_V]
        v = (action[0] + 1) / 2 * self.MAX_V
        
        # w: [-1, 1] -> [-MAX_W, MAX_W]
        w = action[1] * self.MAX_W
        
        # Publish action
        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.cmd_vel_pub.publish(twist)
        
        self.last_action = np.array([v, w])
        
        # Wait for action to take effect
        time.sleep(0.1) # 10Hz control
        
        # Get new state
        lidar, state, dist, heading_error = self.get_state()
        
        # Calculate Reward
        reward = 0
        done = False
        
        # 1. Goal Reward
        if dist < self.GOAL_DIST_THRESHOLD:
            reward += 200
            done = True
            print("Goal Reached!")
            
        # 2. Collision Penalty
        min_dist = np.min(lidar * 3.5)
        if min_dist < self.COLLISION_DIST:
            reward -= 100
            done = True
            print("Collision!")
            
        # 3. Approach Reward (Simplified)
        # Ideally we need previous distance, but for now just distance penalty
        reward -= dist * 0.1

        # Proximity reward: closer to goal yields larger bonus
        proximity_bonus = 1.0 / (dist + 0.5)
        reward += proximity_bonus
        
        # 4. Smoothness/Time Penalty
        reward -= 0.01 # Time step
        
        self.step_count += 1
        if self.step_count >= self.MAX_STEPS:
            done = True
            
        return lidar, state, reward, done, {}

