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
    def __init__(self, goal_tolerance: float = 0.2, collision_penalty: float = 50.0):
        super().__init__('rl_navigation_env')
        
        # Constants
        # Distance within which the goal is considered reached
        self.GOAL_DIST_THRESHOLD = float(goal_tolerance)
        self.COLLISION_DIST = 0.18
        self.COLLISION_PENALTY = float(collision_penalty)
        self.MAX_STEPS = 500

        # Anti-spin shaping
        self.SPIN_V_THRESHOLD = 0.02   # m/s, consider near-zero forward motion
        self.SPIN_W_THRESHOLD = 0.3    # rad/s, turning considered significant
        self.SPIN_PENALTY = 0.2        # penalty per step for spinning in place

        # Heading alignment shaping
        self.HEADING_ALIGN_GAIN = 0.5  # reward for reducing |heading_error|

        # Forward bias when far from goal
        self.FORWARD_BIAS_GAIN = 0.05  # small reward proportional to forward speed
        
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
        self.prev_goal_dist = None  # Track previous distance to goal for shaped reward
        self.APPROACH_GAIN = 2.0    # Reward weight for getting closer to the goal
        self.robot_name = 'burger'  # Entity name in Gazebo
        self.last_collision_time = None  # Track last collision time for cooldown
        self.COLLISION_COOLDOWN = 2.0  # Seconds before applying collision penalty again
        
        # Flip detection thresholds (in radians)
        self.FLIP_ROLL_THRESHOLD = 0.35  # ~20 degrees
        self.FLIP_PITCH_THRESHOLD = 0.35  # ~20 degrees
        
        # QoS-
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
        
        # Define house boundaries (TurtleBot3 House - U-shaped asymmetric)
        # Based on actual wall positions from model.sdf:
        # - Left wall: x ~ -7.5
        # - Right wall: x ~ 6.2  
        # - Top wall: y ~ 5.275
        # - Bottom wall: y ~ -3.925
        # U-shape opens toward the bottom-right
        self.house_bounds = {
            'x_min': -7.3,
            'x_max': 6.0,
            'y_min': -3.8,
            'y_max': 5.1
        }
        
        # Define safe zones for spawning and goals (U-shaped house)
        # The house is U-shaped and asymmetric, opening toward bottom-right
        # Format: (x_min, x_max, y_min, y_max, name)
        self.safe_zones = [
            # Top-left room (large room at -7.5, 5.275)
            (-7.0, -5.5, 3.5, 5.0, "Top-Left Room"),
            
            # Top-middle-left room
            (-4.5, -2.8, 3.5, 5.0, "Top-Middle-Left Room"),
            
            # Top-middle-right room  
            (-0.5, 1.0, 3.5, 5.0, "Top-Middle-Right Room"),
            
            # Top-right room (at 5.07752, 5.2688)
            (3.5, 5.5, 3.5, 5.0, "Top-Right Room"),
            
            # Left corridor (vertical hallway on left side)
            (-7.0, -5.5, 0.5, 2.8, "Left Corridor"),
            
            # Bottom-left room (at -6.325, -3.925)
            (-7.0, -5.5, -3.5, -1.8, "Bottom-Left Room"),
            
            # Central area (middle open space)
            (-2.5, 1.5, -0.5, 2.5, "Central Hall"),
            
            # Right corridor area
            (2.0, 5.5, -0.5, 2.5, "Right Corridor"),
            
            # Note: Bottom-right is OPEN (U-shape opening), so we avoid this area
            # The U opens to the bottom-right quadrant
        ]
        
        # Margin from walls for safety (to avoid collisions with walls)
        # Larger margins for U-shaped asymmetric house
        self.spawn_margin = 0.4
        self.goal_margin = 0.35
        
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
            
    def _quaternion_to_euler(self, q):
        """Convert quaternion to euler angles (roll, pitch, yaw)."""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def _is_robot_flipped(self):
        """Check if robot is flipped based on roll and pitch angles."""
        if self.odom_data is None:
            return False, 0.0, 0.0
        
        with self.lock:
            q = self.odom_data.pose.pose.orientation
            roll, pitch, yaw = self._quaternion_to_euler(q)
        
        is_flipped = (abs(roll) > self.FLIP_ROLL_THRESHOLD or 
                     abs(pitch) > self.FLIP_PITCH_THRESHOLD)
        
        return is_flipped, roll, pitch
    
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
            
            # Quaternion to Euler (get yaw)
            q = self.odom_data.pose.pose.orientation
            roll, pitch, yaw = self._quaternion_to_euler(q)
            
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
        # Delete existing entity and WAIT for completion
        if self.delete_client.service_is_ready():
            del_req = DeleteEntity.Request()
            del_req.name = self.robot_name
            del_future = self.delete_client.call_async(del_req)
            
            # Wait for delete to complete
            start_del = time.time()
            while not del_future.done():
                if time.time() - start_del > 3.0:
                    self.get_logger().warn('delete_entity call timed out, proceeding anyway')
                    break
                time.sleep(0.05)
            
            # Give Gazebo extra time to fully remove the entity
            time.sleep(0.3)

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
            if time.time() - start > 8.0:  # Increased timeout
                self.get_logger().error('spawn_entity call timed out')
                return False
            time.sleep(0.05)

        result = future.result()
        if result is None or not getattr(result, 'success', True):
            self.get_logger().warn('spawn_entity failed; robot may stay at old pose')
            return False
        
        # Give time for robot to stabilize in Gazebo
        time.sleep(0.2)
        return True
        
    def _is_within_house(self, x, y):
        """Check if position is within the house boundaries."""
        return (self.house_bounds['x_min'] <= x <= self.house_bounds['x_max'] and
                self.house_bounds['y_min'] <= y <= self.house_bounds['y_max'])
    
    def _get_room_name(self, x, y):
        """Determine which room the position is in."""
        for zone in self.safe_zones:
            x_min, x_max, y_min, y_max, name = zone
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return name
        return "Unknown"
    
    def _is_valid_position(self, x, y):
        """Check if position is within safe zone and house bounds."""
        if not self._is_within_house(x, y):
            return False
        
        # Check if position is within any safe zone
        for zone in self.safe_zones:
            x_min, x_max, y_min, y_max, name = zone
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return True
        return False

    def generate_random_spawn_pose(self):
        """Generate a random spawn position within safe zones of the house."""
        max_attempts = 50
        for attempt in range(max_attempts):
            # Randomly select a safe zone
            zone = random.choice(self.safe_zones)
            x_min, x_max, y_min, y_max, zone_name = zone
            
            # Add margin to avoid spawning too close to walls
            x_min += self.spawn_margin
            x_max -= self.spawn_margin
            y_min += self.spawn_margin
            y_max -= self.spawn_margin
            
            # Ensure min < max after margin
            if x_min >= x_max or y_min >= y_max:
                continue
            
            # Generate random position within the zone
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            z = 0.01
            
            # Validate position is within house
            if not self._is_valid_position(x, y):
                continue
            
            # Generate random yaw (orientation) in [0, 2*pi]
            yaw = random.uniform(0, 2 * math.pi)
            
            room = self._get_room_name(x, y)
            self.get_logger().debug(f"Spawn pose generated in {room}: ({x:.2f}, {y:.2f})")
            
            return x, y, z, yaw
        
        # Fallback to center of first safe zone
        self.get_logger().warn("Could not find valid spawn pose after max attempts, using fallback")
        zone = self.safe_zones[0]
        x_min, x_max, y_min, y_max, zone_name = zone
        x = (x_min + x_max) / 2
        y = (y_min + y_max) / 2
        return x, y, 0.01, 0.0
    
    def generate_random_goal(self):
        """Generate a random goal position within safe zones of the house."""
        max_attempts = 50
        for attempt in range(max_attempts):
            # Randomly select a safe zone
            zone = random.choice(self.safe_zones)
            x_min, x_max, y_min, y_max, zone_name = zone
            
            # Add margin to avoid goals too close to walls
            x_min += self.goal_margin
            x_max -= self.goal_margin
            y_min += self.goal_margin
            y_max -= self.goal_margin
            
            # Ensure min < max after margin
            if x_min >= x_max or y_min >= y_max:
                continue
            
            # Generate random position within the zone
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            
            # Validate position is within house
            if not self._is_valid_position(x, y):
                continue
            
            room = self._get_room_name(x, y)
            self.get_logger().debug(f"Goal position generated in {room}: ({x:.2f}, {y:.2f})")
            
            return x, y
        
        # Fallback to center of first safe zone
        self.get_logger().warn("Could not find valid goal pose after max attempts, using fallback")
        zone = self.safe_zones[0]
        x_min, x_max, y_min, y_max, zone_name = zone
        x = (x_min + x_max) / 2
        y = (y_min + y_max) / 2
        return x, y
        
    def reset(self):
        # Stop robot
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        time.sleep(0.1)
        
        # Generate random spawn pose within house
        sx, sy, sz, syaw = self.generate_random_spawn_pose()
        
        # Respawn robot at random pose
        success = self.respawn_robot(sx, sy, syaw)
        if not success:
            self.get_logger().warn('Respawn failed; robot may stay at previous location')
        
        # Generate random goal position within house
        self.current_goal = self.generate_random_goal()
        gx, gy = self.current_goal
        self.get_logger().info(f"New goal set: ({gx:.2f}, {gy:.2f})")
        
        self.step_count = 0
        self.last_action = np.array([0.0, 0.0])
        self.last_collision_time = None  # Reset collision cooldown
        
        # Wait longer for sensor data to stabilize after respawn
        time.sleep(0.8)
        
        # Clear old sensor data
        with self.lock:
            self.scan_data = None
            self.imu_data = None
            self.odom_data = None
        
        lidar, state, dist, heading_error = self.get_state()
        self.prev_goal_dist = dist
        self.prev_heading_error = heading_error
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
        
        # Check if robot is flipped and fix it
        is_flipped, roll, pitch = self._is_robot_flipped()
        if is_flipped:
            self.get_logger().warn(f"Robot flipped detected! Roll: {math.degrees(roll):.1f}°, Pitch: {math.degrees(pitch):.1f}°. Resetting position...")
            # Stop the robot
            twist = Twist()
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.1)
            
            # Get current position
            with self.lock:
                px = self.odom_data.pose.pose.position.x
                py = self.odom_data.pose.pose.position.y
                q = self.odom_data.pose.pose.orientation
                _, _, yaw = self._quaternion_to_euler(q)
            
            # Respawn at current position with correct orientation (no flip)
            self.respawn_robot(px, py, yaw)
            time.sleep(0.3)
        
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
            
        # 2. Collision Penalty (do not terminate, just large penalty)
        # With 2-second cooldown to avoid repeated penalties
        min_dist = np.min(lidar * 3.5)
        if min_dist < self.COLLISION_DIST:
            current_time = time.time()
            # Check if enough time has passed since last collision
            if self.last_collision_time is None or (current_time - self.last_collision_time) >= self.COLLISION_COOLDOWN:
                reward -= self.COLLISION_PENALTY
                self.last_collision_time = current_time
                print(f"Collision! penalty={self.COLLISION_PENALTY}")
            
        # 3. Approach Reward: encourage progress toward goal using distance delta
        if not done:
            if self.prev_goal_dist is not None:
                delta = self.prev_goal_dist - dist  # positive if getting closer
                reward += delta * self.APPROACH_GAIN
            self.prev_goal_dist = dist

        # 3.1 Heading Alignment Reward: reward reduction in absolute heading error
        if not done and self.prev_heading_error is not None:
            heading_delta = abs(self.prev_heading_error) - abs(heading_error)
            reward += heading_delta * self.HEADING_ALIGN_GAIN
        self.prev_heading_error = heading_error

        # 3.2 Anti-Spin Penalty: discourage turning in place
        if v < self.SPIN_V_THRESHOLD and abs(w) > self.SPIN_W_THRESHOLD:
            reward -= self.SPIN_PENALTY

        # 3.3 Forward Bias when far from goal
        if not done and dist > (self.GOAL_DIST_THRESHOLD + 0.2):
            reward += self.FORWARD_BIAS_GAIN * (v / self.MAX_V)
        
        # 4. Smoothness/Time Penalty
        reward -= 0.01 # Time step
        
        self.step_count += 1
        if self.step_count >= self.MAX_STEPS:
            done = True
            
        return lidar, state, reward, done, {}

