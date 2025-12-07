#!/bin/bash
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=burger

# Clean stale FastRTPS SHM segments to avoid port lock errors
rm -f /dev/shm/fastrtps_* 2>/dev/null
# Kill previous instances
pkill -9 gzserver
pkill -9 gzclient
pkill -f robot_state_publisher
sleep 2

# Run custom launch file
ros2 launch rl_nav/custom_launch.py
