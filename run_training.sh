#!/bin/bash
# Source ROS 2 (Adjust if your installation is different, e.g. foxy, galactic, humble)
source /opt/ros/humble/setup.bash

# Set Python Path to include current directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the training script
python3 rl_nav/train_gazebo.py
