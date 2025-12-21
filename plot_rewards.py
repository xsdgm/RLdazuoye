#!/usr/bin/env python3
"""
Plot training rewards and metrics from log files.
Usage: python plot_rewards.py [log_file]
"""

import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_rewards_from_json(json_file):
    """Plot rewards from JSON metrics file."""
    with open(json_file, 'r') as f:
        metrics = json.load(f)
    
    episode_rewards = metrics.get('episode_rewards', [])
    avg_rewards = metrics.get('avg_rewards', [])
    actor_losses = metrics.get('actor_losses', [])
    critic_losses = metrics.get('critic_losses', [])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16)
    
    # Plot episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.6, label='Episode Reward')
    axes[0, 0].plot(avg_rewards, linewidth=2, label='Average Reward (100 ep)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot average rewards only
    axes[0, 1].plot(avg_rewards, linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].set_title('Moving Average Reward (100 episodes)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot actor losses
    if actor_losses:
        axes[1, 0].plot(actor_losses, color='red', alpha=0.7)
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Actor Loss')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot critic losses
    if critic_losses:
        axes[1, 1].plot(critic_losses, color='green', alpha=0.7)
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Critic Loss')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = json_file.replace('.json', '_plot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    plt.show()

def plot_rewards_from_txt(txt_file):
    """Plot rewards from TXT log file."""
    episodes = []
    rewards = []
    avg_rewards = []
    
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3:
                episodes.append(int(parts[0]))
                rewards.append(float(parts[1]))
                avg_rewards.append(float(parts[2]))
    
    # Create figure
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, rewards, alpha=0.6, label='Episode Reward')
    plt.plot(episodes, avg_rewards, linewidth=2, label='Average Reward (100 ep)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    output_file = txt_file.replace('.txt', '_plot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    plt.show()

def find_latest_log():
    """Find the latest metrics JSON file in logs directory."""
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        return None
    
    json_files = [f for f in os.listdir(logs_dir) if f.startswith('metrics_') and f.endswith('.json')]
    if not json_files:
        return None
    
    json_files.sort(reverse=True)
    return os.path.join(logs_dir, json_files[0])

if __name__ == '__main__':
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        # Try to find latest log file
        log_file = find_latest_log()
        if log_file is None:
            print("No log file found in 'logs' directory.")
            print("Usage: python plot_rewards.py [log_file]")
            sys.exit(1)
        print(f"Using latest log file: {log_file}")
    
    if not os.path.exists(log_file):
        print(f"Error: File {log_file} not found.")
        sys.exit(1)
    
    # Determine file type and plot accordingly
    if log_file.endswith('.json'):
        plot_rewards_from_json(log_file)
    elif log_file.endswith('.txt'):
        plot_rewards_from_txt(log_file)
    else:
        print("Error: Unknown file format. Use .json or .txt file.")
        sys.exit(1)
