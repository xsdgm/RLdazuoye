import rclpy
import threading
import numpy as np
import torch
import os
from rl_nav.env_gazebo import GazeboEnv
from rl_nav.ppo_agent import PPOAgent

def main():
    # Initialize ROS 2
    rclpy.init()
    
    # Create Environment Node
    env = GazeboEnv()
    
    # Spin in a separate thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(env,), daemon=True)
    spin_thread.start()
    
    # Initialize PPO Agent
    agent = PPOAgent(state_dim=7, lidar_dim=360, action_dim=2)
    
    # Training Parameters
    MAX_EPISODES = 1000
    MAX_STEPS_PER_EPISODE = 500
    UPDATE_TIMESTEP = 2000
    
    buffer = {
        'lidars': [],
        'states': [],
        'actions': [],
        'rewards': [],
        'dones': [],
        'log_probs': [],
        'values': []
    }
    
    timestep_count = 0
    
    print("Starting Training...")
    
    try:
        for episode in range(MAX_EPISODES):
            lidar, state = env.reset()
            episode_reward = 0
            
            for step in range(MAX_STEPS_PER_EPISODE):
                # Get Action
                action, log_prob, value = agent.get_action(lidar, state)
                
                # Step Environment
                next_lidar, next_state, reward, done, _ = env.step(action)
                
                # Store in buffer
                buffer['lidars'].append(lidar)
                buffer['states'].append(state)
                buffer['actions'].append(action)
                buffer['rewards'].append(reward)
                buffer['dones'].append(done)
                buffer['log_probs'].append(log_prob)
                buffer['values'].append(value)
                
                lidar = next_lidar
                state = next_state
                episode_reward += reward
                timestep_count += 1
                
                # Update PPO
                if timestep_count % UPDATE_TIMESTEP == 0:
                    print(f"Updating PPO at timestep {timestep_count}...")
                    actor_loss, critic_loss = agent.update(buffer)
                    print(f"Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
                    
                    # Clear buffer
                    buffer = {
                        'lidars': [],
                        'states': [],
                        'actions': [],
                        'rewards': [],
                        'dones': [],
                        'log_probs': [],
                        'values': []
                    }
                
                if done:
                    break
            
            print(f"Episode {episode+1}: Reward = {episode_reward:.2f}")
            
            # Save Model occasionally
            if (episode + 1) % 10 == 0:
                torch.save(agent.policy.state_dict(), f"ppo_model_{episode+1}.pth")
                
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        # Cleanup
        env.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
