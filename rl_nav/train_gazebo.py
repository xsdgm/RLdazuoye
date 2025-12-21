import rclpy
import threading
import numpy as np
import torch
import os
import json
from datetime import datetime
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
    
    # Determine device (GPU or CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Initialize PPO Agent with GPU support
    agent = PPOAgent(state_dim=7, lidar_dim=360, action_dim=2, device=device)
    
    # Training Parameters
    MAX_EPISODES = 1000
    MAX_STEPS_PER_EPISODE = 500
    UPDATE_TIMESTEP = 2000
    
    # Reward tracking
    episode_rewards = []
    avg_rewards = []
    actor_losses = []
    critic_losses = []
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    reward_log_file = f'logs/rewards_{timestamp}.txt'
    metrics_log_file = f'logs/metrics_{timestamp}.json'
    
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
                    
                    # Record losses
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
                    
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
            
            # Record episode reward
            episode_rewards.append(episode_reward)
            
            # Calculate moving average (last 100 episodes)
            window_size = min(100, len(episode_rewards))
            avg_reward = np.mean(episode_rewards[-window_size:])
            avg_rewards.append(avg_reward)
            
            print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Avg Reward (last {window_size}) = {avg_reward:.2f}")
            
            # Save reward to log file
            with open(reward_log_file, 'a') as f:
                f.write(f"{episode+1},{episode_reward:.4f},{avg_reward:.4f}\n")
            
            # Save Model occasionally
            if (episode + 1) % 10 == 0:
                torch.save(agent.policy.state_dict(), f"ppo_model_{episode+1}.pth")
                
                # Save comprehensive metrics
                metrics = {
                    'episode': episode + 1,
                    'episode_rewards': episode_rewards,
                    'avg_rewards': avg_rewards,
                    'actor_losses': actor_losses,
                    'critic_losses': critic_losses
                }
                with open(metrics_log_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                print(f"Model and metrics saved at episode {episode+1}")
                
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        # Save final metrics
        final_metrics = {
            'total_episodes': len(episode_rewards),
            'episode_rewards': episode_rewards,
            'avg_rewards': avg_rewards,
            'actor_losses': actor_losses,
            'critic_losses': critic_losses,
            'final_avg_reward': avg_rewards[-1] if avg_rewards else 0
        }
        with open(metrics_log_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        print(f"Final metrics saved to {metrics_log_file}")
        
        # Cleanup
        env.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
