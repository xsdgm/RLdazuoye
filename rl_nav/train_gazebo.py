import rclpy
import threading
import numpy as np
import torch
import os
import json
import argparse
from datetime import datetime
from rl_nav.env_gazebo import GazeboEnv
from rl_nav.ppo_agent import PPOAgent

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def main():
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Train a PPO agent for TurtleBot navigation.')
    parser.add_argument('--load-model', type=str, default=None, help='Path to the pre-trained model to load.')
    parser.add_argument('--start-episode', type=int, default=1, help='Episode number to start training from.')
    parser.add_argument('--load-metrics', type=str, default=None, help='Path to the metrics log file to continue tracking from.')
    args = parser.parse_args()

    # Initialize ROS 2
    rclpy.init()
    
    # Create Environment Node
    env = GazeboEnv(
        goal_tolerance=0.3, 
        collision_penalty=50.0,
        start_episode=args.start_episode
    )
    
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
    agent = PPOAgent(state_dim=7, lidar_dim=360, action_dim=2, device=device, entropy_coef=0.01)
    
    # Load a pre-trained model if specified
    if args.load_model:
        try:
            agent.policy.load_state_dict(torch.load(args.load_model, map_location=device))
            print(f"Successfully loaded model from {args.load_model}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    
    # Training Parameters
    MAX_EPISODES = 1000
    MAX_STEPS_PER_EPISODE = 2000
    UPDATE_TIMESTEP = 1024  # Increased update frequency for faster learning
    
    # Reward tracking
    episode_rewards = []
    avg_rewards = []
    actor_losses = []
    critic_losses = []

    # Load metrics if specified
    if args.load_metrics:
        try:
            with open(args.load_metrics, 'r') as f:
                metrics = json.load(f)
                episode_rewards = metrics.get('episode_rewards', [])
                avg_rewards = metrics.get('avg_rewards', [])
                actor_losses = metrics.get('actor_losses', [])
                critic_losses = metrics.get('critic_losses', [])
                print(f"Successfully loaded metrics from {args.load_metrics}")
                if avg_rewards:
                    print(f"Resuming with last average reward of: {avg_rewards[-1]:.2f}")
        except Exception as e:
            print(f"Could not load metrics file: {e}")

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
        for episode in range(args.start_episode - 1, MAX_EPISODES):
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
                    
                    # Calculate next_value for bootstrapping if episode didn't end
                    if not done:
                        # Get the value of the next state for proper bootstrapping
                        _, _, next_value = agent.get_action(next_lidar, next_state)
                    else:
                        # Episode ended, next state has no value
                        next_value = 0.0
                    
                    actor_loss, critic_loss = agent.update(buffer, next_value=next_value)
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
                    json.dump(metrics, f, indent=2, cls=NumpyEncoder)
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
            json.dump(final_metrics, f, indent=2, cls=NumpyEncoder)
        print(f"Final metrics saved to {metrics_log_file}")
        
        # Cleanup
        env.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
