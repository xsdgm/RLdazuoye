import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .networks import FusionNetwork

class PPOAgent:
    def __init__(self, state_dim=7, lidar_dim=360, action_dim=2, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2, entropy_coef=0.01, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        
        self.policy = FusionNetwork(state_dim, lidar_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()
        
    def get_action(self, lidar, state, deterministic=False):
        lidar = torch.FloatTensor(lidar).to(self.device).unsqueeze(0)
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(lidar, state, deterministic)
            
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]
    
    def update(self, buffer, batch_size=64, epochs=10):
        # Unpack buffer
        lidars = torch.FloatTensor(np.array(buffer['lidars'])).to(self.device)
        states = torch.FloatTensor(np.array(buffer['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(buffer['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(buffer['log_probs'])).to(self.device)
        rewards = buffer['rewards']
        dones = buffer['dones']
        values = buffer['values']
        
        # Calculate Advantages and Returns (GAE)
        returns = []
        advantages = []
        gae = 0
        
        # Append 0 for the last value if not provided (assuming done)
        values = values + [0]
        
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i+1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
            
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO Update Loop
        dataset_size = len(rewards)
        indices = np.arange(dataset_size)
        
        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]
                
                b_lidars = lidars[idx]
                b_states = states[idx]
                b_actions = actions[idx]
                b_old_log_probs = old_log_probs[idx]
                b_advantages = advantages[idx]
                b_returns = returns[idx]
                
                # Forward pass
                mean, log_std, value = self.policy(b_lidars, b_states)
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                
                log_prob = dist.log_prob(b_actions).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # Ratio
                ratio = torch.exp(log_prob - b_old_log_probs)
                
                # Surrogate Loss
                surr1 = ratio * b_advantages.unsqueeze(1)
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * b_advantages.unsqueeze(1)
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic Loss
                critic_loss = self.mse_loss(value, b_returns.unsqueeze(1))
                
                # Total Loss
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
        return actor_loss.item(), critic_loss.item()
