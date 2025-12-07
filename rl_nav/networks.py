import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FusionNetwork(nn.Module):
    def __init__(self, state_dim=7, lidar_dim=360, action_dim=2):
        super(FusionNetwork, self).__init__()
        
        # LiDAR Stream (1D CNN)
        # Input: (Batch, 1, 360)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        
        # Calculate CNN output size
        # L_out = floor((L_in - kernel_size)/stride + 1)
        # 360 -> (360-5)/2 + 1 = 178
        # 178 -> (178-3)/2 + 1 = 88
        # 88 -> (88-3)/2 + 1 = 43
        self.cnn_out_dim = 64 * 43
        
        self.lidar_fc = nn.Linear(self.cnn_out_dim, 128)
        
        # State Stream (MLP)
        self.state_fc1 = nn.Linear(state_dim, 128)
        self.state_fc2 = nn.Linear(128, 64)
        
        # Fusion
        self.fusion_dim = 128 + 64
        
        # Actor Head
        self.actor_fc1 = nn.Linear(self.fusion_dim, 256)
        self.actor_mean = nn.Linear(256, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic Head
        self.critic_fc1 = nn.Linear(self.fusion_dim, 256)
        self.critic_value = nn.Linear(256, 1)
        
    def forward(self, lidar, state):
        # LiDAR processing
        # lidar shape: (Batch, 360) -> (Batch, 1, 360)
        if lidar.dim() == 2:
            lidar = lidar.unsqueeze(1)
            
        x_l = F.relu(self.conv1(lidar))
        x_l = F.relu(self.conv2(x_l))
        x_l = F.relu(self.conv3(x_l))
        x_l = x_l.view(x_l.size(0), -1)
        x_l = F.relu(self.lidar_fc(x_l))
        
        # State processing
        x_s = F.relu(self.state_fc1(state))
        x_s = F.relu(self.state_fc2(x_s))
        
        # Fusion
        x = torch.cat([x_l, x_s], dim=1)
        
        # Actor
        a = F.relu(self.actor_fc1(x))
        mean = torch.tanh(self.actor_mean(a))
        
        # Critic
        c = F.relu(self.critic_fc1(x))
        value = self.critic_value(c)
        
        return mean, self.actor_logstd, value

    def get_action(self, lidar, state, deterministic=False):
        mean, log_std, value = self.forward(lidar, state)
        std = log_std.exp()
        
        dist = torch.distributions.Normal(mean, std)
        
        if deterministic:
            action = mean
        else:
            action = dist.sample()
            
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob, value
