import torch
import torch.nn as nn

class Critic(nn.Module):

    def __init__(self, state_dim):

        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1), 
        )

        
        self.state_dim = state_dim
        
    def forward(self, x):
        value = self.actor(x)
        
        return value