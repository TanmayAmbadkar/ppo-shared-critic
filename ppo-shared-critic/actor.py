import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, continuous=True):

        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim), 
            nn.Tanh()
        )

        if continuous:
            self.actor_std = nn.Parameter(torch.randn(size = (action_dim, )))
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        
    def forward(self, x):
        actor_mean = self.actor(x)
        actor_std = None
        if self.continuous:
            actor_std = self.actor_std.exp()
            dist = MultivariateNormal(loc = actor_mean, scale_tril = torch.diag(actor_std))
        
        else:
            dist = Categorical(logits = actor_mean)


        return dist