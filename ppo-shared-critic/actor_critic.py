import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
from actor import Actor
from critic import Critic

class TaskActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, continuous=True):
        super(TaskActorCritic, self).__init__()

        # actor
        self.actor = Actor(state_dim, action_dim, continuous)
        self.critic = Critic(state_dim)
        self.continuous = True

    def forward(self):
        raise NotImplementedError
    

    def act(self, state):
        
        dist = self.actor(state)

        action = dist.sample()
        action = torch.clip(action, -1, 1)
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    

    def evaluate(self, state, action):
        
       
        dist = self.actor(state)
        state_values = self.critic(state)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, state_values, dist_entropy