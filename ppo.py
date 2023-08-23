import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from datetime import datetime
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import gymnasium as gym
import sinergym
from sinergym.utils.wrappers import LoggerWrapper
from sinergym.utils.rewards import ExpReward

class RolloutBuffer:
    def __init__(self, max_len = 200):
        self.actions = []
        self.states = []
        self.max_len = 200
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
    
    def __len__(self):
        return len(self.actions)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # actor
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim), 
        )
        self.actor_std = nn.Parameter(torch.zeros((action_dim,)), requires_grad=True)
        
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.state_dim = state_dim

    def forward(self):
        raise NotImplementedError
    

    def act(self, state):
        
        action_mean = self.actor_mean(state)       
        action_std = self.actor_std.exp()*torch.eye(action_mean.shape[1])
        dist = MultivariateNormal(loc = action_mean[0], covariance_matrix=action_std)

        action = dist.sample()
        action = torch.clip(action, -1, 1)
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    

    def evaluate(self, state, action):
        
        action_mean = self.actor_mean(state)   
        action_std = self.actor_std.exp()*torch.eye(action_mean.shape[1])
        
        dist = MultivariateNormal(loc = action_mean, covariance_matrix=action_std)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor_mean.parameters(), 'lr': lr_actor},
                        {'params': self.policy.actor_std, 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()


    def select_action(self, state):
        
        state = state.reshape(1, *state.shape)
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.cpu().numpy()


    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        for _ in tqdm(range(self.K_epochs)):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        print("Loss:", loss.mean().item())
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        
      
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))



#Main loop

env_name = "Acrobot-v1"
max_ep_len = 4040                    # max timesteps in one episode
max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps
update_timestep = 4040      # update policy every n timesteps


K_epochs = 40               
eps_clip = 0.2              
gamma = 0.99                

lr_actor = 0.0001       
lr_critic = 0.001       

random_seed = 0        

print("training environment name : " + env_name)

extra_params={'timesteps_per_hour' : 6,
              'runperiod' : (1,1,1997,28,1,1997)}

env = gym.make('Eplus-office-mixed-continuous-v1', config_params = extra_params)
env = LoggerWrapper(env)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

device = torch.device('cpu') 
ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device)
 
time_step = 0
i_episode = 0
rewards = []
max_eps = 50


while i_episode < max_eps:
    
    state, info = env.reset()
    current_ep_reward = 0
    done = False
    # curr_state = [np.zeros((28, *state.shape))]*24
    # curr_state[info['init_hour']][-1] = state
    
    for t in range(1, max_ep_len+1):
        
        # # select action with policy
        # try:

        #     action = ppo_agent.select_action(curr_state[info['hour']])
        # except:
        #     action = ppo_agent.select_action(curr_state[info['init_hour']])
        
        action = ppo_agent.select_action(state)
        # action = np.clip(action, a_min = -1, a_max=1)
        # action = env.action_space.sample()
        
        # action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        
        # reward = info['reward_energy']*0.5 + info['reward_comfort']*0.5
        
        # saving reward and is_terminals
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(done)
        
        current_ep_reward += reward
        time_step +=1

        # curr_state[info['hour']][:-1] = curr_state[info['hour']][1:]
        # curr_state[info['hour']][-1] = state

        # update PPO agent
        if time_step % update_timestep == 0:
            ppo_agent.update()

        if done:
            break
        
    
    print("Episode : {} \t Average Reward : {}".format(i_episode, current_ep_reward))
    i_episode += 1
    rewards.append(current_ep_reward)

    if i_episode == 500:
        break


env.close()
plt.plot(rewards)
plt.title("PPO Implementation for sinergym")
plt.savefig('rewards_cnn_7day_discretetime.png')

