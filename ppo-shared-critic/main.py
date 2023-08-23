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
from ppo import PPO
from utils import sample_trajectory
import pickle

env_name = "Eplus-office-mixed-continuous-v1"
max_ep_len = 10000                    # max timesteps in one episode

max_eps = 50

K_epochs = 40               
eps_clip = 0.2              
gamma = 0.5                

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
ppo_agent = PPO(state_dim = state_dim, action_dim = action_dim, lr_actor = lr_actor, lr_critic = lr_critic, gamma = gamma, K_epochs = K_epochs, eps_clip = eps_clip, shared_critic_alpha = 0.2, device = device, tasks = 3)
 
time_step = 0
i_episode = 0
rewards = []
tasks = 3

for episode in range(max_eps):
    for task in range(tasks):   
        reward = sample_trajectory(env = env, agent = ppo_agent, task=task, max_ep_len=max_ep_len)
        print(f"Episode: {episode}, task: {task}, reward: {reward}")

    ppo_agent.update()


pickle.dump(ppo_agent, open( "agent.pkl", "wb" ) )
