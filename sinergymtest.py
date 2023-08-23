import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO, HerReplayBuffer
import sinergym
from sinergym.utils.wrappers import LoggerWrapper

extra_params={'timesteps_per_hour' : 6,
              'runperiod' : (1,1,1997,28,1,1997)}

env = gym.make('Eplus-office-mixed-continuous-v1', config_params = extra_params)
env = LoggerWrapper(env)


model = PPO(policy="MlpPolicy", env=env, verbose=1)

model.learn(total_timesteps=201600)

