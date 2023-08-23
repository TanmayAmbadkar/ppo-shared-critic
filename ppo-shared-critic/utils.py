import numpy as np

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
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

def sample_trajectory(env, agent, task, max_ep_len):

    terminated = False
    current_ep_reward = 0
    done = False
    state, info = env.reset()
    # curr_state = [np.zeros((28, *state.shape))]*24
    # curr_state[info['init_hour']][-1] = state
    
    for _ in range(max_ep_len):
        
        # select action with policy
        try:

            action = agent.select_action(state, task)
        except:
            action = agent.select_action(state, task)
        # action = np.clip(action, a_min = -1, a_max=1)
        # action = env.action_space.sample()
        
        # action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action[0])
        
        # reward = info['reward_energy']*0.5 + info['reward_comfort']*0.5
        
        # saving reward and is_terminals
        if task == 0:
            reward = info["reward_energy"]
        elif task == 1:
            reward = info["reward_comfort"]
            
        agent.buffers[task].rewards.append(reward)
        agent.buffers[task].is_terminals.append(terminated)
        
        current_ep_reward += reward

        # curr_state[info['hour']][:-1] = curr_state[info['hour']][1:]
        # curr_state[info['hour']][-1] = state

        if terminated:
            break
    
    return current_ep_reward
