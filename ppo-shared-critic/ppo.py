import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from utils import RolloutBuffer
from actor_critic import TaskActorCritic
from critic import Critic


class PPO:
    def __init__(self, tasks, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, shared_critic_alpha, device):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.shared_critic_alpha = shared_critic_alpha
        
        self.buffers = {}
        self.policies = {}
        self.policies_old = {}
        self.optimizers = {}
        self.tasks = tasks
        for task in range(0, tasks):
            self.buffers[task] = RolloutBuffer()
            self.policies[task] = TaskActorCritic(state_dim, action_dim).to(self.device)
            self.policies_old[task] = TaskActorCritic(state_dim, action_dim).to(self.device)
            self.policies_old[task].load_state_dict(self.policies[task].state_dict())

            self.optimizers[task] = torch.optim.Adam(
                [
                    {'params': self.policies[task].actor.parameters(), 'lr': lr_actor},
                    {'params': self.policies[task].critic.parameters(), 'lr': lr_critic}
                ]
            )
        
        self.shared_critic = Critic(state_dim)
        
        self.shared_critic_optimizer = torch.optim.Adam(self.shared_critic.parameters(), lr = lr_critic)

    def select_action(self, state, task):
        
        state = state.reshape(1, *state.shape)
        with torch.no_grad():
            # state = torch.FloatTensor(state).to(self.device).permute([0, 2, 1])
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob = self.policies_old[task].act(state)
        
        self.buffers[task].states.append(state)
        self.buffers[task].actions.append(action)
        self.buffers[task].logprobs.append(action_logprob)

        return action.cpu().numpy()


    def update(self):

        # Monte Carlo estimate of returns
        rewards = {}
        old_states = {}
        old_actions = {}
        old_logprobs = {}
        for task in range(self.tasks):
            rewards[task] = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(self.buffers[task].rewards), reversed(self.buffers[task].is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards[task].insert(0, discounted_reward)
            
            
            # Normalizing the rewards
            rewards[task] = torch.tensor(rewards[task], dtype=torch.float32).to(self.device)
            rewards[task] = (rewards[task] - rewards[task].mean()) / (rewards[task].std() + 1e-7)

            # convert list to tensor
            old_states[task] = torch.squeeze(torch.stack(self.buffers[task].states, dim=0)).detach().to(self.device)
            old_actions[task] = torch.squeeze(torch.stack(self.buffers[task].actions, dim=0)).detach().to(self.device)
            old_logprobs[task] = torch.squeeze(torch.stack(self.buffers[task].logprobs, dim=0)).detach().to(self.device)

        for _ in tqdm(range(self.K_epochs)):
            loss_shared = 0
            losses = 0
            for task in range(self.tasks):
                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policies[task].evaluate(old_states[task], old_actions[task])

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)
                
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs[task].detach())

                # Finding Surrogate Loss
                advantages = rewards[task] - state_values.detach()  

                state_values_shared = self.shared_critic(old_states[task])
                shared_advantages = rewards[task] - state_values_shared.detach()
                final_advantages = (1 - self.shared_critic_alpha)*advantages + self.shared_critic_alpha*shared_advantages

                surr1 = ratios * final_advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * final_advantages

                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + 0.5*F.mse_loss(state_values, rewards[task]) - 0.01*dist_entropy

                loss_shared += F.mse_loss(state_values_shared, rewards[task]) 
            
            # take gradient step
                self.optimizers[task].zero_grad()
                loss.mean().backward()

                self.optimizers[task].step()
                

                losses += loss.mean().item()
            
            self.shared_critic_optimizer.zero_grad()
            loss_shared.backward()
            self.shared_critic_optimizer.step()
        
        print("Loss:", losses + loss_shared.item())
            
        # Copy new weights into old policy
        for task in range(self.tasks):
            self.policies_old[task].load_state_dict(self.policies[task].state_dict())

            # clear buffer
            self.buffers[task].clear()
        
      
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


