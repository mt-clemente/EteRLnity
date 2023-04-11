from datetime import datetime
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from param import *
from hashlib import sha256
from torch.utils.data import TensorDataset,DataLoader
import torch.nn.init as init




class EpisodeBuffer:
    def __init__(self,ep_len:int, n_tiles:int,bsize:int,device) -> None:
        self.ep_len = ep_len
        self.device = device
        self.state_buf = torch.zeros((ep_len,bsize,bsize,4*COLOR_ENCODING_SIZE),device=self.device)
        self.act_buf = torch.zeros((ep_len),dtype=int,device=self.device) - 2
        self.policy_buf = torch.zeros((ep_len,n_tiles),device=self.device)
        self.value_buf = torch.zeros((ep_len),device=self.device)
        self.next_value_buf = torch.zeros((ep_len),device=self.device)
        self.rew_buf = torch.zeros((ep_len),device=self.device)
        self.rtg_buf = torch.zeros((ep_len+1),device=self.device)
        self.final_buf = torch.zeros((ep_len),dtype=int,device=self.device)
        self.adv_buf = torch.zeros((ep_len+1),device=self.device)
        self.ptr = 0

    def push(
            self,
            state,
            action,
            policy,
            value,
            next_value,
            reward,
            reward_to_go,
            final
            ):


        self.state_buf[self.ptr] = state
        self.act_buf[self.ptr] = action
        self.policy_buf[self.ptr] = policy
        self.value_buf[self.ptr] = value
        self.next_value_buf[self.ptr] = next_value
        self.rew_buf[self.ptr] = reward
        self.rtg_buf[self.ptr+1] = reward_to_go
        self.final_buf[self.ptr] = final

        self.ptr += 1
        


    def get_lastk(self,step:int,k:int = None):#FIXME: add padding
        if k is None:
            k = self.ptr

        states = self.state_buf
        actions = self.act_buf.unsqueeze(-1)
        returns_to_go = self.rtg_buf.unsqueeze(-1)
        return {
            'actions':actions.to(UNIT),
            'states':states.to(UNIT),
            'returns_to_go':returns_to_go.to(UNIT),
            'timesteps':torch.arange(k+1,device=states.device)
        }
        
    def compute_gae(self,  gamma, gae_lambda):

        if self.ptr != self.ep_len:
            print(Warning("Computing advantages on an unfinished episode"))

        td_errors = self.rew_buf + gamma * self.next_value_buf * (1 - self.final_buf) - self.value_buf
        gae = 0
        advantages = torch.zeros_like(td_errors)
        for t in reversed(range(len(td_errors))):
            gae = td_errors[t] + gamma * gae_lambda * (1 - self.final_buf[t]) * gae
            advantages[t] = gae

        self.adv_buf = advantages


    def reset(self):
        self.ptr = 0

class BatchMemory:
    def __init__(self,n_tiles:int,bsize:int,capacity:int=MINIBATCH_SIZE, ep_length:int=256,device='cpu') -> None:
        self.capacity = capacity * (ep_length)
        self.ep_length = ep_length
        self.n_tiles = n_tiles
        self.device = device
        self.bsize = bsize
        self.ptr = 0
        self.reset()


    def load(self,buff:EpisodeBuffer,step:int):
        b_dict = buff.get_lastk(step)

        # pad for the unfinished trajectories
        self.act_buf[self.ptr] = b_dict['actions'].squeeze(-1)
        self.rtg_buf[self.ptr] = b_dict['returns_to_go'].squeeze(-1)
        self.state_buf[self.ptr] = buff.state_buf[buff.ptr - 1]
        self.timestep_buf[self.ptr] = buff.ptr - 1
        self.policy_buf[self.ptr] = buff.policy_buf[buff.ptr - 1]
        self.value_buf[self.ptr] = buff.value_buf[buff.ptr - 1]

        self.ptr += 1

    def load_advantages(self,advantages):
        self.adv_buf[self.ptr-advantages.size()[0]:self.ptr] = advantages


    def reset(self):

        if self.ptr != self.capacity:
            print(Warning(f'Memory not full : {self.ptr}/{self.capacity}'))
        self.state_buf = torch.empty((self.capacity,self.bsize,self.bsize,4*COLOR_ENCODING_SIZE),device=self.device)
        self.act_buf = torch.empty((self.capacity,self.ep_length),dtype=int,device=self.device)
        self.policy_buf = torch.empty((self.capacity,self.n_tiles),device=self.device)
        self.rtg_buf = torch.empty((self.capacity,self.ep_length+1),device=self.device)
        self.adv_buf = torch.zeros((self.capacity),device=self.device)
        self.value_buf = torch.empty((self.capacity),device=self.device)
        self.timestep_buf = torch.empty((self.capacity),dtype=int,device=self.device)
        self.ptr = 0

    def __getitem__(self,key):
        return getattr(self,key)[:self.ptr]



class AdvantageBuffer():

    def __init__(self) -> None:

        self.state = None
        self.action = None
        self.policy = None
        self.value = None
        self.reward = None
        self.reward_to_go = None



class StoppingCriterion():

    def __init__(self,threshold) -> None:
        """
        Stopping critirion for a trajectory.
        Internal counter is updated each step :
         * \+ 1 if degrading move
         * \- 0.5 if the new score is better than the previous one
         * Reset to 0 if new best score
        """
        self.counter = 0
        self.prev_score = 0
        self.eos = False
        self.threshold = threshold


    def update(self,score,best_score):

        if score > best_score:
            self.counter = 0

        elif score > self.prev_score:
            self.counter -= 0.5

        else:
            self.counter += 1

        self.prev_score = score

        if self.counter > self.threshold:
            self.eos = True

    def is_stale(self):
        return self.eos
        
    
    def reset(self):
        self.counter = 0
        self.prev_score = 0
        self.eos = False




