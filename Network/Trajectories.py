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
    def __init__(self,ep_len:int, n_tiles:int,bsize:int, seq_len:int, horizon:int, gamma, gae_lambda, device) -> None:
        self.ep_len = horizon
        self.device = device
        self.bsize = bsize
        self.horizon = horizon
        self.seq_len = seq_len
        self.n_tiles = n_tiles
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def push(
            self,
            state,
            action,
            policy,
            value,
            next_value,
            reward,
            final
            ):

        self.state_buf[self.ptr] = state
        self.policy_buf[self.ptr] = policy
        self.value_buf[self.ptr] = value
        self.next_value_buf[self.ptr] = next_value
        self.rew_buf[self.ptr] = reward
        self.act_buf[self.ptr+1] = action
        self.final_buf[self.ptr] = final

        self.ptr += 1

        if self.ptr % self.horizon == 0:
            self.compute_gae_rtg(self.gamma,self.gae_lambda)


    def get_lastk(self,step:int,k:int = None):
        #FIXME: add padding
        if k is None:
            k = self.seq_len

        if self.ptr >= k:
            bot = self.ptr-k
            top = self.ptr
        
        else:
            bot = 0
            top = k

        states = self.state_buf[self.ptr]
        actions = self.act_buf.unsqueeze(-1)[bot:top]
        returns_to_go = self.rtg_buf.unsqueeze(-1)[bot:top]
        return {
            'actions':actions.to(UNIT),
            'states':states.to(UNIT),
            'returns_to_go':returns_to_go.to(UNIT),
            'timesteps':torch.arange(k+1,device=states.device) + step
        }
    
        
    def compute_gae_rtg(self,  gamma, gae_lambda): #FIXME:MASKS?

        if (self.ptr) % self.horizon != 0:
            raise BufferError("Calculating GAE at wrong time")
        
        if self.ptr >= self.horizon:
            bot = self.ptr-self.horizon
            top = self.ptr
        
        else:
            raise IndexError

        rewards = self.rew_buf[bot:top]
        values = self.value_buf[bot:top]
        next_values = self.next_value_buf[bot:top]
        finals = self.final_buf[bot:top]

        td_errors = rewards + gamma * next_values * (1 - finals) - values
        gae = 0
        rtgs = 0
        advantages = torch.zeros_like(td_errors)

        for t in reversed(range(len(td_errors))):
            gae = td_errors[t] + gamma * gae_lambda * (1 - finals[t]) * gae
            advantages[t] = gae


        returns_to_go = torch.zeros_like(rewards)
        return_to_go = 0
        for t in reversed(range(len(rewards))):
            return_to_go = rewards[t] + gamma * (1 - finals[t]) * return_to_go
            returns_to_go[t] = return_to_go 

        self.adv_buf[bot:top] = advantages
        self.rtg_buf[bot:top] = returns_to_go


    def reset(self):
        self.state_buf = torch.zeros((self.horizon,self.bsize,self.bsize,4*COLOR_ENCODING_SIZE),device=self.device).to(UNIT)
        self.act_buf = torch.zeros((self.horizon+1),dtype=int,device=self.device).to(UNIT) - 1
        self.act_buf[0] = -2 #BOS
        self.rtg_buf = torch.zeros((self.horizon+1),device=self.device).to(UNIT) - 20
        self.policy_buf = torch.zeros((self.horizon,self.n_tiles),device=self.device).to(UNIT)
        self.value_buf = torch.zeros((self.horizon),device=self.device).to(UNIT)
        self.next_value_buf = torch.zeros((self.horizon),device=self.device).to(UNIT)
        self.rew_buf = torch.zeros((self.horizon),device=self.device).to(UNIT)
        self.final_buf = torch.zeros((self.horizon),dtype=int,device=self.device).to(UNIT)
        self.adv_buf = torch.zeros((self.horizon),device=self.device).to(UNIT)
        self.ptr = 0

class BatchMemory:
    """
    Helpfull buffer for decision transformers, might be able to manage with only episode buffer for other network architectures
    """
    def __init__(self,n_tiles:int,bsize:int, seq_len:int,capacity:int,horizon:int,device='cpu') -> None:
        self.capacity = capacity
        self.horizon = horizon
        self.n_tiles = n_tiles
        self.seq_len = seq_len
        self.device = device
        self.bsize = bsize
        self.ptr = 0
        self.reset()


    def load(self,buff:EpisodeBuffer,step:int):

        # pad for the unfinished trajectories
        k = self.seq_len

        if buff.ptr >= k+1:
            bot = buff.ptr-1-k
            top = buff.ptr
        
        else:
            bot = 0
            top = k+1

        self.act_buf[self.ptr] = buff.act_buf.squeeze(-1)[bot:top]
        self.state_buf[self.ptr] = buff.state_buf[buff.ptr - 1]
        self.timestep_buf[self.ptr] = buff.ptr - 1
        self.policy_buf[self.ptr] = buff.policy_buf[buff.ptr - 1]
        self.value_buf[self.ptr] = buff.value_buf[buff.ptr - 1]
        self.ptr += 1

    def load_advantages_rtg(self,ep_buf:EpisodeBuffer): #FIXME:
        if ep_buf.ptr != ep_buf.ep_len:
            raise IndexError(ep_buf.ptr,ep_buf.ep_len)

        self.adv_buf[self.ptr - self.horizon:self.ptr] = ep_buf.adv_buf
        self.rtg_buf[self.ptr - self.horizon:self.ptr] = ep_buf.rtg_buf[-self.seq_len-1:]




    def reset(self):

        if self.ptr != self.capacity:
            print(Warning(f'Memory not full : {self.ptr}/{self.capacity}'))
        self.state_buf = torch.empty((self.capacity,self.bsize,self.bsize,4*COLOR_ENCODING_SIZE),device=self.device).to(UNIT)
        self.act_buf = torch.empty((self.capacity,self.seq_len+1),dtype=int,device=self.device)
        self.rtg_buf = torch.empty((self.capacity,self.seq_len+1),device=self.device).to(UNIT)
        self.policy_buf = torch.empty((self.capacity,self.n_tiles),device=self.device).to(UNIT)
        self.adv_buf = torch.zeros((self.capacity),device=self.device).to(UNIT)
        self.value_buf = torch.empty((self.capacity),device=self.device).to(UNIT)
        self.timestep_buf = torch.empty((self.capacity),dtype=int,device=self.device)
        self.ptr = 0

    def __getitem__(self,key):
        return getattr(self,key)[self.ptr-self.horizon:self.ptr]



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




