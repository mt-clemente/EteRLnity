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
    def __init__(self,ep_len:int, bsize:int, seq_len:int, horizon:int, gamma, gae_lambda, first_corner, init_state, device) -> None:
        self.ep_len = ep_len
        self.device = device
        self.bsize = bsize
        self.horizon = horizon
        self.seq_len = seq_len
        self.first_corner = first_corner
        self.init_state = init_state
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def push(
            self,
            state,
            action,
            tile,
            policy,
            tile_mask,
            value,
            reward,
            ep_step,
            final
            ):

        self.state_buf[self.ptr+1] = state
        self.policy_buf[self.ptr] = policy
        self.mask_buf[self.ptr] = tile_mask
        self.value_buf[self.ptr] = value
        self.rew_buf[self.ptr] = reward
        self.timestep_buf[self.ptr] = ep_step
        self.final_buf[self.ptr] = final
        self.act_buf[self.ptr] = action
        self.tile_buf[self.ptr + 1] = tile

        if self.ptr != 0:
            self.next_value_buf[self.ptr - 1] = value

        if self.ptr + 1 < self.seq_len:
            bot = 0
            top = self.seq_len
        else:
            bot = self.ptr+2-self.seq_len
            top = self.ptr+2

        self.tile_seq[self.ptr+1] = self.tile_buf[bot:top]

        self.ptr += 1


        if (self.ptr-1) % self.horizon == 0 and self.ptr != 1 or self.ptr == self.ep_len:
            self.compute_gae_rtg(self.gamma,self.gae_lambda)

            if self.rtg_buf.max() > 10000 or self.adv_buf.max() > 10000:
                raise OSError(self.adv_buf)

    def compute_gae_rtg(self,  gamma, gae_lambda): #FIXME:MASKS?

        if (self.ptr) % self.horizon != 0:
            # raise BufferError("Calculating GAE at wrong time")
            pass

        if self.ptr  >= self.horizon:
            bot = self.ptr-self.horizon-1
            top = self.ptr-1
        else:
            raise IndexError
        
        if self.ptr == self.ep_len:
            bot += 1
            top += 1
        
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
        return_to_go = next_values[-1]
        returns_to_go[-1] = return_to_go
        for t in reversed(range(len(rewards)-1)):
            return_to_go = rewards[t] + gamma * (1 - finals[t]) * return_to_go
            returns_to_go[t] = return_to_go 

        self.adv_buf[bot:top] = advantages
        self.rtg_buf[bot:top] = returns_to_go


    def reset(self):
        self.state_buf = torch.zeros((self.ep_len+1,self.bsize,self.bsize,4*COLOR_ENCODING_SIZE),device=self.device).to(UNIT) -20
        self.state_buf[0] = self.init_state
        self.act_buf = torch.empty((self.ep_len),dtype=int,device=self.device) - 1
        self.tile_buf = torch.zeros((self.ep_len+1,4),dtype=int,device=self.device).to(UNIT) - 1
        self.tile_buf[0] = self.first_corner #BOS
        self.tile_seq = torch.zeros((self.ep_len+1,self.seq_len,4),dtype=int,device=self.device).to(UNIT) - 1
        self.tile_seq[:,0,:] = self.first_corner #BOS
        self.rtg_buf = torch.zeros((self.ep_len),device=self.device).to(UNIT)
        self.policy_buf = torch.empty((self.ep_len,self.ep_len),device=self.device).to(UNIT)
        self.mask_buf = torch.empty((self.ep_len,self.ep_len),dtype=bool,device=self.device)
        self.value_buf = torch.empty((self.ep_len),device=self.device).to(UNIT)
        self.next_value_buf = torch.zeros((self.ep_len),device=self.device).to(UNIT)
        self.rew_buf = torch.empty((self.ep_len),device=self.device).to(UNIT)
        self.final_buf = torch.empty((self.ep_len),dtype=int,device=self.device)
        self.adv_buf = torch.zeros((self.ep_len),device=self.device).to(UNIT) - 6666
        self.timestep_buf = torch.empty((self.ep_len),device=self.device,dtype=int)
        self.ptr = 0

    def __getitem__(self,key):
        
        if self.ptr == self.ep_len:
            return getattr(self,key)[self.ptr-self.horizon:self.ptr]
        else:
            return getattr(self,key)[self.ptr-1-self.horizon:self.ptr-1]



class AdvantageBuffer():

    def __init__(self) -> None:

        self.state = None
        self.action = None
        self.policy = None
        self.value = None
        self.tile = None
        self.reward = None
        self.reward_to_go = None
        self.tile_mask = None



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




