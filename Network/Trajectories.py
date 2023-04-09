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
    def __init__(self,capacity:int, n_tiles:int,bsize:int,device) -> None:
        self.capacity = capacity
        self.ep_length = n_tiles
        self.device = device
        self.state_buf = torch.zeros((capacity,bsize,bsize,4*COLOR_ENCODING_SIZE),device=self.device)
        self.act_buf = torch.zeros((capacity),dtype=int,device=self.device)
        self.policy_buf = torch.zeros((capacity,n_tiles),device=self.device)
        self.value_buf = torch.zeros((capacity),device=self.device)
        self.next_value_buf = torch.zeros((capacity),device=self.device)
        self.rew_buf = torch.zeros((capacity),device=self.device)
        self.final_buf = torch.zeros((capacity),dtype=int,device=self.device)
        self.ptr = 0

    def push(
            self,
            state,
            action,
            policy,
            value,
            next_value,
            reward_to_go,
            final
            ):


        self.state_buf[self.ptr] = state
        self.act_buf[self.ptr] = action
        self.policy_buf[self.ptr] = policy
        self.value_buf[self.ptr] = value
        self.next_value_buf[self.ptr] = next_value
        self.rew_buf[self.ptr] = reward_to_go
        self.final_buf[self.ptr] = final

        self.ptr += 1
        if self.ptr == self.capacity:
            self.ptr = 0

    def get_lastk(self,step:int,k:int = None):#FIXME: add padding
        if k is None:
            k = self.ptr

        policy_SOS = torch.zeros_like(self.policy_buf[0]).unsqueeze(0) - 1
        states_SOS = torch.zeros_like(self.state_buf[0]).unsqueeze(0) - 1
        returns_to_go_SOS = torch.zeros_like(self.rew_buf[0]).unsqueeze(0) - 1
        if step == 0:
            return {
                'policies':policy_SOS.to(UNIT),
                'states':states_SOS.to(UNIT),
                'returns_to_go':returns_to_go_SOS.unsqueeze(0).to(UNIT),
                'timesteps':torch.tensor(0,device='cuda' if torch.cuda.is_available() else 'cpu')
            }

        policy = torch.vstack((policy_SOS ,self.policy_buf[:k]))
        states = torch.vstack((states_SOS ,self.state_buf[:k]))
        returns_to_go = torch.vstack((returns_to_go_SOS ,self.rew_buf[:k].unsqueeze(-1)))
        return {
            'policies':policy.to(UNIT),
            'states':states.to(UNIT),
            'returns_to_go':returns_to_go.to(UNIT),
            'timesteps':torch.arange(k+1,device='cuda' if torch.cuda.is_available() else 'cpu')
        }
        
    def reset(self):
        if self.ptr != 0:
            raise OSError(self.ptr)
        
        self.ptr = 0



class BatchMemory:
    def __init__(self,n_tiles:int,bsize:int,capacity:int=MINIBATCH_SIZE, ep_length:int=256,device='cpu') -> None:
        self.capacity = capacity
        self.ep_length = ep_length
        self.n_tiles = n_tiles
        self.device = device
        self.bsize = bsize
        self.ptr = 0
        self.reset()


    def load(self,buff:EpisodeBuffer,step:int):
        b_dict = buff.get_lastk(step)

        # pad for the unfinished trajectories
        if step < self.ep_length:
            self.state_buf[self.ptr] = torch.vstack((b_dict['states'],torch.zeros((self.ep_length-step,*b_dict['states'].size()[1:]),device=self.device)))
            self.policy_buf[self.ptr] = torch.vstack((b_dict['policies'],torch.zeros((self.ep_length-step,*b_dict['policies'].size()[1:]),device=self.device)))
            self.rew_buf[self.ptr] = torch.hstack((b_dict['returns_to_go'].squeeze(-1),torch.zeros((self.ep_length-step),device=self.device)))
        else:
            self.state_buf[self.ptr] = b_dict['states']
            self.policy_buf[self.ptr] = b_dict['policies']
            self.rew_buf[self.ptr] = b_dict['returns_to_go'].squeeze(-1)

        self.timestep_buff[self.ptr] = buff.ptr - 1
        self.act_buf[self.ptr] = buff.act_buf[buff.ptr - 1]
        self.value_buf[self.ptr] = buff.value_buf[buff.ptr - 1]
        self.next_value_buf[self.ptr] = buff.next_value_buf[buff.ptr - 1]
        self.final_buf[self.ptr] = buff.final_buf[buff.ptr - 1]

        self.ptr += 1


    def reset(self):

        if self.ptr != self.capacity:
            print(self.ptr)
            print(Warning(f'Memory not full : {self.ptr}/{self.capacity}'))
        self.state_buf = torch.empty((self.capacity,self.ep_length+1,self.bsize,self.bsize,4*COLOR_ENCODING_SIZE),device=self.device)
        self.act_buf = torch.empty((self.capacity),dtype=int,device=self.device)
        self.policy_buf = torch.empty((self.capacity,self.ep_length+1,self.n_tiles),device=self.device)
        self.value_buf = torch.empty((self.capacity),device=self.device)
        self.next_value_buf = torch.empty((self.capacity),device=self.device)
        self.rew_buf = torch.empty((self.capacity,self.ep_length+1),device=self.device)
        self.timestep_buff = torch.empty((self.capacity),dtype=int,device=self.device)
        self.final_buf = torch.empty((self.capacity),dtype=int,device=self.device)
        self.ptr = 0

    def __getitem__(self,key):
        return getattr(self,key)[:self.ptr]



class AdvantageBuffer():

    def __init__(self) -> None:

        self.state = None
        self.action = None
        self.policy = None
        self.value = None
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




