import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from param import COLOR_ENCODING_SIZE, MAX_BSIZE, TABU_LENGTH, UNIT
from segment_tree import MinSegmentTree, SumSegmentTree
from hashlib import blake2b, sha256
from pytorch_memlab import profile
from torchrl.modules import NoisyLinear


# Memory buffer with PER
class PrioritizedReplayMemory():

    def __init__(self, size,Transition,alpha, batch_size,max_bsize,encoding_size,neighborhood_size):

        device = 'cpu'

        self.Transition = Transition
        self.state_buf = np.empty((size,max_bsize+2,max_bsize+2,4*encoding_size))
        self.next_state_buf = torch.empty((size,max_bsize+2,max_bsize+2,4*encoding_size))
        self.rews_buf = np.zeros(size)
        self.target_max_val_buf = np.zeros(size)
        self.state_val_buff = np.zeros(size)
        self.obs_mask_buf = np.empty((size,neighborhood_size),dtype=bool)
        self.max_size = size
        self.batch_size =  batch_size
        self.ptr = 0
        self.size = 0
        self.max_priority = 1
        self.tree_ptr = 0
        self.alpha = alpha

        tree_capacity = 1
        while tree_capacity < size:
            tree_capacity *= 2
        
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def push(
        self,
        state: torch.Tensor,
        next_state,
        reward,
        target_max_val,
        state_val,
        mask
    ):

        self.state_buf[self.ptr] = state
        self.next_state_buf[self.ptr] = next_state
        self.rews_buf[self.ptr] = reward
        self.target_max_val_buf[self.ptr] = target_max_val
        self.state_val_buff[self.ptr] = state_val
        self.obs_mask_buf[self.ptr] = mask
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size


    def sample(self, beta: float = 0.4):
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0
        
        indices = self._sample_proportional()
        state = self.state_buf[indices]
        next_state = self.next_state_buf[indices]
        rews = self.rews_buf[indices]
        mask = self.obs_mask_buf[indices]
        tgt_val = self.target_max_val_buf[indices]
        state_val = self.state_val_buff[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
        return self.Transition(
            state,
            next_state,
            rews,
            tgt_val,
            state_val,
            mask,
            weights.reshape(-1,1),
            indices,
        )

        
    def update_priorities(self, indices, priorities:np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            try:
                assert 0 <= idx < len(self)
            except:
                print(len(self))
                print(idx)
                print(indices)

                raise BaseException
            
            
            self.sum_tree[idx] = priority.item() ** self.alpha
            self.min_tree[idx] = priority.item() ** self.alpha
            self.max_priority = max(self.max_priority, priority.item())
            
    
    def _sample_proportional(self):
        """Sample indexes based on proportions."""
        indexes = []
        p_total = self.sum_tree.sum(0, len(self)-1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            try:
                upperbound =  np.random.uniform(a,b)
            except:
                print(a,b)
                print(p_total)
                print(segment)
                print(i)
                raise OSError
            idx = self.sum_tree.find_prefixsum_idx(copy.copy(upperbound))
            
            # handle overflow when buffer is not filled
            if idx >= len(self):
                idx = len(self) - 1
                print('idx ovf')
            indexes.append(idx)
            
        return indexes
    
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight        


    def _load(
        self,
        state_buf:torch.Tensor,
        next_state_buf:torch.Tensor,
        rew_buf:torch.Tensor,
        tgtval_buf:torch.Tensor,
        heurval_buf:torch.Tensor,
        mask_buf:torch.Tensor
        ):

        state_buf = state_buf.detach().cpu()
        next_state_buf = next_state_buf.detach().cpu()
        rew_buf = rew_buf.detach().cpu()
        tgtval_buf = tgtval_buf.detach().cpu()
        heurval_buf = heurval_buf.detach().cpu()
        mask_buf = mask_buf.detach().cpu()
        
        for i in range(rew_buf.size()[0]):
            self.push(
                state_buf[i],
                next_state_buf[i],
                rew_buf[i],
                tgtval_buf[i],
                heurval_buf[i],
                mask_buf[i]
            )


    def __len__(self):
        return self.size


class CPUBuffer:
    def __init__(self,capacity:int,neighborhood_size:int,linked_mem:PrioritizedReplayMemory) -> None:
        self.capacity = capacity
        self.linked_mem = linked_mem
        self.state_buf = torch.empty((capacity,MAX_BSIZE+2,MAX_BSIZE+2,4*COLOR_ENCODING_SIZE))
        self.next_state_buf = torch.empty((capacity,MAX_BSIZE+2,MAX_BSIZE+2,4*COLOR_ENCODING_SIZE))
        self.rew_buf = torch.empty((capacity))
        self.tgtval_buf = torch.empty((capacity))
        self.heurval_buf = torch.empty((capacity))
        self.mask_buf = torch.empty((capacity,neighborhood_size))
        self.ptr = 0

    def push(
            self,
            state,
            next_state,
            reward,
            target_val,
            heur_val,
            mask
            ):


        self.state_buf[self.ptr] = state
        self.next_state_buf[self.ptr] = next_state
        self.rew_buf[self.ptr] = reward
        self.tgtval_buf[self.ptr] = target_val
        self.heurval_buf[self.ptr] = heur_val
        self.mask_buf[self.ptr] = mask

        self.ptr += 1
        if self.ptr == self.capacity:
            self.dump()
            self.ptr = 0

    def dump(self):
        self.linked_mem._load(
            self.state_buf,
            self.next_state_buf,
            self.rew_buf,
            self.tgtval_buf,
            self.heurval_buf,
            self.mask_buf,
        )
        




class MoveBuffer():

    def __init__(self) -> None:
        self.state = None
        self.next_state = None
        self.reward = None
        self.state_val = None


class TabuList():

    def __init__(self,size) -> None:
        self.size = size
        self.tabu = {}

    def push(self,state:torch.Tensor, step:int):
        key = sha256(state.cpu().numpy()).hexdigest()
        self.tabu[key] = step + TABU_LENGTH
    
    def in_tabu(self,state):
        key = sha256(state.cpu().numpy()).hexdigest()
        return key in self.tabu.keys()
    
    def update(self,step:int):
        self.tabu = {k:v for k,v in self.tabu.items() if v > step}

    def fast_foward(self):
        vals = self.tabu.values()
        m = min(vals)
        print(m)
        for k in self.tabu.keys():
            self.tabu[k] -= m
        


        
# Neural Network used in our model
class DQN(nn.Module):

    def __init__(self, h, w, outputs, device, encoding_size):

        k1 = 3
        k2 = 3
        k3 = 2

        conv2d1_size = 64
        conv2d2_size = 64
        lin_size_fact = 32

        super(DQN, self).__init__()
        self.device = device
        self.conv3d1 = nn.Conv3d(
            in_channels=1,
            out_channels=conv2d1_size,
            kernel_size= (k1,k1, 4 * encoding_size),
            dtype=UNIT,
            device=self.device,
            )
        
        self.bn1 = nn.BatchNorm2d(conv2d1_size, device=self.device)
        self.conv2d1 = nn.Conv2d(
            conv2d1_size,
            conv2d2_size,
            kernel_size=k2,
            stride=1,
            dtype=UNIT,
            device=self.device
            )
        
        self.bn2 = nn.BatchNorm2d(conv2d2_size, device=self.device)

        self.conv2d2 = nn.Conv2d(
            conv2d2_size,
            lin_size_fact,
            kernel_size=k3,
            stride=1,
            dtype=UNIT,
            device=self.device
            )

        self.bn3 = nn.BatchNorm2d(lin_size_fact, device=self.device)

        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,k1),k2),k3)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,k1),k2),k3)

        

        linear_input_size = convw * convh * lin_size_fact
        noisy_size = 64
        self.linear1 = NoisyLinear(
            linear_input_size,
            outputs,
            device=self.device,
            dtype=UNIT
            )
        


    def forward(self, x:torch.Tensor):
        x = x.to(self.device)
        x = self.conv3d1(x).squeeze(-1)
        x = F.relu(self.bn1(x))
        x = F.relu(self.bn2(self.conv2d1(x)))
        x = F.relu(self.bn3(self.conv2d2(x)))
        return self.linear1(x.view(x.size(0), -1))


