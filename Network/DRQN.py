import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from param import UNIT
from segment_tree import MinSegmentTree, SumSegmentTree

from pytorch_memlab import profile



# Memory buffer with PER
class PrioritizedReplayMemory():

    def __init__(self, size,Transition,alpha, batch_size,max_bsize,encoding_size,neighborhood_size):

        self.Transition = Transition
        self.state_buf = torch.empty((size,max_bsize+2,max_bsize+2,4*encoding_size),dtype=UNIT)
        self.next_state_buf = torch.empty((size,max_bsize+2,max_bsize+2,4*encoding_size),dtype=UNIT)
        self.rews_buf = torch.zeros(size, dtype=UNIT)
        self.obs_mask_buf = torch.empty((size,neighborhood_size),dtype=bool)
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
        mask
    ):

        self.state_buf[self.ptr] = state
        self.next_state_buf[self.ptr] = next_state
        self.rews_buf[self.ptr] = reward
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
        # try:        
        state = self.state_buf[indices]
        # except:
            # print(indices)
            # print(self.__dict__)
            # raise OSError
        next_state = self.next_state_buf[indices]
        rews = self.rews_buf[indices]
        mask = self.obs_mask_buf[indices]
        weights = torch.tensor([self._calculate_weight(i, beta) for i in indices])
        
        return self.Transition(
            state,
            next_state,
            rews,
            mask,
            weights.reshape(-1,1),
            indices,
        )

        
    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        # assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    
    def _sample_proportional(self):
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    
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


    def __len__(self):
        return self.size
        
        
# Neural Network used in our model
class DQN(nn.Module):

    def __init__(self, h, w, outputs, device, encoding_size):

        k1 = 3
        k2 = 3
        k3 = 2

        super(DQN, self).__init__()
        self.device = device
        self.conv3d1 = nn.Conv3d(
            in_channels=1,
            out_channels=16,
            kernel_size= (k1,k1, 4 * encoding_size),
            dtype=UNIT,
            device=self.device,
            )
        
        self.bn1 = nn.BatchNorm2d(16, device=self.device)
        self.conv2d1 = nn.Conv2d(
            16,
            16,
            kernel_size=k2,
            stride=1,
            dtype=UNIT,
            device=self.device
            )
        
        self.bn2 = nn.BatchNorm2d(16, device=self.device)

        self.conv2d2 = nn.Conv2d(
            16,
            8,
            kernel_size=k3,
            stride=1,
            dtype=UNIT,
            device=self.device
            )
        self.bn3 = nn.BatchNorm2d(8, device=self.device)

        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,k1),k2),k3)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,k1),k2),k3)

        linear_input_size = convw * convh * 8
        self.linear = nn.Linear(
            linear_input_size,
            outputs,
            device=self.device,
            dtype=UNIT
            )

        # self.half()

    def forward(self, x:torch.Tensor):
        x = x.to(self.device)
        x = self.conv3d1(x).squeeze(-1)
        x = F.relu(self.bn1(x))
        x = F.relu(self.bn2(self.conv2d1(x)))
        x = F.relu(self.bn3(self.conv2d2(x)))
        return self.linear(x.view(x.size(0), -1))


