import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_tree import MinSegmentTree, SumSegmentTree
import numpy as np




# Memory buffer with PER
class PrioritizedReplayMemory():

    def __init__(self, size,Transition,alpha, batch_size):

        self.Transition = Transition
        self.state_buf = np.empty((size,9,9),dtype=np.float64)
        self.next_state_buf = np.empty(size, dtype=torch.Tensor)
        self.rews_buf = np.zeros(size, dtype=np.float32)
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
    ):

        self.state_buf[self.ptr] = state
        self.next_state_buf[self.ptr] = next_state
        self.rews_buf[self.ptr] = reward
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
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
        return self.Transition(
            state,
            next_state,
            rews,
            weights.reshape(-1,1),
            indices,
        )

        
    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

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
        
        
# Neural Network used in our modek
class DQN(nn.Module):

    def __init__(self, h, w, outputs,device):
        super(DQN, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(1, 9 * 9, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(9 * 9)
        self.conv2 = nn.Conv2d(9 * 9, 64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=2, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,5),3),2)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,5),3),2)
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)


    def forward(self, x):
        x = x.to(self.device)
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = F.elu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class BestMove():
    def __init__(self) -> None:
        self.move = None

    def update(self, act) -> None:
        self.move = act


class MoveBuffer():
    def __init__(self) -> None:
        self.statem = None
        self.reward = None

    def update(self, statem: torch.Tensor, reward):
        self.statem = statem
        self.reward = reward
    
    def reset(self):
        self.statem = None
        self.reward = None