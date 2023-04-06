import copy
import math
import random
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from param import *
from segment_tree import MinSegmentTree, SumSegmentTree
from hashlib import sha256
from pytorch_memlab import profile
from torch.nn import init

# Memory buffer with PER
class PrioritizedReplayMemory():

    def __init__(self, size,Transition,alpha, batch_size,encoding_size,n_tiles):

        device = 'cpu'

        self.Transition = Transition
        self.state_buf = torch.empty((size,n_tiles),dtype=bool)
        self.next_state_buf = torch.empty((size,n_tiles),dtype=bool)
        self.rews_buf = torch.zeros(size)
        self.act_buf = torch.zeros(size,dtype=int)
        self.final_buf = torch.empty((size),dtype=bool)
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
        act,
        final
    ):

        self.state_buf[self.ptr] = state
        self.next_state_buf[self.ptr] = next_state
        self.rews_buf[self.ptr] = reward
        self.act_buf[self.ptr] = act
        self.final_buf[self.ptr] = final
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
        acts = self.act_buf[indices]
        final = self.final_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
        return self.Transition(
            state,
            next_state,
            rews,
            acts,
            final,
            weights.reshape(-1,1),
            indices
        )

        
    def update_priorities(self, indices, priorities:np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            if priority < 0:
                print("AAAAAAAAAAAAAAAAA",priority)
                raise OSError
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
        act_buf:torch.Tensor,
        final_buf:torch.Tensor
        ):

        state_buf = state_buf.detach().cpu()
        next_state_buf = next_state_buf.detach().cpu()
        rew_buf = rew_buf.detach().cpu()
        act_buf = act_buf.detach().cpu()
        final_buf = final_buf.detach().cpu()
        
        for i in range(rew_buf.size()[0]):
            self.push(
                state_buf[i],
                next_state_buf[i],
                rew_buf[i],
                act_buf[i],
                final_buf[i]
            )


    def __len__(self):
        return self.size


class CPUBuffer:
    def __init__(self,capacity:int,n_tiles:int,linked_mem:PrioritizedReplayMemory) -> None:
        self.capacity = capacity
        self.linked_mem = linked_mem
        self.state_buf = torch.empty((capacity,n_tiles),dtype=bool)
        self.next_state_buf = torch.empty((capacity,n_tiles),dtype=bool)
        self.rew_buf = torch.empty((capacity))

        self.act_buf = torch.empty((capacity),dtype=int)

        self.final_buf = torch.empty((capacity),dtype=bool)
        self.ptr = 0

    def push(
            self,
            state,
            next_state,
            reward,
            action,
            final
            ):


        self.state_buf[self.ptr] = state
        self.next_state_buf[self.ptr] = next_state
        self.rew_buf[self.ptr] = reward
        self.act_buf[self.ptr] = action
        self.final_buf[self.ptr] = final

        self.ptr += 1
        if self.ptr == self.capacity:
            self.dump()
            self.ptr = 0

    def dump(self):
        self.linked_mem._load(
            self.state_buf,
            self.next_state_buf,
            self.rew_buf,
            self.act_buf,
            self.final_buf,
        )
        




class MoveBuffer():

    def __init__(self) -> None:
        self.state = None
        self.next_state = None
        self.reward = None
        self.action = None


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
    
    def filter(self,step:int):
        self.tabu = {k:v for k,v in self.tabu.items() if v > step}

    def get_update(self,batch:torch.Tensor,step:int):

        np_batch = batch.cpu().numpy()
        for i in range(np_batch.shape[0]):

            key = sha256(np_batch[i]).hexdigest()

            if key not in self.tabu.keys():

                self.push(batch[i],step)
                return torch.from_numpy(np_batch[i]).to(device=batch.device).to(dtype=batch.dtype),i
            
        return None,None

    def fast_foward(self):
        vals = self.tabu.values()
        m = min(vals)
        print(m)
        for k in self.tabu.keys():
            self.tabu[k] -= m
        

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

    def stop(self):
        return self.eos
        
    
    def reset(self):
        self.counter = 0
        self.prev_score = 0
        self.eos = False


class oldDTQN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_layers, num_heads, dim_hidden, device):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim,device=device,dtype=UNIT)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                embedding_dim,
                num_heads,
                dim_hidden,
                batch_first = True,
                device=device,
                dtype=UNIT
            ),
            num_layers,
        )
        self.fc = nn.Linear(embedding_dim, num_embeddings,device=device,dtype=UNIT)

        self.repeat_id = torch.arange(1024,device=device,dtype=int).repeat(BATCH_SIZE,1)

    def forward(self, mask):
        
        if mask.dim() == 1:
            embedded_tiles = self.embedding(self.repeat_id[0][mask])
        elif mask.dim() == 2:
            embedded_tiles = self.embedding(self.repeat_id[:mask.size()[0]][mask])
        else:
            raise RuntimeError
        
        print(embedded_tiles.size(),mask.size())
        x = embedded_tiles * mask.unsqueeze(-1)
        transformer_output = self.transformer_encoder(x,mask=None)


        if mask.dim() == 1:
            predicted_tile_scores = self.fc(transformer_output[-1])
        elif mask.dim() == 2:
            predicted_tile_scores = self.fc(transformer_output[:,-1])

        return predicted_tile_scores
    

class DTQN(nn.Module):
    def __init__(self, board_size, num_tiles, embedding_dim, num_layers, num_heads, dim_hidden, device):
        super(DTQN, self).__init__()
        self.board_size = board_size
        self.num_tiles = num_tiles
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_hidden = dim_hidden
        
        self.positional_encoding = self._get_positional_encoding()
        self.tile_embedding = nn.Embedding(num_tiles, embedding_dim)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=dim_hidden,
                device=device
            ),
            num_layers=num_layers,
        )
        
        self.fc = nn.Linear(embedding_dim, num_tiles,device=device)


    def forward(self, board:torch.Tensor, tile:torch.Tensor,mask:torch.BoolTensor):
        batch_size = board.shape[0]
        
        # Flatten neighbors
        sequence = torch.hstack((
            board[torch.arange(batch_size,device=board.device),tile[:,0]-1,tile[:,1]].view(4),
            board[torch.arange(batch_size,device=board.device),tile[:,0],tile[:,1]-1].view(4),
            board[torch.arange(batch_size,device=board.device),tile[:,0],tile[:,1]+1].view(4)))
        

        pad_mask = torch.zeros(sequence) == False

        # The tile on the right is only important on the right of the board
        pad_mask[:,8:] = (tile[:,] != MAX_BSIZE)
        
        # Embed tiles
        tile_embedded = self.tile_embedding(sequence)
        
        # Add positional encoding
        positional_encoding = self.positional_encoding[:tile_embedded.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1)
        positional_encoded = tile_embedded + positional_encoding.to(tile_embedded.device)

        # Transformer encoding
        transformer_output = self.transformer(positional_encoded.transpose(0, 1),src_key_padding_mask=pad_mask).transpose(0, 1)
        
        # Output
        q = self.fc(transformer_output)
        q = q * mask
        return q
    
    def _get_positional_encoding(self):
        pe = torch.zeros(self.board_size ** 2, self.embedding_dim)
        position = torch.arange(0, self.board_size ** 2, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * (-math.log(10000.0) / self.embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe