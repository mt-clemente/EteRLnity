import torch
import torch.nn as nn
import torch.nn.functional as F
from param import *
from hashlib import sha256
from pytorch_memlab import profile
from torch.utils.data import TensorDataset,DataLoader

class EpisodeBuffer:
    def __init__(self,capacity:int, n_tiles:int) -> None:
        self.capacity = capacity
        self.state_buf = torch.empty((capacity,PADDED_SIZE,PADDED_SIZE,4*COLOR_ENCODING_SIZE))
        self.act_buf = torch.empty((capacity),dtype=int)
        self.policy_buf = torch.empty((capacity,4*n_tiles))
        self.value_buf = torch.empty((capacity))
        self.next_value_buf = torch.empty((capacity))
        self.rew_buf = torch.empty((capacity))


        self.final_buf = torch.empty((capacity),dtype=bool)
        self.ptr = 0

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
        self.act_buf[self.ptr] = action
        self.policy_buf[self.ptr] = policy
        self.value_buf[self.ptr] = value
        self.next_value_buf[self.ptr] = next_value
        self.rew_buf[self.ptr] = reward
        self.final_buf[self.ptr] = final

        self.ptr += 1
        if self.ptr == self.capacity:
            self.ptr = 0
        
    def reset(self):
        self.ptr = 0

class BatchMemory:
    def __init__(self,n_tiles:int,capacity:int=BATCH_SIZE, ep_length:int=256) -> None:
        self.capacity = ep_length
        self.state_buf = torch.empty((ep_length,PADDED_SIZE,PADDED_SIZE,4*COLOR_ENCODING_SIZE))
        self.act_buf = torch.empty((ep_length),dtype=int)
        self.policy_buf = torch.empty((ep_length,4*n_tiles))
        self.value_buf = torch.empty((ep_length))
        self.next_value_buf = torch.empty((ep_length))
        self.rew_buf = torch.empty((ep_length))


        self.final_buf = torch.empty((ep_length),dtype=bool)
        self.ptr = 0


    def load(self,buff:EpisodeBuffer):
        self.state_buf[self.ptr] = buff.state_buf
        self.act_buf[self.ptr] = buff.act_buf
        self.policy_buf[self.ptr] = buff.policy_buf
        self.value_buf[self.ptr] = buff.value_buf
        self.next_value_buf[self.ptr] = buff.next_value_buf
        self.rew_buf[self.ptr] = buff.rew_buf

        self.ptr += 1
    
        buff.reset()
    
    def reset(self):
        self.ptr = 0




class AdvantageBuffer():

    def __init__(self) -> None:

        self.state = None
        self.action = None
        self.policy = None
        self.value = None
        self.reward = None



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

    def is_stale(self):
        return self.eos
        
    
    def reset(self):
        self.counter = 0
        self.prev_score = 0
        self.eos = False




class BoardConv(nn.Module):

    def __init__(self,kernel_size,layer_size,device,encoding_size=5) -> None:
        super().__init__()

        self.conv3d = nn.Conv3d(
            in_channels=1,
            out_channels=layer_size,
            kernel_size= (kernel_size,kernel_size, 4 * encoding_size),
            dtype=UNIT,
            padding_mode='zeros',
            device=device,
            )
        
    def  forward(self,x):
        return self.conv3d(x).squeeze(-1)

class View(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        print(input.size())
        print(self.shape)
        return input.view(*self.shape)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, kernel_sizes, device):

        self.state_dim =state_dim
        super(ActorCritic, self).__init__()

        lin_size = self.lin_size(kernel_sizes,hidden_sizes)

        print(lin_size)

        self.actor = nn.Sequential(
            BoardConv(kernel_sizes[0],hidden_sizes[0],device=device),
            nn.ELU(),
            nn.Conv2d(hidden_sizes[0], hidden_sizes[1],kernel_sizes[1],device=device),
            View((lin_size,)),
            nn.ELU(),
            nn.Linear(lin_size, action_dim,device=device),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            BoardConv(kernel_sizes[0],hidden_sizes[0],device=device),
            nn.ELU(),
            nn.Conv2d(hidden_sizes[0], hidden_sizes[1],kernel_sizes[1],device=device),
            View((lin_size,)),
            nn.ELU(),
            nn.Linear(lin_size, 1,device=device),
            nn.Softmax(dim=-1)
        )


    def lin_size(self,kernel_sizes,hidden_sizes,strides=None):

        size = self.state_dim

        if strides is None:
            strides = [1] * len(kernel_sizes)

        for ks,st in zip(kernel_sizes,strides):
            size = (size - ks) // st + 1

        return size ** 2 * hidden_sizes[-1]


    def forward(self, state):
        print("AAAAA")
        print(state.size())
        policy = self.actor(state)
        value = self.critic(state)
        return policy, value

class PPOAgent:
    def __init__(self,config):
        self.gamma = config['gamma']
        self.clip_ratio = config['clip_ratio']
        self.target_kl = config['target_kl']
        self.lr = config['lr']
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.state_dim = config['state_dim']
        device = config['device']
        n_tiles = config['n_tiles']
        hidden_sizes = config['hidden_sizes']
        kernel_sizes = config['kernel_sizes']

        self.action_dim = n_tiles*4
        self.model = ActorCritic(self.state_dim, self.action_dim, hidden_sizes, kernel_sizes, device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
    def get_action(self, policy:torch.Tensor):

        print(policy.size())
        sm = torch.softmax(policy,-1)
        print(sm.size())
        return torch.multinomial(sm,1) # TODO: Change for beam search
        
    def compute_advantage(self, rewards, values, next_value, finals):
        td_target = rewards + self.gamma * (1 - finals) * next_value
        td_error = td_target - values
        advantage = []
        adv = 0
        for delta in td_error.detach().numpy()[::-1]:
            adv = self.gamma * adv * (1 - self.clip_ratio) + torch.clip(delta, -self.clip_ratio, self.clip_ratio)
            advantage.append(adv)
        advantage.reverse()
        return advantage
    
    # def update(self, states, actions, old_policies, values, advantages, returns):
    def update(self,mem:BatchMemory):

        advantages = self.compute_advantage(
            reward=mem.rew_buf,
            values=mem.value_buf,
            next_value=mem.next_value_buf,
            finals=mem.final_buf
            )

        returns = advantages + mem.value_buf


        dataset = TensorDataset(
            mem.state_buf,
            mem.act_buf,
            mem.policy_buf,
            mem.value_buf,advantages,
            returns)
        
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Perform multiple update epochs
        for _ in range(self.epochs):
            for batch in loader:
                batch_states, batch_actions, batch_old_policies, batch_values, batch_advantages, batch_returns = batch
                
                # Calculate new policy and value estimates
                batch_policy, batch_value = self.model(batch_states)
                
                # Calculate ratios and surrogates for PPO loss
                ratio = torch.exp(torch.log(batch_policy) - torch.log(batch_old_policies))
                clip_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                surrogate1 = ratio * batch_advantages
                surrogate2 = clip_ratio * batch_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Calculate value function loss
                value_loss = F.mse_loss(batch_value.squeeze(), batch_returns)
                
                # Calculate entropy bonus
                entropy = -torch.sum(batch_policy * torch.log(batch_policy), dim=1).mean()
                nn.CrossEntropyLoss()
                entropy_loss = -self.entropy_weight * entropy
                
                # Compute total loss and update parameters
                loss = policy_loss + self.value_weight * value_loss + entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()