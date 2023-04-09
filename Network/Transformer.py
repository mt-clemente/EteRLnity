import torch
import torch.nn as nn
from datetime import datetime
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from param import *
from Trajectories import BatchMemory
from torch.utils.data import TensorDataset,DataLoader
import torch.nn.init as init


# -------------------- AGENT --------------------
class PPOAgent:
    def __init__(self,config):
        self.gamma = config['gamma']
        self.clip_eps = config['clip_eps']
        self.lr = config['lr']
        self.epochs = config['epochs']
        self.minibatch_size = config['minibatch_size']
        self.horizon = config['horizon']
        self.state_dim = config['state_dim']
        self.entropy_weight = config['entropy_weight']
        self.value_weight = config['value_weight']
        self.gae_lambda = config['gae_lambda']
        device = config['device']
        n_tiles = config['n_tiles']
        hidden_size = config['hidden_size']
        conv_sizes = config['conv_sizes']
        kernel_sizes = config['kernel_sizes']

        self.action_dim = n_tiles
        self.model = DecisionTransformerAC(
            self.state_dim,
            self.action_dim,
            hidden_size,
            conv_sizes,
            kernel_sizes,
            device,
            )
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,weight_decay=1e-4)
        
    def get_action(self, policy:torch.Tensor,mask:torch.BoolTensor):

        if mask.count_nonzero() == 0:
            raise Exception("No playable tile")
        sm = torch.softmax(policy,-1) * mask
        sm /= sm.sum(dim=-1, keepdim=True)
        return torch.multinomial(sm,1)
        
    def compute_gae(self, rewards, values, next_values, finals):

        td_errors = rewards + self.gamma * next_values * (1 - finals) - values
        gae = 0
        advantages = torch.zeros_like(td_errors)
        for t in reversed(range(len(td_errors))):
            gae = td_errors[t] + self.gamma * self.gae_lambda * (1 - finals[t]) * gae
            advantages[t] = gae
        return advantages
    
    # def update(self, states, actions, old_policies, values, advantages, returns):
    def update(self,mem:BatchMemory):
        
        # FIXME: check timestep
        last_seq_rewards = mem['rew_buf'][torch.arange(mem['rew_buf'].size()[0]),mem['timestep_buff']]

        advantages = self.compute_gae(
            rewards=last_seq_rewards,
            values=mem['value_buf'],
            next_values=mem['next_value_buf'],
            finals=mem['final_buf']
            )

        returns = advantages + mem['value_buf']

        
        device = 'cuda'
        t0 = datetime.now()
        batch_size, ep_length = mem['state_buf'].size()[:2]
        timestep_seq = repeat(torch.arange(ep_length-1,device=mem.device),'a -> b a',b = batch_size)
        timestep_seq[timestep_seq > repeat(mem['timestep_buff'],'b -> b e',e=ep_length-1)] = 0

        dataset = TensorDataset(
            mem['state_buf'][:,1:],
            mem['act_buf'],
            mem['rew_buf'][:,1:],
            mem['policy_buf'][:,1:],
            advantages,
            returns,
            timestep_seq
        )

        loader = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)
        
        self.model = self.model.to(device)
        # Perform multiple update epochs
        for k in range(self.epochs):
            for batch in loader:
                batch_states, batch_actions, batch_rtg,batch_old_policies, batch_advantages, batch_returns, batch_timesteps = batch

                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std()+1e-7)
                batch_returns = (batch_returns - batch_returns.mean()) / (batch_returns.std()+1e-7)

                # Calculate new policy and value estimates
                
                batch_policy, batch_value = self.model(
                    batch_states,
                    batch_old_policies,
                    batch_rtg,
                    batch_timesteps
                )

                # Calculate ratios and surrogates for PPO loss
                action_probs = batch_policy.gather(1, batch_actions.unsqueeze(1))
                old_action_probs = batch_old_policies[torch.arange(batch_timesteps.size()[0],device=device),batch_timesteps.argmax(dim=-1)].gather(1, batch_actions.unsqueeze(1))
                ratio = action_probs / (old_action_probs + 1e-4)
                print(ratio.max())
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                surrogate1 = ratio * batch_advantages.unsqueeze(1)
                surrogate2 = clipped_ratio * batch_advantages.unsqueeze(1)
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                # Calculate value function loss
                value_loss = F.mse_loss(batch_value.squeeze(-1), batch_returns) * self.value_weight

                # Calculate entropy bonus
                entropy = -(batch_policy[batch_policy != 0] * torch.log(batch_policy[batch_policy != 0])).sum(dim=-1).mean()
                entropy_loss = -self.entropy_weight * entropy
                # Compute total loss and update parameters


                loss = policy_loss + value_loss + entropy_loss
                if True:
                    print("--------")
                    print(batch_actions.max())
                    print(batch_actions.min())
                    print(batch_advantages.max())
                    print(batch_advantages.min())
                    print(batch_old_policies.max())
                    print(batch_old_policies.min())
                    print(batch_returns.max())
                    print(batch_returns.min())
                    print(batch_policy.min())
                    print(entropy_loss.item(),value_loss.item(),policy_loss.item(),loss)
                    print(batch_states.max())
                    print(batch_states.min())

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),0.5)
                self.optimizer.step()
                
                g=0
                for name,param in self.model.named_parameters():
                    # print(f"{name:>28} - {torch.norm(param.grad)}")
                    g+=torch.norm(param.grad)
                    param.grad.clamp_(-1,1)

            if k == self.epochs -1:
                    

                wandb.log({
                    "Total loss":loss,
                    "Cumul grad norm":g,
                    "Value loss":value_loss,
                    "Entropy loss":entropy_loss,
                    "Policy loss":policy_loss,
                    })

        if  not torch.cuda.is_available():
            self.model = self.model.cpu()
        print(datetime.now()-t0)
        mem.reset()
    




def init_weights(m):
    if type(m) == nn.Module:
        init.xavier_normal_(m.weight)

# self.critic.apply(init_weights)
# self.actor.apply(init_weights)

def lin_size(kernel_sizes, dim, strides=None):

    size = dim

    if strides is None:
        strides = [1] * len(kernel_sizes)

    for ks,st in zip(kernel_sizes,strides):
        
        try:
            size = (size - ks) // st + 1
        except:
            size = (size - ks[0]) // st + 1


    return size


# -------------------- ACTOR / CRITIC --------------------
    
class DecisionTransformerAC(nn.Module):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            conv_sizes,
            kernel_sizes,
            device,
            max_length=None,
            max_ep_len=256,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.max_length = max_length

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=128,
                dim_feedforward=128,
                nhead=4,
                batch_first=True,
                device=device,
                dtype=UNIT
            ),
            num_layers=3,
            norm=None,
        ) #FIXME:



        self.dim_embed = hidden_size #FIXME:

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size,device=device,dtype=UNIT)
        self.embed_return = torch.nn.Linear(1, hidden_size,device=device,dtype=UNIT)
        self.embed_policy = torch.nn.Linear(act_dim, hidden_size,device=device,dtype=UNIT)
        self.embed_state = nn.Sequential(
            Conv3to2d(kernel_sizes[0],
                      1,
                      conv_sizes[0],
                      device
                      ),
            nn.Conv2d(conv_sizes[0],conv_sizes[1],kernel_sizes[1],device=device,dtype=UNIT),
            nn.Conv2d(conv_sizes[1],conv_sizes[2],kernel_sizes[2],device=device,dtype=UNIT),
            nn.AdaptiveAvgPool2d(1),
            View((conv_sizes[-1],-1)),
            nn.Linear(conv_sizes[-1],hidden_size,device=device,dtype=UNIT)
        )
        

        self.embed_ln = nn.LayerNorm(hidden_size,device=device,dtype=UNIT)

        # note: we don't predict states or returns for the paper
        # self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        # self.predict_return = torch.nn.Linear(hidden_size, 1)

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, act_dim,device=device,dtype=UNIT),
            SoftmaxStable()
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 1,device=device,dtype=UNIT)
        )


    def forward(self, states, policy, returns_to_go, timesteps, attention_mask=None):
        
        if states.size()[1] != 1:
            batch_size, seq_length = states.shape[0], states.shape[1]
            states_ = rearrange(states,'b ep h w d -> (b ep) h w d').unsqueeze(1)
            returns_to_go_ = rearrange(returns_to_go,'b ep -> (b ep)').unsqueeze(1)
            policy_ = rearrange(policy,'b ep a -> (b ep) a')
            timesteps_ = rearrange(timesteps,'b ep -> (b ep)')
        else:
            batch_size, seq_length = 1, states.shape[0]
            states_ = states
            policy_ = policy
            returns_to_go_ = returns_to_go
            timesteps_ = timesteps

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        
        
        # embed each modality with a different head
        state_embeddings = self.embed_state(states_)
        
        policy_embeddings = self.embed_policy(policy_.to(UNIT))
        returns_embeddings = self.embed_return(returns_to_go_)
        time_embeddings = self.embed_timestep(timesteps_)


        if states.size()[1] != 1:
            state_embeddings = rearrange(state_embeddings,'(b ep) e -> b ep e',b=batch_size,e=self.dim_embed)
            policy_embeddings = rearrange(policy_embeddings,'(b ep) e -> b ep e',b=batch_size,e=self.dim_embed)
            returns_embeddings = rearrange(returns_embeddings,'(b ep) e -> b ep e',b=batch_size,e=self.dim_embed)
            time_embeddings = rearrange(time_embeddings,'(b ep) e -> b ep e',b=batch_size,e=self.dim_embed)


        state_embeddings = state_embeddings + time_embeddings
        policy_embeddings = policy_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict policy
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, policy_embeddings), dim=1
        )
        
        # FIXME: st.permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        # if states.size()[1] != 1:
        stacked_inputs = stacked_inputs.reshape(batch_size, 3*seq_length, self.hidden_size)

            
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        # stacked_attention_mask = torch.stack(
        # (attention_mask, attention_mask, attention_mask), dim=1
        # ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            memory=torch.zeros_like(stacked_inputs),
            tgt=stacked_inputs,
        )

        x = transformer_outputs.reshape(batch_size, 3, seq_length, self.dim_embed)

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or policy (2); i.e. x[:,1,t] is the token for s_t
        # x =

        # get predictions
        policy_preds = self.policy_head(x[:,2,-1,:])  # predict next policy given state
        value_preds = self.value_head(x[:,2,-1,:])


        return policy_preds, value_preds

    def get_action(self, sequence_buf:dict):
        # we don't care about the past rewards in this model
        states = sequence_buf['states']
        policies = sequence_buf['policies']
        returns_to_go = sequence_buf['returns_to_go']
        timesteps = sequence_buf['timesteps']
        states.unsqueeze_(1)

        policy_preds, value_preds = self.forward(
            states, policies, returns_to_go, timesteps, attention_mask=None)
        
        return policy_preds[0,-1], value_preds[0,-1]
    



class Conv3to2d(nn.Module):

    def __init__(self,kernel_size,input_channels,layer_size,device) -> None:
        super().__init__()

        self.conv = nn.Conv3d(
            in_channels=input_channels,
            out_channels=layer_size,
            kernel_size= (kernel_size,kernel_size,5),
            dtype=UNIT,
            device=device,
            )
        
    def  forward(self,x):
        x = self.conv(x)
        x = rearrange(x,'b c h w d -> b c h (w d)')
        return x


class View(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return rearrange(input,'b c h w -> b (c h w)')

class SoftmaxStable(nn.Module):
    def forward(self, x):
        x = x - x.max(dim=-1, keepdim=True).values
        return F.softmax(x, dim=-1)

