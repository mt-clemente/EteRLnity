import copy
import math
import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from param import *
from Trajectories import EpisodeBuffer
from torch.utils.data import TensorDataset,DataLoader
# -------------------- AGENT --------------------
class PPOAgent:
    def __init__(self,config):
        self.gamma = config['gamma']
        self.clip_eps = config['clip_eps']
        self.lr = config['lr']
        self.epochs = config['epochs']
        self.minibatch_size = config['minibatch_size']
        self.horizon = config['horizon']
        self.seq_len = config['seq_len']
        self.state_dim = config['state_dim']
        self.entropy_weight = config['entropy_weight']
        self.value_weight = config['value_weight']
        self.gae_lambda = config['gae_lambda']
        self.device = 'cuda' if torch.cuda.is_available() and CUDA_ONLY else 'cpu'
        n_tiles = config['n_tiles']
        hidden_size = config['hidden_size']
        dim_embed = config['dim_embed']
        n_encoder_layers = config['n_encoder_layers']
        n_decoder_layers = config['n_decoder_layers']
        n_heads = config['n_heads']

        self.action_dim = n_tiles

        self.model = DecisionTransformerAC(
            state_dim=self.state_dim,
            act_dim=self.action_dim,
            dim_embed=dim_embed,
            hidden_size=hidden_size,
            n_decoder_layers=n_decoder_layers,
            n_encoder_layers=n_encoder_layers,
            n_heads=n_heads,
            max_length=self.seq_len,
            device=self.device,
            )
        
        self.workers = nn.ModuleList([copy.deepcopy(self.model) for i in range(NUM_WORKERS)])
        

        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr,weight_decay=1e-4,eps=OPT_EPSILON)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=self.optimizer,
            start_factor=0.01,
            end_factor=1,
            total_iters=1e5,
        )

        

    # def update(self, states, actions, old_policies, values, advantages, returns):
    def update(self,loader:DataLoader):
        """
        Updates the ppo agent, using the trajectories in the memory buffer.
        For states, policy, rewards, advantages, and timesteps the data is in a 
        straightforward format [batch,*values]
        For the returns-to-go and actions the data has a format [batch,sequence_len+1].
        We need the sequence coming before the state to make a prediction, and the current
        action to calculate the policy and ultimately the policy loss.

        """


        t0 = datetime.now()

        

        # Perform multiple update epochs
        for k in range(self.epochs):
            for batch in loader:

                (
                    batch_states,
                    batch_actions,
                    batch_tile_seq,
                    batch_masks,
                    batch_advantages,
                    batch_old_policies,
                    batch_returns,
                    batch_timesteps,
                    
                ) = batch

                
                batch_policy, batch_value = self.model(
                    batch_states,
                    batch_tile_seq,
                    batch_timesteps,
                    batch_masks,
                )

                # batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std()+1e-4)
                # batch_returns = (batch_returns - batch_returns.mean()) / (batch_returns.std()+1e-4)

                # Calculate ratios and surrogates for PPO loss
                action_probs = batch_policy.gather(1, batch_actions.unsqueeze(1))
                old_action_probs = batch_old_policies.gather(1, batch_actions.unsqueeze(1))
                ratio = action_probs / (old_action_probs + 1e-5)
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
                if DEBUG:
                    print(f"---- EPOCH {k} ----")
                    print("ba",batch_actions.max())
                    print("ba",batch_actions.min())
                    print("badv",batch_advantages.max())
                    print("badv",batch_advantages.min())
                    print("bop",batch_old_policies.max())
                    print("bop",batch_old_policies.min())
                    print("bret",batch_returns.max())
                    print("bret",batch_returns.min())
                    print("bp",batch_policy.max())
                    print("bp",batch_policy.min())
                    print("Loss",entropy_loss.item(),value_loss.item(),policy_loss.item(),loss)
                    print("ratio",ratio.max())
                    print("ratio",ratio.min())
                    print("ratio",((ratio > 1 + CLIP_EPS).count_nonzero() + (ratio < 1 - CLIP_EPS).count_nonzero()))
                    print("bst",batch_states.max())
                    print("bst",batch_states.min())

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.5)
                self.optimizer.step()
                self.scheduler.step()
                g=0
                i = 0
                for name,param in self.model.named_parameters():
                    i+=1
                    # print(f"{name:>28} - {torch.norm(param.grad)}")
                    # print(f"{name:>28}")
                    g+=torch.norm(param.grad)

        print(datetime.now()-t0)
        wandb.log({
            "Total loss":loss,
            "Current learning rate":self.scheduler.get_last_lr()[0],
            "Cumul grad norm":g,
            "Value loss":value_loss,
            "Entropy loss":entropy_loss,
            "Policy loss":policy_loss,
            "Value repartition":batch_value.squeeze(-1).detach(),
            "KL div": (batch_old_policies * (torch.log(batch_old_policies + 1e-5) - torch.log(batch_policy + 1e-5))).sum(dim=-1).mean()
            })


        for worker in self.workers:
            worker.load_state_dict(self.model.state_dict())


        if not CUDA_ONLY:
            self.model = self.model.cpu()
    
    def save_models(self,dir:str,episode:int):

        Path(dir).mkdir(parents=True, exist_ok=True)
        Path(f"{dir}/transformers").mkdir(parents=True, exist_ok=True)
        Path(f"{dir}/heads").mkdir(parents=True, exist_ok=True)
        Path(f"{dir}/embeds").mkdir(parents=True, exist_ok=True)

        torch.save(self.model.actor_dt.state_dict(), f'{dir}/transformers/actor_{episode}.pt')
        torch.save(self.model.critic_dt.state_dict(), f'{dir}/transformers/critic_{episode}.pt')

        torch.save(self.model.actor_head.state_dict(), f'{dir}/heads/actor_{episode}.pt')
        torch.save(self.model.critic_head.state_dict(), f'{dir}/heads/critic_{episode}.pt')


        torch.save(self.model.embed_actions.state_dict(), f'{dir}/embeds/action_{episode}.pt')
        torch.save(self.model.embed_state.state_dict(), f'{dir}/embeds/state_{episode}.pt')
        torch.save(self.model.embed_timestep.state_dict(), f'{dir}/embeds/time_{episode}.pt')


    def load_model(self,load_dict:dict):
        """
        Loads a model depending on the given loading dictionary:

        Keys
        -----------------
        * actor_dt : the actor transformer
        * critic_dt : the critic transformer
        * actor_head : the actor head
        * critic_head : the critic head
        * embed_action : action color embedding
        * embed_state : state color embedding
        * embed_time : timestep embedding
        """

        main_dir = 'models/checkpoint'

        corresp_dict : dict[nn.Module] = {
            'actor_dt': {
                'model':self.model.actor_dt,
                'dir':f'{main_dir}/transformers'
                },
            'critic_dt': {
                'model':self.model.critic_dt,
                'dir':f'{main_dir}/transformers'
                },
            'actor_head': {
                'model':self.model.actor_head,
                'dir':f'{main_dir}/heads'
                },
            'critic_head': {
                'model':self.model.critic_head,
                'dir':f'{main_dir}/heads'
                },
            'embed_action': {
                'model':self.model.embed_actions,
                'dir':f'{main_dir}/embeds'
                },
            'embed_state': {
                'model':self.model.embed_state,
                'dir':f'{main_dir}/embeds'
                },
            'embed_time': {
                'model':self.model.embed_timestep,
                'dir':f'{main_dir}/embeds'
                },
        }


        for key in load_dict.keys:
            state_dict = torch.load(f"{corresp_dict[key]['dir']/{key}}.pt")
            model = corresp_dict[key]['model']
            model.load_state_dict(state_dict)







# -------------------- ACTOR / CRITIC --------------------
    
class DecisionTransformerAC(nn.Module):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            dim_embed,
            hidden_size,
            n_encoder_layers,
            n_decoder_layers,
            n_heads,
            device,
            max_length,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.bsize = state_dim - 2
        self.n_tiles = (state_dim-2)**2
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.seq_length = max_length


        self.ep_buf = EpisodeBuffer(
            ep_len=self.n_tiles,
            n_tiles=self.n_tiles,
            bsize=state_dim,
            horizon=HORIZON,
            seq_len=self.seq_length,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            device=device
        )


        self.actor_dt =  nn.Transformer(
                d_model=4*dim_embed,
                num_encoder_layers=n_encoder_layers,
                num_decoder_layers=n_decoder_layers,
                nhead=n_heads,
                dim_feedforward=self.hidden_size,
                dropout=0.05,
                activation=F.gelu,
                batch_first=True,
                norm_first=True,
                device=device,
                dtype=UNIT,
            )
        
        self.critic_dt =  nn.Transformer(
                d_model=4*dim_embed,
                num_encoder_layers=n_encoder_layers,
                num_decoder_layers=n_decoder_layers,
                dim_feedforward=self.hidden_size,
                nhead=n_heads,
                dropout=0.05,
                activation=F.gelu,
                batch_first=True,
                norm_first=True,
                device=device,
                dtype=UNIT,
            )
        

        self.actor_head = nn.Sequential(
            nn.GELU(),
            nn.Linear(4*dim_embed,act_dim,device=device,dtype=UNIT),
        )
        self.policy_head = MaskedStableSoftmax()

        self.critic_head = nn.Sequential(
            nn.Linear(4*dim_embed, 1,device=device,dtype=UNIT),
        )

        def init_weights(module):
            if isinstance(module, nn.Linear):
                if UNIT == torch.half:
                    nn.init.xavier_normal_(module.weight)
                else:
                    nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


        self.actor_dt.apply(init_weights)
        self.critic_dt.apply(init_weights)
        # Apply the initialization to the sublayers of the transformer layers

        wandb.watch(self.actor_dt,log='all',log_freq=500)
        wandb.watch(self.actor_head,log='all',log_freq=500)
        wandb.watch(self.critic_dt,log='all',log_freq=500)
        wandb.watch(self.critic_head,log='all',log_freq=500)
            
        self.dim_embed = dim_embed 
        self.embed_timestep = nn.Embedding(self.n_tiles, 4*dim_embed,device=device,dtype=UNIT)
        self.embed_actions = torch.nn.Embedding(N_COLORS + 2, dim_embed,device=device,dtype=UNIT) # NCOLLORS, BOS, PAD
        self.embed_state = nn.Embedding(N_COLORS,dim_embed,device=device,dtype=UNIT)
        self.positional_encoding = PE3D(dim_embed)
        self.embed_ln = nn.LayerNorm(4*dim_embed,eps=1e-5,device=device,dtype=UNIT)



    def forward(self, states, actions, timesteps, tile_mask, attention_mask=None):


        #BOS   

        if states.dim() == 4:

            batch_size = states.shape[0]
            states_ = states
            actions_ = actions.unsqueeze(-1)
            timesteps_ = timesteps.unsqueeze(-1)
        else:
            batch_size = 1
            states_ = states.unsqueeze(0)
            actions_ = actions.unsqueeze(0)
            timesteps_ = timesteps.unsqueeze(0)

        # embed each modality with a different head
        key_padding_mask = (torch.any((actions_ == -1).squeeze(-1),-1))
        actions_embeddings = self.embed_actions(actions_.int() + 2).reshape(batch_size, self.seq_length + 1,self.dim_embed * 4)
        time_embeddings = self.embed_timestep(timesteps_)
        actions_embeddings = actions_embeddings + time_embeddings
        state_embeddings = self.embed_state(states_[:,1:-1,1:-1,:].int())
        state_embeddings += self.positional_encoding(state_embeddings)
        state_embeddings = state_embeddings.view(-1,self.n_tiles,self.dim_embed * 4)

        tgt_inputs = self.embed_ln(actions_embeddings)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        policy_tokens = self.actor_dt(
            src=state_embeddings,
            tgt=tgt_inputs,
            tgt_key_padding_mask=key_padding_mask
        )

        value_tokens = self.critic_dt(
            src=state_embeddings,
            tgt=tgt_inputs,
            tgt_key_padding_mask=key_padding_mask
        )

        policy_logits = self.actor_head(policy_tokens[torch.arange(batch_size,device=policy_tokens.device),(timesteps)%(HORIZON+1)].reshape(batch_size,self.dim_embed*4))
        policy_pred = self.policy_head(policy_logits,tile_mask)
        value_pred = self.critic_head(value_tokens[torch.arange(batch_size,device=value_tokens.device),(timesteps)%(HORIZON+1)].reshape(batch_size,self.dim_embed*4))

        # policy_ouputs = policy_ouputs.reshape(batch_size, 3, self.seq_length, self.dim_embed)
        # value_ouputs = value_ouputs.reshape(batch_size, 3, self.seq_length, self.dim_embed)

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t

        # get predictions
        # policy_preds = self.policy_head(x[:,2,-1,:])  # predict next actions given state
        # value_preds = self.value_head(x[:,1,-1,:])

        return policy_pred, value_pred

    
    def get_action(self, policy:torch.Tensor):
        return torch.multinomial(policy,1)
    
class TransformerOutput(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        input = input.reshape(input.size()[0],*self.shape)
        return input[:,-1,:]

class View(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)

class MaskedStableSoftmax(nn.Module):
    def __init__(self, eps = 1e5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits,mask):
        if mask.count_nonzero() == 0:
            raise Exception("No playable tile")
        logits = logits - logits.max(dim=-1, keepdim=True).values
        return torch.softmax(logits - self.eps* torch.logical_not(mask),-1) 

class PE3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PE3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = self.get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = self.get_emb(sin_inp_y).unsqueeze(1)
        emb_z = self.get_emb(sin_inp_z)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        self.cached_penc = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return self.cached_penc
    
    def get_emb(self,sin_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)