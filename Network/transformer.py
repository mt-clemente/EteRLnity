import copy
from pathlib import Path
import random
from einops import einsum, rearrange
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from Network.param import *
from Network.trajectories import EpisodeBuffer
from torch.utils.data import DataLoader
# -------------------- AGENT --------------------

class PPOAgent(nn.Module):
    def __init__(self,config,tiles = None,init_state = None, eval_model_dir = None, device = None,load_list=None):
        super().__init__()

        self.tiles = tiles

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
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available()  else 'cpu'
        else:
            self.device = device
        n_tiles = config['n_tiles']
        hidden_size = config['hidden_size']
        dim_embed = config['dim_embed']
        n_encoder_layers = config['n_encoder_layers']
        n_decoder_layers = config['n_decoder_layers']
        n_heads = config['n_heads']
        first_corner = config['first_corner']

        self.action_dim = n_tiles - 1

        self.model = DecisionTransformerAC(
            state_dim=self.state_dim,
            act_dim=self.action_dim,
            dim_embed=dim_embed,
            hidden_size=hidden_size,
            n_decoder_layers=n_decoder_layers,
            n_encoder_layers=n_encoder_layers,
            n_heads=n_heads,
            first_corner=first_corner,
            init_state=init_state,
            tiles = tiles,
            device=self.device,
        )
        
        

        self.optimizer = torch.optim.SGD(self.model.parameters(), momentum=0.6, dampening=0, lr=self.lr,weight_decay=0,nesterov=True)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,weight_decay=0,eps=OPT_EPSILON)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=self.optimizer,
            start_factor=0.01,
            end_factor=1,
            total_iters=1e5,
        )
    

        if not eval_model_dir is None:
            self.load_model(eval_model_dir,load_list)
        

        self.workers = nn.ModuleList([copy.deepcopy(self.model) for i in range(NUM_WORKERS)])

        

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
                    batch_masks,
                    batch_advantages,
                    batch_old_policies,
                    batch_returns,
                    batch_timesteps,
                    
                ) = batch

                
                batch_policy, batch_value = self.model(
                    batch_states,
                    batch_timesteps,
                    batch_masks,
                )

                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std()+1e-4)
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
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(),0.5)
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


    
    def save_models(self,dir:str,episode:int):

        Path(dir).mkdir(parents=True, exist_ok=True)
        Path(f"{dir}/transformers").mkdir(parents=True, exist_ok=True)
        Path(f"{dir}/heads").mkdir(parents=True, exist_ok=True)
        Path(f"{dir}/embeds").mkdir(parents=True, exist_ok=True)

        torch.save(self.model.actor_dt.state_dict(), f'{dir}/transformers/actor_{episode}.pt')
        torch.save(self.model.critic_dt.state_dict(), f'{dir}/transformers/critic_{episode}.pt')

        torch.save(self.model.actor_head.state_dict(), f'{dir}/heads/actor_{episode}.pt')
        torch.save(self.model.critic_head.state_dict(), f'{dir}/heads/critic_{episode}.pt')


        torch.save(self.model.embed_tiles.state_dict(), f'{dir}/embeds/tiles_{episode}.pt')
        torch.save(self.model.embed_state.state_dict(), f'{dir}/embeds/state_{episode}.pt')


    def load_model(self,model_dir, load_list = None):
        """
        Loads a model depending on the given loading dictionary:

        Keys
        -----------------
        * actor_dt : the actor transformer
        * critic_dt : the critic transformer
        * actor_head : the actor head
        * critic_head : the critic head
        * embed_tiles : action color embedding
        * embed_state : state color embedding
        If no keys are given, the whole model is loaded.
        """

        corresp_dict : dict[nn.Module] = {
            'actor_dt': {
                'model':self.model.actor_dt,
                'dir':f'{model_dir}/transformers'
                },
            'critic_dt': {
                'model':self.model.critic_dt,
                'dir':f'{model_dir}/transformers'
                },
            'actor_head': {
                'model':self.model.actor_head,
                'dir':f'{model_dir}/heads'
                },
            'critic_head': {
                'model':self.model.critic_head,
                'dir':f'{model_dir}/heads'
                },
            'embed_tiles': {
                'model':self.model.embed_tiles,
                'dir':f'{model_dir}/embeds'
                },
            'embed_state': {
                'model':self.model.embed_state,
                'dir':f'{model_dir}/embeds'
                },
        }

        if load_list is None:
            load_list = corresp_dict.keys()


        for key in load_list:
            print(key)
            state_dict = torch.load(f"{corresp_dict[key]['dir']}/{key}.pt")
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
            first_corner,
            init_state,
            tiles,
            device,
    ):
        super().__init__()
        self.tiles = tiles
        self.rolled_tiles = [tiles.roll(i + 1,-1) for i in range(3)]
        self.state_dim = state_dim
        self.bsize = state_dim - 2
        self.act_dim = act_dim
        self.hidden_size = hidden_size


        self.ep_buf = EpisodeBuffer(
            ep_len=self.act_dim,
            bsize=state_dim,
            horizon=HORIZON,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            init_state=init_state,
            seq_len=act_dim,
            first_corner=first_corner,
            device=device
        )


        self.actor_dt =  Transformer(
                d_model=dim_embed*4,
                num_encoder_layers=n_encoder_layers,
                num_decoder_layers=n_decoder_layers,
                nhead=n_heads,
                dim_feedforward=self.hidden_size,
                dropout=0,
                activation=F.gelu,
                batch_first=True,
                norm_first=True,
                return_mem=POINTER,
                device=device,
            )
        
        self.critic_dt =  nn.Transformer(
                d_model=dim_embed*4,
                num_encoder_layers=n_encoder_layers,
                num_decoder_layers=n_decoder_layers,
                dim_feedforward=self.hidden_size,
                nhead=n_heads,
                dropout=0,
                activation=F.gelu,
                batch_first=True,
                norm_first=True,
                device=device,
                dtype=UNIT,
            )
        

        if POINTER:
            self.policy_attn_head = Pointer(
                dim_embed*4,
                device,
                UNIT
                )

        else:
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
        self.dim_embed = dim_embed 
        self.embed_tiles = torch.nn.Embedding(N_COLORS + 2, dim_embed,device=device,dtype=UNIT) # NCOLLORS, BOS, PAD
        self.embed_state = nn.Embedding(N_COLORS,dim_embed,device=device,dtype=UNIT)
        self.positional_encoding = PositionalEncoding2D(dim_embed*4)
        self.embed_state_ln = nn.LayerNorm(dim_embed*4,eps=1e-5,device=device,dtype=UNIT)
        self.embed_tile_ln = nn.LayerNorm(dim_embed*4,eps=1e-5,device=device,dtype=UNIT)



    def forward(self, states, timesteps, tile_mask, guide=True, attention_mask=None):


        #BOS   


        if states.dim() == 4:

            batch_size = states.shape[0]
            states_ = states
            timesteps_ = timesteps.unsqueeze(-1)
        else:
            batch_size = 1
            states_ = states.unsqueeze(0)
            timesteps_ = timesteps.unsqueeze(0)

        # embed each modality with a different head
        tiles_embeddings = self.embed_tiles(self.tiles.int()).reshape(self.act_dim,self.dim_embed*4)
        tiles_embeddings = tiles_embeddings.expand(batch_size,-1,-1)

        state_embeddings = self.embed_state(states_[:,1:-1,1:-1,:].int())
        state_embeddings = state_embeddings.view(-1,self.bsize,self.bsize,self.dim_embed*4)
        state_embeddings += self.positional_encoding(state_embeddings)
        state_embeddings = state_embeddings.view(-1,self.bsize**2,self.dim_embed*4)

        src_inputs = self.embed_tile_ln(tiles_embeddings)
        tgt_inputs = self.embed_state_ln(state_embeddings)

        tgt_key_padding_mask = torch.arange(self.act_dim+1,device=timesteps_.device).repeat(batch_size,1) > timesteps_
        # tgt_key_padding_mask = torch.repeat_interleave(tgt_key_padding_mask_,4,dim=-1)


        #TODO: Cleanup mask usage
        if guide and random.random() < GUIDE_PROB:# and random.random < GUIDE_PROB:
            tile_mask_ = self.guide_action(states_,timesteps_+1,tile_mask)
            # tile_mask = torch.repeat_interleave(tile_mask_,4,dim=-1)
        else:
            tile_mask_ = tile_mask
            if batch_size == 1:
                tile_mask_ = tile_mask.unsqueeze(0)
                # tile_mask = torch.repeat_interleave(tile_mask_,4,dim=-1)
            else:
                tile_mask_ = tile_mask
                # tile_mask = torch.repeat_interleave(tile_mask_,4,dim=-1)

        tile_mask = tile_mask_ #FIXME:
        causal_mask = torch.triu(torch.ones(tgt_inputs.size()[1], tgt_inputs.size()[1],device=tgt_inputs.device), diagonal=1)
        # Convert the mask to a boolean tensor with 'True' values below the diagonal and 'False' values on and above the diagonal
        causal_mask = causal_mask.bool()

        if POINTER:
            tgt_tokens, mem_tokens = self.actor_dt(
                src=src_inputs,
                tgt=tgt_inputs,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            tgt_tokens = tgt_tokens.reshape(batch_size,self.bsize**2,self.dim_embed*4)
            tgt_tokens = tgt_tokens[torch.arange(batch_size,device=tgt_tokens.device),timesteps+1].reshape(batch_size,self.dim_embed*4)
            mem_tokens = mem_tokens.reshape(batch_size,self.act_dim,self.dim_embed*4)
            policy_pred = self.policy_attn_head(mem_tokens,tgt_tokens,torch.logical_not(tile_mask_))
        
        else:
            policy_tokens = self.actor_dt(
                src=src_inputs,
                tgt=tgt_inputs,
                tgt_key_padding_mask=tgt_key_padding_mask
            ).reshape(batch_size,self.bsize**2,self.dim_embed*4)

            policy_logits = self.actor_head(policy_tokens[torch.arange(batch_size,device=policy_tokens.device),timesteps+1].reshape(batch_size,self.dim_embed*4))
            policy_pred = self.policy_head(policy_logits,tile_mask_)

        value_tokens = self.critic_dt(
            src=src_inputs,
            tgt=tgt_inputs,
            tgt_key_padding_mask=tgt_key_padding_mask
        ).reshape(batch_size,self.bsize**2,self.dim_embed*4)

        value_pred = self.critic_head(value_tokens[torch.arange(batch_size,device=value_tokens.device),timesteps+1].reshape(batch_size,self.dim_embed*4))


        return policy_pred, value_pred

    
    def get_action(self, policy:torch.Tensor):
        return torch.multinomial(policy,1)
    


    def guide_action(self, states, pos1d, mask):
        """
        Only implemented for ordinal encoding for now
        """

        i = (pos1d // self.bsize).squeeze(-1) + 1
        j = (pos1d % self.bsize).squeeze(-1) + 1

        colors = torch.zeros((4,pos1d.size()[0]),device=states.device)
        #NORTH COLORS
        colors[0] = states[torch.arange(states.size()[0]),i+1,j,2]
        #WEST COLORS
        colors[1] = states[torch.arange(states.size()[0]),i,j-1,3]
        #SOUTH COLORS
        colors[2] = states[torch.arange(states.size()[0]),i-1,j,0]
        #WEST COLORS
        colors[3] = states[torch.arange(states.size()[0]),i,j+1,1]

        south = (colors[2].unsqueeze(-1).unsqueeze(-1) == self.tiles.expand(colors.size()[1],-1,-1))
        west = (colors[1].unsqueeze(-1).unsqueeze(-1) == self.rolled_tiles[0].expand(colors.size()[1],-1,-1))

        north = (colors[0].unsqueeze(-1).unsqueeze(-1) == self.rolled_tiles[1].expand(colors.size()[1],-1,-1)) * (i == self.bsize).unsqueeze(-1).unsqueeze(-1) + (i != self.bsize).unsqueeze(-1).unsqueeze(-1)

        east = (colors[3].unsqueeze(-1).unsqueeze(-1) == self.rolled_tiles[2].expand(colors.size()[1],-1,-1)) * (j == self.bsize).unsqueeze(-1).unsqueeze(-1) + (j != self.bsize).unsqueeze(-1).unsqueeze(-1)
        

        consecutive_presence = (south & west & north & east).any(dim=-1)


        new_mask = torch.logical_and(mask,consecutive_presence)

        if states.size()[0] == 1:
            mask = mask.unsqueeze(0)

        new_mask[torch.logical_not(new_mask.any(dim=-1))] = mask[torch.logical_not(new_mask.any(dim=-1))]

        return new_mask


class Transformer(nn.Module):

    #TODO: Cache encoder output

    def __init__(self, d_model,num_encoder_layers,num_decoder_layers,dim_feedforward,nhead,activation,device,batch_first,norm_first,return_mem=True,dropout=0) -> None:
        super().__init__()

        self.transformer = nn.Transformer(
                d_model=d_model,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                nhead=nhead,
                dropout=dropout,
                activation=activation,
                batch_first=batch_first,
                norm_first=norm_first,
                device=device,
                dtype=UNIT,
        )
        self.return_mem = return_mem


    def forward(self,src,tgt,tgt_mask,src_key_padding_mask=None,tgt_key_padding_mask=None):


        memory = self.transformer.encoder(src,src_key_padding_mask=src_key_padding_mask)
        output = self.transformer.decoder(tgt,memory,tgt_mask=tgt_mask.half()*(-1e6), memory_key_padding_mask=src_key_padding_mask,tgt_key_padding_mask=tgt_key_padding_mask.half()*-1e6)

        if self.return_mem:
            return output, memory
        
        return output
    
class Pointer(nn.Module):

    def __init__(self, d_model, device, unit):
        super().__init__()
        self.d_model = d_model

        self.Wq = nn.Linear(d_model,d_model, device=device, dtype=unit,bias=False)
        self.Wk = nn.Linear(d_model,d_model, device=device, dtype=unit,bias=False)
        self.v = nn.Linear(d_model, 1, device=device, dtype=unit,bias=False)

        torch.nn.init.normal_(self.v.weight,0,0.01)

        nn.MultiheadAttention
    def forward(self, memory:torch.Tensor, target:torch.Tensor, memory_mask:torch.BoolTensor):
        q = self.Wq(target).unsqueeze(1)
        k = self.Wk(memory)
        out = self.v(torch.tanh(q + k)).squeeze(-1)
        probs = F.softmax(out - 1e9 * memory_mask,dim=-1)
        return probs

class DotAttentionPointer(nn.Module):

    def __init__(self, d_model, device, unit):
        super().__init__()
        self.d_model = d_model

        self.Wq = nn.Sequential(
            nn.Linear(d_model,128, device=device, dtype=unit),
            nn.GELU(),
            nn.Linear(128, d_model, device=device, dtype=unit)
        )
        self.Wk = nn.Sequential(
            nn.Linear(d_model,128, device=device, dtype=unit),
            nn.GELU(),
            nn.Linear(128, d_model, device=device, dtype=unit)
        )

        nn.MultiheadAttention
    def forward(self, nodes:torch.Tensor, tokens:torch.Tensor, node_mask:torch.BoolTensor, attn_mask:torch.BoolTensor):
        q = self.Wq(tokens) * attn_mask.unsqueeze(-1)
        k = self.Wk(nodes)
        out = k @ rearrange(q,'b k d -> b d k')
        probs = F.softmax((out - 1e9 * node_mask.unsqueeze(-1)).sum(dim=-1) / self.d_model**0.5,dim=-1)
        return probs

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
    
def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)
    






class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc