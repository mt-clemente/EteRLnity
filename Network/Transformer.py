import math
from matplotlib import pyplot as plt
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
# -------------------- AGENT --------------------
class PPOAgent:
    def __init__(self,config,tiles):
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
        n_layers = config['n_layers']
        n_heads = config['n_heads']


        self.action_dim = n_tiles
        self.model = DecisionTransformerAC(
            tiles = tiles,
            state_dim=self.state_dim,
            act_dim=self.action_dim,
            dim_embed=dim_embed,
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_heads=n_heads,
            max_length=self.seq_len,
            device=self.device,
            )
        

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,weight_decay=1e-4,eps=OPT_EPSILON)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=self.optimizer,
            start_factor=0.001,
            end_factor=1.0,
            total_iters=20000,
        )

    def get_action(self, policy:torch.Tensor,mask:torch.BoolTensor):

        if mask.count_nonzero() == 0:
            raise Exception("No playable tile")
        sm = torch.softmax(policy,-1) * mask
        sm /= sm.sum(dim=-1, keepdim=True)
        return torch.multinomial(sm,1)
        

    # def update(self, states, actions, old_policies, values, advantages, returns):
    def update(self,mem:BatchMemory):
        """
        Updates the ppo agent, using the trajectories in the memory buffer.
        For states, policy, rewards, advantages, and timesteps the data is in a 
        straightforward format [batch,*values]
        For the returns-to-go and actions the data has a format [batch,sequence_len+1].
        We need the sequence coming before the state to make a prediction, and the current
        action to calculate the policy and ultimately the policy loss.

        """


        t0 = datetime.now()

        if torch.cuda.is_available() and not CPU_TRAINING:
            self.model = self.model.cuda()
            training_device = 'cuda'
        else:
            training_device = 'cpu'


        dataset = TensorDataset(
            mem['state_buf'].to(training_device),
            mem['act_buf'].to(training_device),
            mem['mask_buf'].to(training_device),
            mem['adv_buf'].to(training_device),
            mem['policy_buf'].to(training_device),
            mem['value_buf'].to(training_device),
            mem['timestep_buf'].to(training_device),
        )

        if (self.horizon) % MINIBATCH_SIZE < MINIBATCH_SIZE / 2 and self.horizon % MINIBATCH_SIZE != 0:
            print("dropping last ",(self.horizon) % MINIBATCH_SIZE)
            drop_last = True
        else:
            drop_last = False


        loader = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=False, drop_last=drop_last)
        

        # Perform multiple update epochs
        for k in range(self.epochs):
            for batch in loader:

                (
                    batch_states,
                    batch_actions,
                    batch_masks,
                    batch_advantages,
                    batch_old_policies,
                    batch_values,
                    batch_timesteps,
                    
                ) = batch

                batch_returns = batch_advantages + batch_values

                print(batch_states.size())
                print(batch_actions.size())
                batch_policy, batch_value = self.model(
                    batch_states,
                    batch_masks,
                    batch_timesteps,
                )

                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std()+1e-4)
                batch_returns = (batch_returns - batch_returns.mean()) / (batch_returns.std()+1e-4)

                # Calculate ratios and surrogates for PPO loss
                action_probs = batch_policy.gather(1, batch_actions.unsqueeze(1))
                old_action_probs = batch_old_policies.gather(1, batch_actions.unsqueeze(1))
                ratio = action_probs / (old_action_probs + 1e-4)
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),0.5)
                self.optimizer.step()
                self.scheduler.step()
                g=0
                for name,param in self.model.named_parameters():
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
            "KL div": (batch_old_policies * (torch.log(batch_old_policies + 1e-8) - torch.log(batch_policy + 1e-8))).sum(dim=-1).mean()
            })

        if not CUDA_ONLY:
            self.model = self.model.cpu()
    

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
            n_layers,
            n_heads,
            tiles,
            device,
            max_length,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_tiles = (state_dim-2)**2
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.seq_length = max_length
        self.tiles = tiles

        
        self.actor_dt =  nn.Transformer(
                d_model=dim_embed*4,
                num_decoder_layers=N_LAYERS,
                num_encoder_layers=N_DECODE_LAYERS,
                dim_feedforward=self.hidden_size,
                dropout=0.05,
                activation=F.gelu,
                batch_first=True,
                norm_first=True,
                device=device,
                dtype=UNIT,
            )
        
        self.critic_dt =  nn.Transformer(
                d_model=dim_embed*4,
                num_decoder_layers=N_LAYERS,
                num_encoder_layers=N_DECODE_LAYERS,
                dim_feedforward=self.hidden_size,
                dropout=0.05,
                activation=F.gelu,
                batch_first=True,
                norm_first=True,
                device=device,
                dtype=UNIT,
            )
        
        self.detokenize_policy = nn.Linear(4 * dim_embed,1, device=device, dtype=UNIT)
        self.detokenize_value = nn.Linear(4 * dim_embed,1, device=device, dtype=UNIT)

        self.actor_head = nn.Sequential(
            nn.Linear(self.seq_length, act_dim,device=device,dtype=UNIT),
            SoftmaxStable()
        )
        self.critic_head = nn.Sequential(
            nn.Linear(self.seq_length, 1,device=device,dtype=UNIT),
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

        wandb.watch(self.actor_dt,log='all',log_freq=50)
        wandb.watch(self.actor_head,log='all',log_freq=50)
        wandb.watch(self.critic_dt,log='all',log_freq=50)
        wandb.watch(self.critic_head,log='all',log_freq=50)
            
        self.dim_embed = dim_embed
        self.pos_encoding = PE2D(self.dim_embed * 4,remove_padding=False)
        self.embed_actions = nn.Embedding(self.n_tiles, dim_embed,device=device,dtype=UNIT)
        self.embed_state = nn.Embedding(self.n_tiles,dim_embed,device=device,dtype=UNIT)
        # self.embed_time = nn.Embedding(1,4 * dim_embed,device=device,dtype=UNIT)



    def forward(self, states, tile_mask, timesteps):


        if states.dim() == 4:

            batch_size = states.shape[0]
            states_ = states
            tiles_ = self.tiles
            timesteps_ = timesteps.unsqueeze(0)
            tile_mask = torch.logical_not(tile_mask)
        else:
            batch_size = 1
            states_ = states.unsqueeze(0)
            tiles_ = self.tiles.unsqueeze(0)
            timesteps_ = timesteps.unsqueeze(0)
            tile_mask = torch.logical_not(tile_mask.unsqueeze(0))

        tile_embeddings = self.embed_actions(tiles_)
        state_embeddings = self.embed_state(states_[:,1:-1,1:-1,:].int()).view(-1,self.n_tiles,self.dim_embed*4)
        state_embeddings = self.pos_encoding(state_embeddings)
        tile_embeddings = tile_embeddings.reshape(self.n_tiles,self.dim_embed * 4)

        tgt_input = tile_embeddings.expand(batch_size,self.n_tiles,self.dim_embed * 4)

        policy_tokens = self.actor_dt(
            src=state_embeddings,
            tgt=tgt_input,
            tgt_key_padding_mask=tile_mask
        )

        value_tokens = self.critic_dt(
            src=state_embeddings,
            tgt=tgt_input,
            tgt_key_padding_mask=tile_mask
        )

        policy_preds = self.detokenize_policy(policy_tokens)
        value_preds = self.detokenize_value(value_tokens)

        policy = self.actor_head(policy_preds.squeeze(-1))
        value = self.critic_head(value_preds.squeeze(-1))

        return policy, value

    def get_action(self, state,tiles, tile_mask,timestep):
        # we don't care about the past rewards in this model

        policy_preds, value_preds = self.forward(
            state, tiles, tile_mask, timestep)
        
        return policy_preds, value_preds
    
class TransformerOutput(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        input = input.reshape(input.size()[0],*self.shape)
        return input[:,1,-1,:]

class View(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)

class SoftmaxStable(nn.Module):
    def forward(self, x):
        x = x - x.max(dim=-1, keepdim=True).values
        return F.softmax(x, dim=-1)

class PE2D(nn.Module):

    def __init__(self, d_model, remove_padding=True) -> None:
        super().__init__()
        self.d_model = d_model
        self.remove_padding = remove_padding

    def forward(self, x:torch.Tensor) -> torch.Tensor:

        """
        Generate a 2D sinusoidal positional encoding for an input of size
        [...,row,column,features]
        
        Args:
            rows (int): Number of rows in the 2D grid.
            cols (int): Number of columns in the 2D grid.
            d_model (int): Dimension of the encoding vector.
            
        Returns:
            pos_enc (torch.Tensor): The 2D positional encoding of shape (rows, cols, d_model).
        """

        if self.remove_padding:
            #batched
            if x.dim() == 5:
                x = x[:,:,1:-1,1:-1,:]
            else:
                x = x[:,1:-1,1:-1,:]

        rows, cols = x.size()[-3:-1]

        assert self.d_model % 2 == 0, "self.d_model should be even."

        # Generate row and column coordinates
        row_coords, col_coords = torch.meshgrid(torch.arange(rows), torch.arange(cols),indexing='ij')
        
        # Expand to match the required encoding dimensions
        row_coords = row_coords.unsqueeze(-1).repeat(1, 1, self.d_model // 2)
        col_coords = col_coords.unsqueeze(-1).repeat(1, 1, self.d_model // 2)

        # Calculate the encoding frequencies for rows and columns
        row_freqs = torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model)
        col_freqs = torch.arange(1, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model)
        
        # Calculate the sinusoidal encodings for row and column coordinates
        row_encodings = torch.sin(row_coords * torch.exp(row_freqs))
        col_encodings = torch.cos(col_coords * torch.exp(col_freqs))
        
        # Combine row and column encodings into a single tensor
        pos_enc = torch.stack((row_encodings, col_encodings), dim=3).view(rows, cols, self.d_model).to(x.device)

        return (pos_enc.to(UNIT) + x)