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
        n_layers = config['n_layers']
        n_heads = config['n_heads']

        self.action_dim = n_tiles
        self.model = DecisionTransformerAC(
            state_dim=self.state_dim,
            act_dim=self.action_dim,
            dim_embed=dim_embed,
            hidden_size=hidden_size,
            n_layers=n_layers,
            n_heads=n_heads,
            max_length=self.seq_len,
            device=self.device,
            )
        

        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr,weight_decay=1e-4,eps=OPT_EPSILON)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=self.optimizer,
            start_factor=0.01,
            end_factor=1,
            total_iters=10000,
        )

    def get_action(self, policy:torch.Tensor):
        return torch.multinomial(policy,1)
        

    # def update(self, states, actions, old_policies, values, advantages, returns):
    def update(self,mem:EpisodeBuffer):
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
            mem.state_buf.to(training_device),
            mem.act_buf.to(training_device),#BOS action is not 'taken'
            mem.tile_seq.to(training_device),#BOS action is not 'taken'
            mem.mask_buf.to(training_device),#BOS action is not 'taken'
            mem.adv_buf.to(training_device),
            mem.policy_buf.to(training_device),
            mem.rtg_buf.to(training_device), # Returns to go for the whole episode
            mem.timestep_buf.to(training_device),
        )

        if (self.horizon) % MINIBATCH_SIZE < MINIBATCH_SIZE / 2 and self.horizon % MINIBATCH_SIZE != 0:
            print("dropping last ",(self.horizon) % MINIBATCH_SIZE)
            drop_last = True
        else:
            drop_last = False

        wandb.log({
            'Advantages':mem.adv_buf.to(training_device),
            'Returns to go':mem.rtg_buf.to(training_device),
        })

        loader = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=False, drop_last=drop_last)
        

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
            "values":batch_value.squeeze(-1).detach(),
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
            device,
            max_length,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_tiles = (state_dim-2)**2
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.seq_length = max_length


        self.actor_dt =  nn.Transformer(
                d_model=4*dim_embed,
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
                d_model=4*dim_embed,
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
        self.positional_encoding = PE2D(4*dim_embed,remove_padding=False)
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
        state_embeddings = self.embed_state(states_[:,1:-1,1:-1,:].int()).view(-1,self.n_tiles,self.dim_embed * 4)
        state_embeddings = self.positional_encoding(state_embeddings)

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

        policy_logits = self.actor_head(policy_tokens[torch.arange(batch_size,device=policy_tokens.device),timesteps+1].reshape(batch_size,self.dim_embed*4))
        policy_pred = self.policy_head(policy_logits,tile_mask)
        value_pred = self.critic_head(value_tokens[torch.arange(batch_size,device=value_tokens.device),timesteps+1].reshape(batch_size,self.dim_embed*4))

        # policy_ouputs = policy_ouputs.reshape(batch_size, 3, self.seq_length, self.dim_embed)
        # value_ouputs = value_ouputs.reshape(batch_size, 3, self.seq_length, self.dim_embed)

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t

        # get predictions
        # policy_preds = self.policy_head(x[:,2,-1,:])  # predict next actions given state
        # value_preds = self.value_head(x[:,1,-1,:])

        return policy_pred, value_pred

    def get_action(self, state,actions,timestep,mask):
        # we don't care about the past rewards in this model

        policy_preds, value_preds = self.forward(
            state, actions, timestep, mask)
        
        return policy_preds, value_preds
    
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
        pos_enc = torch.stack((row_encodings, col_encodings), dim=3).view(rows, cols, self.d_model).to(x.device).to(x.dtype)

        return (pos_enc + x)