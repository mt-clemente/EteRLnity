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
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,weight_decay=1e-4)
        
    def get_action(self, policy:torch.Tensor,mask:torch.BoolTensor):

        if mask.count_nonzero() == 0:
            raise Exception("No playable tile")
        sm = torch.softmax(policy,-1) * mask
        sm /= sm.sum(dim=-1, keepdim=True)
        return torch.multinomial(sm,1)
        
    def compute_gae(self, rewards, values, next_values, finals): #FIXME: move to training, to get the good whole trajectory.

        td_errors = rewards + self.gamma * next_values * (1 - finals) - values
        gae = 0
        advantages = torch.zeros_like(td_errors)
        for t in reversed(range(len(td_errors))):
            gae = td_errors[t] + self.gamma * self.gae_lambda * (1 - finals[t]) * gae
            advantages[t] = gae
        return advantages
    
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
            mem['rtg_buf'].to(training_device),
            mem['adv_buf'].to(training_device),
            mem['policy_buf'].to(training_device),
            mem['value_buf'].to(training_device),
            mem['timestep_buf'].to(training_device),
        )

        loader = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)
        

        # Perform multiple update epochs
        for k in range(self.epochs):
            for batch in loader:

                (
                    batch_states,
                    batch_actions_seq,
                    batch_rtg_seq,
                    batch_advantages,
                    batch_old_policies,
                    batch_values,
                    batch_timesteps,
                    
                ) = batch

                has_padding = (batch_actions_seq == -1).any(dim=-1)
                
                # actions taken
                batch_actions = batch_actions_seq[:,-1].clone()
                batch_actions[has_padding] = batch_actions_seq[has_padding,batch_timesteps[has_padding]+1].clone()

                # Remove the actions taken to get the action sequence that was used for inference
                # for sequences with padding
                batch_actions_seq[has_padding,batch_timesteps[has_padding]+1] = -1

                batch_returns = batch_advantages + batch_values

                batch_policy, batch_value = self.model(
                    batch_states,
                    batch_actions_seq[:,:-1],
                    batch_rtg_seq[:,:-1],
                    batch_timesteps,
                )

                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std()+1e-7)
                batch_returns = (batch_returns - batch_returns.mean()) / (batch_returns.std()+1e-7)

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
                    print("--------")
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
                    print("bst",batch_states.max())
                    print("bst",batch_states.min())

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),0.5)
                self.optimizer.step()
                
                g=0
                for name,param in self.model.named_parameters():
                    # print(f"{name:>28} - {torch.norm(param.grad)}")
                    # print(f"{name:>28}")
                    g+=torch.norm(param.grad)

                    
        print(datetime.now()-t0)
        wandb.log({
            "Total loss":loss,
            "Cumul grad norm":g,
            "Value loss":value_loss,
            "Entropy loss":entropy_loss,
            "Policy loss":policy_loss,
            "KL div": (batch_old_policies * (torch.log(batch_old_policies + 1e-8) - torch.log(batch_policy + 1e-8))).sum(dim=-1).mean()
            })

        if not CUDA_ONLY:
            self.model = self.model.cpu()
        mem.reset()
    




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

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.actor_dt =  nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=dim_embed,
                    dim_feedforward=self.hidden_size,
                    nhead=n_heads,
                    batch_first=True,
                    norm_first=True,
                    device=device,
                    dtype=UNIT
                ),
                num_layers=n_layers,
                norm=nn.LayerNorm(dim_embed,device=device)
            )
        

        self.actor_head = nn.Sequential(
            TransformerOutput((3, self.seq_length, dim_embed)),
            nn.Linear(dim_embed, act_dim,device=device,dtype=UNIT),
            SoftmaxStable()
        )
        self.critic_dt =  nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=dim_embed,
                    dim_feedforward=self.hidden_size,
                    nhead=n_heads,
                    batch_first=True,
                    norm_first=True,
                    device=device,
                    dtype=UNIT
                ),
                num_layers=n_layers,
                norm=nn.LayerNorm(dim_embed,device=device)
            )
        

        self.critic_head = nn.Sequential(
            TransformerOutput((3, self.seq_length, dim_embed)),
            nn.Linear(dim_embed, 1,device=device,dtype=UNIT),
        )

        def init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


        self.actor_dt.apply(init_weights)
        self.critic_dt.apply(init_weights)
        # Apply the initialization to the sublayers of the transformer layers


            
        self.dim_embed = dim_embed 
        self.embed_timestep = nn.Embedding(self.n_tiles, dim_embed,device=device,dtype=UNIT)
        self.embed_return = torch.nn.Linear(1, dim_embed,device=device,dtype=UNIT)
        self.embed_actions = torch.nn.Linear(1, dim_embed,device=device,dtype=UNIT)
        self.embed_state = nn.Sequential(
            nn.Linear(4 * COLOR_ENCODING_SIZE,self.dim_embed,device=device),
            PE2D(self.dim_embed),
            View((-1,self.n_tiles,self.dim_embed)),
        )

        self.embed_ln = nn.LayerNorm(dim_embed,device=device,dtype=UNIT)



    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):


        #BOS   

        if states.dim() == 4:

            batch_size = states.shape[0]
            states_ = states
            returns_to_go_ = returns_to_go.unsqueeze(-1)
            actions_ = actions.unsqueeze(-1)
            timesteps_ = timesteps.unsqueeze(-1)
        else:
            batch_size = 1
            states_ = states.unsqueeze(0)
            actions_ = actions.unsqueeze(0)
            returns_to_go_ = returns_to_go.unsqueeze(0)
            timesteps_ = timesteps.unsqueeze(0)

        # embed each modality with a different head

        unpadded_seqs = timesteps_ > self.seq_length
        state_embeddings_full = self.embed_state(states_)
        state_embeddings = state_embeddings_full[:, :self.seq_length, :]
        mask = unpadded_seqs.reshape(states_.size()[0])


        seq_indexes = repeat(torch.arange(self.seq_length,device=states.device),'i -> b i',b = states_.size()[0]).clone()

        if mask.count_nonzero() != 0:
            seq_indexes[mask] += timesteps_[mask] - self.seq_length

        state_embeddings = torch.gather(state_embeddings_full, 1, seq_indexes.unsqueeze(-1).expand(-1, -1, state_embeddings_full.size(2)))

        key_padding_mask = repeat(actions_.squeeze(-1) == -1,'b p -> b (c p)',c=3)


        actions_embeddings = self.embed_actions(actions_.to(UNIT))
        returns_embeddings = self.embed_return(returns_to_go_)
        time_embeddings = self.embed_timestep(timesteps_)

        state_embeddings = state_embeddings + time_embeddings
        actions_embeddings = actions_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, actions_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*self.seq_length, self.dim_embed)

        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well

        stacked_inputs = self.embed_ln(stacked_inputs)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        policy_tokens = self.actor_dt( #FIXME: MASKKK
            memory=torch.zeros_like(stacked_inputs,device=stacked_inputs.device),
            tgt=stacked_inputs,
            tgt_key_padding_mask=key_padding_mask
        )

        value_tokens = self.critic_dt( #FIXME: MASKKK
            memory=torch.zeros_like(stacked_inputs,device=stacked_inputs.device),
            tgt=stacked_inputs,
            tgt_key_padding_mask=key_padding_mask
        )


        policy_pred = self.actor_head(policy_tokens)
        value_pred = self.critic_head(value_tokens)

        # policy_ouputs = policy_ouputs.reshape(batch_size, 3, self.seq_length, self.dim_embed)
        # value_ouputs = value_ouputs.reshape(batch_size, 3, self.seq_length, self.dim_embed)

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t

        # get predictions
        # policy_preds = self.policy_head(x[:,2,-1,:])  # predict next actions given state
        # value_preds = self.value_head(x[:,1,-1,:])


        return policy_pred, value_pred

    def get_action(self, state,policy,return_to_go,timestep):
        # we don't care about the past rewards in this model

        policy_preds, value_preds = self.forward(
            state, policy, return_to_go, timestep)
        
        return policy_preds, value_preds
    
class TransformerOutput(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        input = input.reshape(input.size()[0],*self.shape)
        return input[:,2,-1,:]

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

        return (pos_enc + x)