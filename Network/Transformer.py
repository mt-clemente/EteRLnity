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
            device=self.device,
            )
        
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr,weight_decay=1e-4)
        
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
        

        # device = 'cuda'
        t0 = datetime.now()

        if torch.cuda.is_available():
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

        loader = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=False)
        

        # Perform multiple update epochs
        for k in range(self.epochs):
            for batch in loader:

                (
                    batch_states_seq,
                    batch_actions_seq,
                    batch_rtg_seq,
                    batch_advantages,
                    batch_old_policies,
                    batch_values,
                    batch_timesteps,
                    
                ) = batch


                # end_rtg = batch_rtg_seq[torch.arange(batch_rtg_seq.size()[0]),batch_timesteps]
                # end_minus1_rtg = batch_rtg_seq[torch.arange(batch_rtg_seq.size()[0]),batch_timesteps+1]
                # last_seq_rewards =  end_rtg - end_minus1_rtg

                batch_returns = batch_advantages + batch_values

                batch_policy, batch_value = self.model(
                    batch_states_seq,
                    batch_actions_seq[:,:-1],
                    batch_rtg_seq[:,:-1],
                    batch_timesteps,
                )


                batch_actions = batch_actions_seq[torch.arange(batch_timesteps.size()[0]),batch_timesteps]

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
                    print("bst",batch_states_seq.max())
                    print("bst",batch_states_seq.min())

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),1)
                self.optimizer.step()
                
                g=0
                for name,param in self.model.named_parameters():
                    # print(f"{name:>28} - {torch.norm(param.grad)}")
                    # print(f"{name:>28}")
                    g+=torch.norm(param.grad)

                    

        wandb.log({
            "Total loss":loss,
            "Cumul grad norm":g,
            "Value loss":value_loss,
            "Entropy loss":entropy_loss,
            "Policy loss":policy_loss,
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
            max_length=None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.seq_length = (state_dim-2)**2

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=dim_embed,
                dim_feedforward=self.hidden_size,
                nhead=n_heads,
                batch_first=True,
                device=device,
                dtype=UNIT
            ),
            num_layers=n_layers,
            norm=nn.LayerNorm(dim_embed)
        ) 



        self.dim_embed = dim_embed 
        self.embed_timestep = nn.Embedding(self.seq_length, dim_embed,device=device,dtype=UNIT)
        self.embed_return = torch.nn.Linear(1, dim_embed,device=device,dtype=UNIT)
        self.embed_actions = torch.nn.Linear(1, dim_embed,device=device,dtype=UNIT)
        self.embed_state = nn.Sequential(
            nn.Linear(4 * COLOR_ENCODING_SIZE,self.dim_embed,device=device),
            PE2D(self.dim_embed),
            View((-1,self.seq_length,self.dim_embed)),
        )

        self.embed_ln = nn.LayerNorm(dim_embed,device=device,dtype=UNIT)

        self.policy_head = nn.Sequential(
            nn.Linear(dim_embed, act_dim,device=device,dtype=UNIT),
            SoftmaxStable()
        )

        self.value_head = nn.Sequential(
            nn.Linear(dim_embed, 1,device=device,dtype=UNIT)
        )


    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):


        #BOS

        if states.dim() == 4:
            action_bos = torch.full((actions.size(0), 1), -2,device=actions.device)

            actions = torch.cat((action_bos, actions), dim=1)
            batch_size = states.shape[0]
            states_ = states
            returns_to_go_ = returns_to_go.unsqueeze(-1)
            actions_ = actions.unsqueeze(-1)
            timesteps_ = timesteps.unsqueeze(-1)
        else:
            batch_size = 1
            actions = torch.vstack((torch.tensor(-2,device=actions.device),actions))      
            states_ = states.unsqueeze(0)
            actions_ = actions.unsqueeze(0)
            returns_to_go_ = returns_to_go.unsqueeze(0)
            timesteps_ = timesteps.unsqueeze(0)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states_)
    
        actions_embeddings = self.embed_actions(actions_.to(UNIT))
        returns_embeddings = self.embed_return(returns_to_go_)
        time_embeddings = self.embed_timestep(timesteps_)

        state_embeddings = state_embeddings + time_embeddings
        actions_embeddings = actions_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, actions_embeddings), dim=1
        )
        
        stacked_inputs = rearrange(stacked_inputs,'b c s e -> b (c s) e')
        stacked_inputs = stacked_inputs.reshape(batch_size, 3*self.seq_length, self.dim_embed)

            
        stacked_inputs = self.embed_ln(stacked_inputs)

        mask = torch.arange(self.seq_length).expand(len(timesteps_), self.seq_length).to(timesteps.device)
        mask = mask > timesteps.unsqueeze(-1)
        mask = repeat(mask,'b s-> b (s k)',k=3)
        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            memory=torch.zeros_like(stacked_inputs),
            tgt=stacked_inputs,
            tgt_key_padding_mask=mask
        )

        x = transformer_outputs.reshape(batch_size, 3, self.seq_length, self.dim_embed)

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        # x =

        # get predictions

        policy_preds = self.policy_head(x[:,2,-1,:])  # predict next actions given state
        value_preds = self.value_head(x[:,2,-1,:])


        return policy_preds, value_preds

    def get_action(self, state,policy,return_to_go,timestep):
        # we don't care about the past rewards in this model

        policy_preds, value_preds = self.forward(
            state, policy, return_to_go, timestep)
        
        return policy_preds, value_preds
    

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