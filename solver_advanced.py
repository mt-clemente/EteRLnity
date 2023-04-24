from hashlib import sha256
import random
import sys

from Network.Transformer import PPOAgent
from Network.param import CPU_TRAINING, CUDA_ONLY
from Network.train_monoagent import get_conflicts, get_connections
from Network.utils import initialize_sol, place_tile, to_list
from eternity_puzzle import EternityPuzzle
import torch
from datetime import datetime,timedelta
import configs
from parse import parse



def solve_advanced(eternity_puzzle:EternityPuzzle):

    """
    Your solver for the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    if torch.cuda.is_available() and not CUDA_ONLY:
        # agent = agent.cuda() #FIXME:
        device = 'cuda'
    else:
        device = 'cpu' 

    DURATION = timedelta(seconds=3600)
    # -------------------- GAME INIT --------------------
    bsize = eternity_puzzle.board_size

    cfg = {
            'n_tiles' : None,
            'gamma' : None,
            'clip_eps' : None,
            'lr' : 1,
            'epochs' : None,
            'minibatch_size' : None,
            'horizon' : None,
            'state_dim' : None,
            'n_heads':60,
            'dim_embed': 15,
            'hidden_size':32,
            'entropy_weight':None,
            'value_weight':None,
            'gae_lambda':None,
            'device' : None,
            'first_corner':None,
    }

    match bsize:
        case 4:
            eval_model_dir = "models/Inference/A"
        case 7:
            eval_model_dir = "models/Inference/B"
        case 8:
            eval_model_dir = "models/Inference/C"
        case 9:
            eval_model_dir = "models/Inference/D"
        case 10:
            eval_model_dir = "models/Inference/E"

    init_state, tiles, first_corner,n_tiles = initialize_sol(eternity_puzzle,device)


    cfg['n_encoder_layers'], cfg['n_decoder_layers'], cfg['hidden_size'] = infer_model_size(eval_model_dir)

    cfg['first_corner'] = first_corner
    cfg['n_tiles'] = n_tiles
    cfg['state_dim'] = bsize + 2

    load_list = [
        'actor_dt',
        'critic_dt',
        'embed_tiles',
        'embed_state',
    ]

    agent = PPOAgent(
        config=cfg,
        eval_model_dir=eval_model_dir,
        tiles=tiles,
        init_state=init_state,
        load_list=load_list,
        device=device
    ).model

    lds = LimitedDiscrepancy(1e-5,5)


    start = datetime.now()

    best_score = -1
    best_sol = None
    
    while start + DURATION > datetime.now():

        mask = (torch.zeros_like(tiles[:,0]) == 0)
        step = 0
        lds.reset()
        state = init_state
        
        while step != n_tiles - 2:

            t0 = datetime.now()            

            policy, value = agent(
                state,
                torch.tensor(step,device=state.device),
                mask
            )
                    
            selected_tile_idx = lds.get_action(policy,n_tiles-step)
            selected_tile = tiles[selected_tile_idx]
            new_state, reward, _ = place_tile(state,selected_tile,step,step_offset=1)

            mask[selected_tile_idx] = False
            state = new_state
            step += 1

            print(datetime.now() - t0)


        episode_score = get_connections(new_state,bsize)
        lds.step(episode_score)
        print(episode_score)
        if episode_score > best_score:
            best_score = episode_score
            best_sol = new_state
            eternity_puzzle.print_solution(to_list(best_sol,bsize),f"adv_{best_score}")
        

    list_sol = to_list(best_sol,bsize)

    return list_sol



def infer_model_size(model_dir,load_list=None):

    corresp_dict = {
    'actor_dt': {
        'dir':f'{model_dir}/transformers'
        },
    'critic_dt': {
        'dir':f'{model_dir}/transformers'
        },
    'actor_head': {
        'dir':f'{model_dir}/heads'
        },
    'critic_head': {
        'dir':f'{model_dir}/heads'
        },
    'embed_tiles': {
        'dir':f'{model_dir}/embeds'
        },
    'embed_state': {
        'dir':f'{model_dir}/embeds'
        },
    }

    if load_list is None:
        load_list = corresp_dict.keys()

    num_enc_layers = 0
    num_dec_layers = 0
    hidden_size = 0

    for key in load_list:

        state_dict = torch.load(f"{corresp_dict[key]['dir']}/{key}.pt")


        for param in state_dict.keys():

            try:
                ne = int(parse("encoder.layers.{}.{}",param)[0])
                if ne > num_enc_layers:
                    num_enc_layers = ne
            except:
                pass

            try:

                nd = int(parse("decoder.layers.{}.{}",param)[0])

                if nd > num_dec_layers:
                    num_dec_layers = nd
                
            except:
                pass
            
            if 'linear1' in param:
                hidden_size = state_dict[param].size()[0]

    return num_enc_layers + 1, num_dec_layers + 1, hidden_size








class LimitedDiscrepancy:

    def __init__(self,epsilon, stale_period,) -> None:
        self.max_descrepancy = 0
        self.epsilon = epsilon # Every node of the tree becomes available eventually
        self.budget = 0
        self.best_score = 0
        self.counter = 0
        self.stale_period = stale_period

    def get_action(self,policy:torch.Tensor,remaining_steps):

        # Choose amongst best actions
        if random.random() < self.budget / (remaining_steps) ** 0.2:
            best_actions = (policy + self.epsilon).topk(self.budget+1, sorted=True).indices.squeeze(0)
        else:
            best_actions = (policy + self.epsilon).topk(self.budget+1, sorted=True).indices.squeeze(0)

        idx = random.randint(0,len(best_actions)-1)

        # Update budget accordingly
        self.budget -= idx
        return best_actions[idx]
    

    def step(self,episode_score):

        if episode_score <= self.best_score:
            self.counter += 1
        
        else:
            self.best_score = episode_score
            self.counter = 0

        if self.counter >= self.stale_period * (self.max_descrepancy**1.5 + 1) or self.max_descrepancy == 0:
            self.max_descrepancy += 1

    
    def reset(self):
        self.budget = self.max_descrepancy


class TabuList:

    def __init__(self,size,tabu_length) -> None:
        self.size = size
        self.tabu = {}
        self.length = tabu_length

    def push(self,state:torch.Tensor, step:int):
        key = sha256(state.cpu().numpy()).hexdigest()
        self.tabu[key] = step + self.length
    
    def in_tabu(self,state):
        key = sha256(state.cpu().numpy()).hexdigest()
        return key in self.tabu.keys()
    
    def update(self,step:int):
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
