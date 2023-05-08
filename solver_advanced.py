import random
from Network.transformer import PPOAgent
from Network.utils import get_conflicts, initialize_sol, place_tile, to_list, get_connections
from eternity_puzzle import EternityPuzzle
import torch
from datetime import datetime,timedelta
from parse import parse



def solve_advanced(eternity_puzzle:EternityPuzzle):

    """
    Your solver for the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu' 

    DURATION = timedelta(minutes=15)
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
        case 16:
            eval_model_dir = "models/Inference/F"

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
        
        while step != n_tiles - 1:

            policy, value = agent(
                state,
                torch.tensor(step,device=state.device),
                mask
            )
                    
            selected_tile_idx = lds.get_action(policy,n_tiles - step)
            selected_tile = tiles[selected_tile_idx]
            new_state, reward, _ = place_tile(state,selected_tile,step,step_offset=1)

            mask[selected_tile_idx] = False
            state = new_state
            step += 1



        episode_score = get_connections(new_state,bsize)
        lds.step(episode_score)
        print(episode_score)
        if episode_score > best_score:
            best_score = episode_score
            best_sol = new_state
            eternity_puzzle.print_solution(to_list(best_sol,bsize),f"adv_{best_score}")
        

    list_sol = to_list(best_sol,bsize)
    eternity_puzzle.display_solution(list_sol,'outoutout')

    return list_sol, get_conflicts(best_sol,bsize)



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
        budget = min(self.budget,policy.size()[1]-1)
        if random.random() < self.budget / (remaining_steps) ** 0.2:
            best_actions = (policy + self.epsilon).topk(budget+1, sorted=True).indices.squeeze(0)
        else:
            best_actions = (policy + self.epsilon).topk(1, sorted=True).indices.squeeze(0)

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

