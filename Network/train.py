from collections import namedtuple

from einops import repeat, rearrange
import torch
from torch import Tensor
from DRQN import *
from math import comb
from pytorch_memlab import MemReporter
from utils import *
from param import *
import wandb

LOG_EVERY = 600

def train_model(hotstart:str = None):

    """
    Your solver for the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    args = parse_arguments()
    pz = EternityPuzzle(args.instance)
    n_tiles = len(pz.piece_list) * 4
    bsize = pz.board_size
    hotstart = args.hotstart
    # torch.cuda.is_available = lambda : False
    
    # -------------------- GAME INIT --------------------

    # -------------------- NETWORK INIT -------------------- 

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    Transition = namedtuple('Transition',
                            ('state',
                            'next_state',
                            'reward',
                            'action',
                            'final',
                            'weights',
                            'indices'
                            )
                           )


    cfg = {
    'n_tiles' : n_tiles ,
    'gamma' : GAMMA ,
    'clip_ratio' : CLIP_RATIO ,
    'target_kl' : 1,
    'lr' : LR ,
    'epochs' : EPOCHS ,
    'batch_size' : BATCH_SIZE ,
    'state_dim' : bsize+2,
    'hidden_sizes': HIDDEN_SIZES,
    'kernel_sizes': KERNEL_SIZES,
    'device' : device 
    }


    agent = PPOAgent(cfg)

    move_buffer = AdvantageBuffer()
    
    ucb_count = torch.zeros(len(pz.piece_list)*4,device=device)

    memory = BatchMemory(
        n_tiles=n_tiles,
        ep_length=bsize**2
    )

    ep_buf = EpisodeBuffer(
        capacity=BATCH_SIZE,
        n_tiles=n_tiles
    )

    max_entropy = torch.log2(torch.tensor(MAX_BSIZE**2))
    policy_entropy = max_entropy

    # -------------------- TRAINING LOOP --------------------

    torch.cuda.empty_cache()

    step = 0

    print("start")

    best_score = 0
    episode = 0

    try:
        
        while 1:

            state, remaining_tiles, n_tiles = initialize_sol(args.instance,device)
            state = state.to(dtype=UNIT)
            mask = (torch.zeros_like(remaining_tiles[:,0]) == 0)
            episode_end = False
            prev_conflicts = get_conflicts(state,bsize)

            print(f"NEW EPISODE : - {episode:>5}")
            conflicts = 0
            ep_reward = 0
            ep_good_moves = 0
            episode_best_score = 0
            ep_start = step
            ep_step = 0

            if episode != 0:
                memory.load(ep_buf)

            while not episode_end:

                if step % 1 == 0:
                    print(f"{step} - {conflicts} - {policy_entropy}",end='\r')


                current_cell = torch.tensor([ep_step // bsize + 1, ep_step % bsize + 1],device=device)

                with torch.no_grad():
                    policy, value = agent.model(state.unsqueeze(0))

                selected_tile_idx = agent.get_action(policy)
                
                selected_tile = remaining_tiles[selected_tile_idx]

                idx = selected_tile_idx % n_tiles
                for dir in range(4):
                    mask[dir * 256 +idx] = False

                new_state = place_tile(state,selected_tile,ep_step)

                conflicts = get_conflicts(new_state,bsize)

                reward = (2 + prev_conflicts - conflicts)


                if ep_step != 0:
                    ep_buf.push(
                        reward,
                        value,
                    )


                # end on either max conflicts or trajectory does not lead to anything
                if ep_step == 255:

                    episode += 1
                    episode_end = True

                    ep_buf.push(
                        state = move_buffer.state,
                        action = move_buffer.action,
                        policy = move_buffer.policy,
                        value = move_buffer.value,
                        next_value = value,
                        reward = move_buffer.reward,
                        final=True
                    )

                    print(f"END EPISODE {episode} - Conflicts {conflicts}/543")

                    if step % LOG_EVERY == 0:
                        wandb.log({"Actuator episode reward":ep_reward/(step - ep_start + 1e-5),'Mean good moves per ep':ep_good_moves/(step - ep_start + 1e-5)})


                
                prev_conflicts = conflicts
                move_buffer.state = state 
                move_buffer.action = selected_tile_idx 
                move_buffer.policy = policy 
                move_buffer.value = value 
                move_buffer.reward = reward 


                state = new_state
                with torch.no_grad():
                    policy_prob = torch.softmax(policy,dim=-1).squeeze(-1)
                    policy_prob = policy_prob[policy_prob != 0]
                    policy_entropy = -(policy_prob * torch.log2(policy_prob)).sum()
                    
                

                if step%LOG_EVERY==0:

                    wandb.log(
                        {   
                            'Score':conflicts,
                            'Relative policy entropy': policy_entropy/max_entropy,
                            'Value': value,
                            'Reward': reward,
                            'Episode':episode,
                            'Select min count.':ucb_count.min(),
                        }
                    )
                # -------------------- MODEL OPTIMIZATION --------------------

                
                if episode_end and episode % BATCH_SIZE:

                    agent.update(
                        ep_buf
                    )

                # torch.cuda.empty_cache() 

                step += 1
                ep_step += 1
        
                # checkpoint the policy net
                if step % CHECKPOINT_PERIOD == 0:
                    torch.save(agent.state_dict(), f'models/checkpoint/{ENCODING}/{step // CHECKPOINT_PERIOD}.pt')


            if episode_best_score > best_score:
                best_score = episode_best_score

    except KeyboardInterrupt:
        pass

    reporter = MemReporter(agent)
    reporter.report()
    print("STILL VALID :",pz.verify_solution(to_list(state,bsize)))
    print(best_score)
    return 




def get_conflicts(state:Tensor, bsize:int) -> int:
    """
    Evaluates the quality of a solution.
    /!\ This only is the true number of connections if the solution was created
    with side_importance = 1 /!\ 

    Input
    --------
    state: evaluated state of size [max_board_size + 2, max_board_size + 2, 4 * color_encoding_size]
    TODO: Shape the reward to help learning that only the sides are grey
    """

    offset = (PADDED_SIZE - bsize) // 2

    board = state[offset:offset+bsize,offset:offset+bsize]
    
    extended_board = state[offset-1:offset+bsize+1,offset-1:offset+bsize+1]

    n_offset = extended_board[2:,1:-1,2*COLOR_ENCODING_SIZE:3*COLOR_ENCODING_SIZE]
    s_offset = extended_board[:-2,1:-1,:COLOR_ENCODING_SIZE]
    w_offset = extended_board[1:-1,:-2,3*COLOR_ENCODING_SIZE:4*COLOR_ENCODING_SIZE]
    e_offset = extended_board[1:-1,2:,COLOR_ENCODING_SIZE:2*COLOR_ENCODING_SIZE]


    print(state.size())
    print(n_offset.size())
    n_connections = board[:,:,:COLOR_ENCODING_SIZE] == n_offset
    s_connections = board[:,:,2*COLOR_ENCODING_SIZE:3*COLOR_ENCODING_SIZE] == s_offset
    w_connections = board[:,:,COLOR_ENCODING_SIZE: 2*COLOR_ENCODING_SIZE] == w_offset
    e_connections = board[:,:,3*COLOR_ENCODING_SIZE: 4*COLOR_ENCODING_SIZE] == e_offset



    redundant_ns = torch.logical_and(n_connections[:-1,:],s_connections[1:,:])
    redundant_we = torch.logical_and(w_connections[:,1:],e_connections[:,:-1])

    redundant_connections = torch.all(redundant_we,dim=-1).sum() + torch.all(redundant_ns,dim=-1).sum()

    all = (torch.all(n_connections,dim=-1).sum() + torch.all(s_connections,dim=-1).sum() + torch.all(e_connections,dim=-1).sum() + torch.all(w_connections,dim=-1).sum())
    
    total_connections = all - redundant_connections

    max_connections = (bsize + 1) * bsize * 2

    if total_connections == max_connections:
        return 10

    return max_connections - total_connections




# ----------- MAIN CALL -----------

if __name__ == "__main__"  and '__file__' in globals():

    args = parse_arguments()
    CONFIG['Instance'] = args.instance.replace('instances/','')

    wandb.init(
        project='Eternity II',
        group='Distributional approach',
        config=CONFIG
    )

    train_model()

    wandb.finish()


