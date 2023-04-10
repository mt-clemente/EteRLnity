from collections import namedtuple
import os

from einops import repeat, rearrange
import torch
from torch import Tensor
from Trajectories import *
from math import comb, exp
from pytorch_memlab import MemReporter
from Transformer import PPOAgent
from utils import *
from param import *
import wandb

LOG_EVERY = 1

def train_model(hotstart:str = None):

    """
    Your solver for the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    args = parse_arguments()
    pz = EternityPuzzle(args.instance)
    n_tiles = len(pz.piece_list)
    bsize = pz.board_size
    hotstart = args.hotstart
    torch.cuda.is_available = lambda : False
    
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
    'clip_eps' : CLIP_EPS ,
    'lr' : LR ,
    'epochs' : EPOCHS ,
    'minibatch_size' : MINIBATCH_SIZE ,
    'horizon' : HORIZON * n_tiles,
    'state_dim' : bsize+2,
    'conv_sizes': CONV_SIZES,
    'hidden_size':HIDDEN_SIZE,
    'kernel_sizes': KERNEL_SIZES,
    'entropy_weight':ENTROPY_WEIGHT,
    'value_weight':VALUE_WEIGHT,
    'gae_lambda':GAE_LAMBDA,
    'device' : device 
    }


    agent = PPOAgent(cfg)

    move_buffer = AdvantageBuffer()
    
    memory = BatchMemory(
        n_tiles=n_tiles,
        bsize=bsize+2,
        ep_length=bsize**2,
        capacity=HORIZON*n_tiles,
        device=device
    )

    ep_buf = EpisodeBuffer(
        capacity=n_tiles,
        n_tiles=n_tiles,
        bsize=bsize+2,
        device=device
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

            # print(f"NEW EPISODE : - {episode:>5}")
            conflicts = 0
            ep_reward = 0
            consec_good_moves = 0
            episode_best_score = 0
            ep_start = step
            ep_step = 0
            reward_to_go = n_tiles * 4 # approx 2*max_reward


            while not episode_end:

                print(step)

                with torch.no_grad():
                    policy, value = agent.model.get_action(
                        state,
                        ep_buf.act_buf,
                        ep_buf.rew_buf,
                        torch.tensor(ep_step,device=device),
                        )

                selected_tile_idx = agent.get_action(policy,mask)
                selected_tile = remaining_tiles[selected_tile_idx]
                mask[selected_tile_idx] = False
                new_state, new_conf = place_tile(state,selected_tile,ep_step)

                conflicts += new_conf

                if new_conf == 0:
                    consec_good_moves += 1
                    reward = streak(consec_good_moves,n_tiles)
                else:
                    consec_good_moves = 0
                    reward = - 1
                
                reward_to_go -= new_conf

                ep_reward += reward

                if ep_step != 0:

                    ep_buf.push(
                        state=move_buffer.state,
                        action=move_buffer.action,
                        policy=move_buffer.policy,
                        value=move_buffer.value,
                        next_value=value,
                        reward_to_go=move_buffer.reward_to_go,
                        final=0
                    )

                    memory.load(
                        ep_buf,
                        ep_step
                    )


                if ep_step == n_tiles-1:

                    # pz.display_solution(to_list(new_state,bsize),f"{step}")
                    ep_buf.push(
                        state=state,
                        action=selected_tile_idx,
                        policy=policy,
                        value=value,
                        next_value=0,
                        reward_to_go=reward_to_go,
                        final=1
                    )
                    memory.load(
                        ep_buf,
                        ep_step+1
                    )
                    if episode % 10 == 0:
                        print(f"END EPISODE {episode} - Conflicts {conflicts}/{bsize * 2 *(bsize+1)}",end='\r')
                    episode_end = True
                    episode += 1

                    if episode % LOG_EVERY == 0:
                        wandb.log({"Mean episode reward":ep_reward/(step - ep_start + 1e-5),'Final conflicts':conflicts})
                    

                if (step % (HORIZON * n_tiles)) == 0 and step != 0:

                    agent.update(
                        mem=memory
                    )


                
                prev_conflicts = conflicts
                move_buffer.state = state 
                move_buffer.action = selected_tile_idx 
                move_buffer.policy = policy 
                move_buffer.value = value 
                move_buffer.reward_to_go = reward_to_go 


                state = new_state

                with torch.no_grad():
                    policy_prob = torch.softmax(policy,dim=-1).squeeze(-1)
                    policy_prob = policy_prob[policy_prob != 0]
                    policy_entropy = -(policy_prob * torch.log2(policy_prob)).sum()
                    
                

                if step % LOG_EVERY==0:

                    wandb.log(
                        {   
                            'Score':conflicts,
                            'Relative policy entropy': policy_entropy/max_entropy,
                            'Value': value,
                            'Reward': reward,
                        }
                    )
                # -------------------- MODEL OPTIMIZATION --------------------


                # torch.cuda.empty_cache() 

                step += 1
                ep_step += 1
        
                # checkpoint the policy net
                if step % CHECKPOINT_PERIOD == 0:
                    inst = args.instance.replace("instances/eternity_","")
                    inst = inst.replace(".txt","")
                    try:
                        torch.save(agent.model.state_dict(), f'models/checkpoint/{inst}/{step // CHECKPOINT_PERIOD}.pt')
                    except:
                        os.mkdir(f"models/checkpoint/{inst}/")
                        torch.save(agent.model.state_dict(), f'models/checkpoint/{inst}/{step // CHECKPOINT_PERIOD}.pt')


            if episode_best_score > best_score:
                best_score = episode_best_score

    except KeyboardInterrupt:
        pass

    # reporter = MemReporter(agent)
    # reporter.report()
    print("STILL VALID :",pz.verify_solution(to_list(state,bsize)))
    print(best_score)
    return 


def place_tile(state:Tensor,tile:Tensor,step:int):

    state = state.clone()
    bsize = state.size()[0] - 2
    best_conf = 540
    for dir in range(4):
        tile = tile.roll(COLOR_ENCODING_SIZE,-1)
        state[step // bsize + 1, step % bsize + 1,:] = tile
        conf = filling_conflicts(state,bsize,step)
        if conf < best_conf:
            best_state=state.clone()
            best_conf=conf

    return best_state, best_conf

def streak(streak_length:int, n_tiles):
    return (2 - exp(-streak_length * 3/(0.8 * n_tiles)))



def filling_conflicts(state:Tensor, bsize:int, step):
    i = step // bsize + 1
    j = step % bsize + 1
    west_tile_color = state[i,j-1,3*COLOR_ENCODING_SIZE:4*COLOR_ENCODING_SIZE]
    south_tile_color = state[i-1,j,:COLOR_ENCODING_SIZE]

    west_border_color = state[i,j,1*COLOR_ENCODING_SIZE:2*COLOR_ENCODING_SIZE]
    south_border_color = state[i,j,2*COLOR_ENCODING_SIZE:3*COLOR_ENCODING_SIZE]

    conf = 0

    if j == 1:
        if not torch.all(west_border_color == 0):
            conf += 1
    
    elif not torch.all(west_border_color == west_tile_color):
        conf += 1

    if i == 1:
        if not torch.all(south_border_color == 0):
            conf += 1
    
    elif not torch.all(south_border_color == south_tile_color):
        conf += 1
   
   
    if j == bsize:

        east_border_color = state[i,j,3*COLOR_ENCODING_SIZE:4*COLOR_ENCODING_SIZE]

        if not torch.all(east_border_color == 0):
            conf += 1
    

    if i == bsize:

        north_border_color = state[i,j,:COLOR_ENCODING_SIZE]
        if not torch.all(north_border_color == 0):
            conf += 1
    

    return conf
        
        



def get_conflicts(state:Tensor, bsize:int, step:int = 0) -> int:

    offset = 1
    mask = torch.ones(bsize**2)
    board = state[offset:offset+bsize,offset:offset+bsize].clone()
    
    extended_board = state[offset-1:offset+bsize+1,offset-1:offset+bsize+1]

    n_offset = extended_board[2:,1:-1,2*COLOR_ENCODING_SIZE:3*COLOR_ENCODING_SIZE]
    s_offset = extended_board[:-2,1:-1,:COLOR_ENCODING_SIZE]
    w_offset = extended_board[1:-1,:-2,3*COLOR_ENCODING_SIZE:4*COLOR_ENCODING_SIZE]
    e_offset = extended_board[1:-1,2:,COLOR_ENCODING_SIZE:2*COLOR_ENCODING_SIZE]

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

    return max_connections - total_connections




# ----------- MAIN CALL -----------

if __name__ == "__main__"  and '__file__' in globals():

    args = parse_arguments()
    CONFIG['Instance'] = args.instance.replace('instances/','')

    wandb.init(
        project='Eternity II',
        group='Decision Transformer',
        config=CONFIG
    )

    train_model()

    wandb.finish()


