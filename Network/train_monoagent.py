import os
from einops import repeat, rearrange
import torch
from torch import Tensor
from Trajectories import *
from math import exp
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
    # torch.cuda.is_available = lambda : False
    

    # -------------------- NETWORK INIT -------------------- 

    if torch.cuda.is_available() and CUDA_ONLY: #FIXME:
        device = 'cuda'
    else:
        device = 'cpu'

    print()
    if n_tiles % HORIZON:
        raise UserWarning(f"Episode length ({n_tiles}) is not a multiple of horizon ({HORIZON})")


    cfg = {
    'n_tiles' : n_tiles ,
    'gamma' : GAMMA ,
    'clip_eps' : CLIP_EPS ,
    'lr' : LR ,
    'epochs' : EPOCHS ,
    'minibatch_size' : MINIBATCH_SIZE,
    'horizon' : HORIZON,
    'seq_len' : SEQ_LEN,
    'state_dim' : bsize+2,
    'n_encoder_layers':N_ENCODER_LAYERS,
    'n_decoder_layers':N_DECODER_LAYERS,
    'n_heads':N_HEADS,
    'dim_embed': DIM_EMBED,
    'hidden_size':HIDDEN_SIZE,
    'entropy_weight':ENTROPY_WEIGHT,
    'value_weight':VALUE_WEIGHT,
    'gae_lambda':GAE_LAMBDA,
    'device' : device 
    }


    agent = PPOAgent(cfg)
    agent.model.train()

    move_buffer = AdvantageBuffer()
    
    ep_buf = EpisodeBuffer(
        ep_len=n_tiles,
        n_tiles=n_tiles,
        bsize=bsize+2,
        horizon=HORIZON,
        seq_len=SEQ_LEN,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        device=device
    )

    max_entropy = torch.log2(torch.tensor(n_tiles))
    policy_entropy = max_entropy

    # -------------------- TRAINING LOOP --------------------

    torch.cuda.empty_cache()

    step = 0

    print("start")

    best_score = 0
    tile_importance = torch.zeros(n_tiles)
    episode = 0

    try:
        
        while 1:

            state, remaining_tiles, n_tiles = initialize_sol(args.instance,device)
            state = state.to(dtype=UNIT)
            mask = (torch.zeros_like(remaining_tiles[:,0]) == 0)
            episode_end = False
            prev_conflicts = get_conflicts(state,bsize)

            # print(f"NEW EPISODE : - {episode:>5}")
            ep_reward = 0
            connections = 0
            episode_best_score = 0
            ep_start = step
            ep_step = 0


            while not episode_end:
                
                with torch.no_grad():
                    policy, value = agent.model.get_action(
                        ep_buf.state_buf[ep_buf.ptr-1],
                        ep_buf.tile_seq[ep_buf.ptr-1],
                        torch.tensor(ep_step,device=device),
                        mask,
                        )
                selected_tile_idx = agent.get_action(policy)
                tile_importance[selected_tile_idx.cpu()] += (n_tiles-ep_step) / 1000
                selected_tile = remaining_tiles[selected_tile_idx]
                new_state, reward, connect = place_tile(state,selected_tile,ep_step)
                connections += connect


                ep_reward += reward


                ep_buf.push(
                    state=state,
                    action=selected_tile_idx,
                    tile=selected_tile,
                    tile_mask=mask,
                    policy=policy,
                    value=value,
                    reward=reward,
                    ep_step=ep_step,
                    final= (ep_step == (n_tiles-1))
                )
            

                if ep_step == n_tiles-1:
                    if step > 16*200:
                        list_sol = to_list(new_state,bsize)
                        pz.display_solution(list_sol,f"disp/{step}")

                    # print(pz.verify_solution(list_sol))
                    
                    curr_valid_state = new_state

                    if episode % 1 == 0:
                        print(f"END EPISODE {episode} - Connections {connections}/{bsize * 2 *(bsize+1)}")
                    episode_end = True
                    episode += 1

                    wandb.log({
                        "Mean episode reward":ep_reward/(step - ep_start + 1e-5),
                        'Final connections':connections,
                        'Tile rank':(tile_importance - tile_importance.mean() )/ (tile_importance.std() + 1e-5)
                        }
                        )
                    

                    
                if (step) % HORIZON == 0 and ep_step != 0 or episode_end:
                    agent.update(
                        mem=ep_buf
                    )
                
                if episode_end:
                    ep_buf.reset()


                
                mask[selected_tile_idx] = False
                state = new_state

                with torch.no_grad():
                    policy_prob = policy[policy != 0]
                    policy_entropy = -(policy_prob * torch.log2(policy_prob)).sum()

                if step%LOG_EVERY ==0:
                    wandb.log(
                        {   
                        'Reward': reward,
                        }
                    )
                if ep_step == 0:

                    wandb.log(
                        {   
                            'Relative policy entropy': policy_entropy/max_entropy,
                            'Value': value,
                        }
                    )

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
    print("STILL VALID :",pz.verify_solution(to_list(curr_valid_state,bsize)))
    print(best_score)
    return 


def place_tile(state:Tensor,tile:Tensor,step:int):

    state = state.clone()
    bsize = state.size()[0] - 2
    best_rew = -1
    best_connect = 0
    for _ in range(4):
        tile = tile.roll(COLOR_ENCODING_SIZE,-1)
        state[step // bsize + 1, step % bsize + 1,:] = tile
        connect,reward = filling_connections(state,bsize,step)
        if reward > best_rew:
            best_state=state.clone()
            best_rew=reward
            best_connect = connect

    return best_state, best_rew, best_connect

def streak(streak_length:int, n_tiles):
    return (2 - exp(-streak_length * 3/(0.8 * n_tiles)))



def filling_connections(state:Tensor, bsize:int, step):
    i = step // bsize + 1
    j = step % bsize + 1
    west_tile_color = state[i,j-1,3*COLOR_ENCODING_SIZE:4*COLOR_ENCODING_SIZE]
    south_tile_color = state[i-1,j,:COLOR_ENCODING_SIZE]

    west_border_color = state[i,j,1*COLOR_ENCODING_SIZE:2*COLOR_ENCODING_SIZE]
    south_border_color = state[i,j,2*COLOR_ENCODING_SIZE:3*COLOR_ENCODING_SIZE]

    connections = 0
    reward = 0

    if j == 1:
        if torch.all(west_border_color == 0):
            reward += 2
            connections += 1
    
    elif torch.all(west_border_color == west_tile_color):
        connections += 1
        reward += 1

    if i == 1:
        if torch.all(south_border_color == 0):
            connections += 1
            reward += 2
    
    elif torch.all(south_border_color == south_tile_color):
        connections += 1
        reward += 1
   
   
    if j == bsize:

        east_border_color = state[i,j,3*COLOR_ENCODING_SIZE:4*COLOR_ENCODING_SIZE]

        if torch.all(east_border_color == 0):
            connections += 1
            reward += 2
    

    if i == bsize:

        north_border_color = state[i,j,:COLOR_ENCODING_SIZE]
        if torch.all(north_border_color == 0):
            reward += 2
            connections += 1
    

    return connections, reward
        
        



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
    torch.autograd.set_detect_anomaly(True)
    wandb.init(
        project='EteRLnity',
        entity='mateo-clemente',
        group='Tests',
        config=CONFIG
    )

    train_model()

    wandb.finish()


