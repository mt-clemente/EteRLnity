from collections import namedtuple
import sys

from einops import repeat, rearrange
import torch
from torch import Tensor
from DRQN import *
from math import comb
from pytorch_memlab import MemReporter
from utils import *
from param import *
import wandb
import torchrl


def train_model(hotstart:str = None):

    """
    Your solver for the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """




    # -------------------- NETWORK INIT -------------------- 

    args = parse_arguments()
    hotstart = args.hotstart
    # torch.cuda.is_available = lambda : False
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    Transition = namedtuple('Transition',
                                     ('state',
                                      'next_state',
                                      'reward',
                                      'target_val',
                                      'state_val',
                                      'mask',
                                      'weights',
                                      'indexes'
                                      )
                                      )



    if hotstart:
        policy_net = DQN(MAX_BSIZE+2, MAX_BSIZE+2, 1, device,COLOR_ENCODING_SIZE)
        policy_net.load_state_dict(torch.load(hotstart))
    else:
        policy_net = DQN(MAX_BSIZE + 2, MAX_BSIZE + 2, 1, device, COLOR_ENCODING_SIZE)
    
    target_net = DQN(MAX_BSIZE+2, MAX_BSIZE+2, 1, device, COLOR_ENCODING_SIZE).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(),lr = LR,eps=1e-6)

    move_buffer = MoveBuffer()
    
    # -------------------- GAME INIT --------------------

    state, bsize = initialize_sol(args.instance)
    
    state = state.to(dtype=UNIT)

    INIT_SOL = to_list(state,bsize)

    swp_idx = gen_swp_idx(bsize)
    swp_mask, rot_mask = gen_masks(bsize,swp_idx)
    neighborhood_size = swp_idx.size()[0] + rot_mask.size()[0]
    
    
    memory = PrioritizedReplayMemory(
        size=MEM_SIZE,
        Transition=Transition,
        alpha=ALPHA,
        batch_size=BATCH_SIZE,
        max_bsize=MAX_BSIZE,
        encoding_size=COLOR_ENCODING_SIZE,
        neighborhood_size=neighborhood_size
    )

    cpu_buf = CPUBuffer(
        capacity=TRAIN_FREQ,
        neighborhood_size=neighborhood_size,
        linked_mem=memory
    )

    tabu = TabuList(TABU_LENGTH)

    obs_neighborhood_size = int(PARTIAL_OBSERVABILITY * neighborhood_size)
    observation_mask = (torch.cat((torch.ones(obs_neighborhood_size),torch.zeros(neighborhood_size - obs_neighborhood_size))) == 1)

    max_entropy = torch.log2(torch.tensor(obs_neighborhood_size))

    # -------------------- TRAINING LOOP --------------------

    torch.cuda.empty_cache()

    step = 0
    pz = EternityPuzzle(args.instance)
    observation_mask = observation_mask[torch.randperm(neighborhood_size)]
    neighborhood = gen_neighborhood(state, observation_mask, swp_idx, swp_mask, rot_mask)
    pz.display_solution(to_list(state,bsize),"start.png")


    move_buffer.state = state
    move_buffer.reward = 0
    move_buffer.state_val = 0
    best_score = 0
    _, prev_state_score = eval_sol(state,bsize,best_score)


    try:

        while 1:

            if step % 10 == 0:
                print(step)

            tabu.update(step)
            heur_val = policy_net(neighborhood.unsqueeze(1))

            with torch.no_grad():
                target_heur_val = target_net(neighborhood.unsqueeze(1)).max()

            best_move_idx = torch.argsort(heur_val.squeeze(-1),descending=True)

            new_state = None
            for idx in best_move_idx:
                if not tabu.in_tabu(neighborhood[idx]):
                    new_state = neighborhood[idx].squeeze(0)
                    best_idx = idx
                    break
            
            if new_state is None:
                tabu.fast_foward()
                continue

            tabu.push(new_state,step)
            #reward
            new_state_val, score = eval_sol(new_state,bsize,best_score)

            # if torch.all(new_state == state):
                # raise OSError
            # if ((bsize + 1) * bsize * 2 - score) != pz.get_total_n_conflict(to_list(state,bsize)):
                # raise OSError

            if score > best_score:
                pz.display_solution(to_list(new_state,bsize),f'best_sol_{score}')
                best_score = score
            reward =  new_state_val + max(score - prev_state_score,0)

            cpu_buf.push(
                move_buffer.state,
                state,
                move_buffer.reward,
                target_heur_val,
                move_buffer.state_val,
                observation_mask
            )
            
            prev_state_score = score
            move_buffer.state = state
            move_buffer.reward = reward
            move_buffer.state_val = heur_val[best_idx].detach()


            wandb.log({'Score':score})
            state = new_state


            observation_mask = observation_mask[torch.randperm(neighborhood_size)]
            neighborhood = gen_neighborhood(new_state,observation_mask,swp_idx,swp_mask,rot_mask)

            # wandb.log({'reward':reward,'State Q value':heur_val[best_move_idx]})
            policy_prob = heur_val / heur_val.sum()
            policy_entropy = -(policy_prob * torch.log2(policy_prob)).sum()
            print(heur_val.max(),heur_val.min(),heur_val.sum(),reward)
            print((policy_entropy/max_entropy).squeeze(-1))
            if heur_val.max() < 0:
                print(heur_val.mean())
                raise OSError
            wandb.log(
                {   
                    'Relative policy entropy': policy_entropy/max_entropy,
                    'Q values':heur_val[best_idx],
                    'Max next Q vlaues':target_heur_val,
                    'reward':reward,
                }
            )
            # -------------------- MODEL OPTIMIZATION --------------------

            
            if len(memory) >= BATCH_SIZE and step % TRAIN_FREQ == 0:
            
                if random.random() > 1 - 10e-3:
                    ok = pz.verify_solution(to_list(state,bsize))

                    if ok == False:
                        
                        print(state)
                        print(to_list(state,bsize))
                        print(expected_state_values)
                        print(eltwise_loss)
                        print(reward_batch)
                        raise ValueError

                for _ in range(BATCH_NB):

                    torch.cuda.empty_cache()

                    batch = memory.sample()

                    state_batch = torch.from_numpy(batch.state).unsqueeze(1).to(UNIT).to(device)
                    # next_state_batch = torch.from_numpy(batch.next_state).to(UNIT).to(device)
                    reward_batch = torch.from_numpy(batch.reward).to(UNIT).to(device)
                    target_val = torch.from_numpy(batch.target_val).to(UNIT).to(device)
                    old_state_vals = torch.from_numpy(batch.state_val).to(UNIT).to(device)
                    # mask_batch = torch.from_numpy(batch.mask).to(UNIT).to(device)
                    weights = torch.from_numpy(batch.weights).to(UNIT).to(device)
                    state_values = policy_net(state_batch)

                    next_state_values = target_val

                    """ with torch.no_grad():

                        a = [gen_neighborhood(next_state_batch[i],mask_batch[i],swp_idx,swp_mask,rot_mask) for i in range(BATCH_SIZE)]
                        next_states = torch.cat(a)

                        target_val = target_net(next_states.unsqueeze(1))
                        target_val = rearrange(target_val.squeeze(-1), '(b n) -> b n ', b=BATCH_SIZE)
                        next_state_values = torch.max(target_val, dim=1)[0] """

                    expected_state_values = (next_state_values * GAMMA) + reward_batch

                    criterion = nn.HuberLoss(reduction='none')
                    eltwise_loss = criterion(state_values, expected_state_values.unsqueeze(1))

                    loss = torch.mean(eltwise_loss * weights)

                    optimizer.zero_grad()
                    loss.backward()

                    # gradient clipping
                    for param in policy_net.parameters():
                        param.grad.clamp_(-1,1)

                    optimizer.step()

                    # PER prio update
                    prio_loss = eltwise_loss
                    new_prio = prio_loss + PRIO_EPSILON
                    memory.update_priorities(batch.indexes, new_prio.cpu().detach().numpy())

                    pst_val = (state_values - state_values.min())/(state_values - state_values.min()).sum()
                    old_pst_val = (old_state_vals - old_state_vals.min())/(old_state_vals - old_state_vals.min()).sum()
                    print(pst_val.sum(),pst_val.min(),pst_val.min())
                    print(old_pst_val.sum(),old_pst_val.min())
                    indx = torch.log
                    print("kl",torch.nn.functional.kl_div(pst_val[pst_val != 0 ],old_pst_val,reduction='mean').squeeze(-1))
                    raise OSError
                    if step > 500:
                        wandb.log(
                            {
                                'Train mean Q values':state_values.mean(),
                                'Train mean next Q values':next_state_values.mean(),
                                'Train mean expected Q vlaues':expected_state_values.mean(),
                                'Train mean reward':reward_batch.mean(),
                                'Train Loss':loss,
                                'KL Divergence':torch.nn.functional.kl_div(state_values,old_state_vals,reduction='batchmean')
                            
                            }
                        )

                    # torch.cuda.empty_cache() 




            step += 1
            #target update
            if step % TARGET_UPDATE == 0:
                
                target_net.load_state_dict(policy_net.state_dict())
    
            # checkpoint the policy net
            # if self.num_episode % (self.TARGET_UPDATE * self.TRAIN_FREQ) == 0:
            if step % CHECKPOINT_PERIOD == 0:
                torch.save(policy_net.state_dict(), f'models/checkpoint/{ENCODING}/{step // CHECKPOINT_PERIOD}.pt')


    except KeyboardInterrupt:
        pass

    # reporter = MemReporter()
    # reporter.report()
    l = to_list(state,bsize)
    print("STILL VALID :",pz.verify_solution(to_list(state,bsize)))
    print(best_score)
    return 


def gen_neighborhood(state:Tensor, observation_mask:Tensor, swp_idx:Tensor, swp_mask:Tensor, rot_mask:Tensor):
    """
    Generate state's an observed part of the full neighborhood including ranged 2-swaps depending
    on SWAP_RANGE specified in param.py, plus the rotations of each tile.

    Input
    ---------
    state: the current state
    observation_mask: a mask describing the part of the neihborhood that is observed
    swp_idx, swp_mask, rot_mask are fixed masks that represent swapped and rotated elements.

    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    obs_swp_idx = swp_idx[observation_mask[:swp_idx.size()[0]]].to(device=device)
    obs_swp_mask = swp_mask[:,observation_mask[:swp_idx.size()[0]]].to(device=device)
    obs_rot_mask = rot_mask[observation_mask[swp_idx.size()[0]:]].to(device=device)


    if obs_rot_mask.size()[0] == 0:
        return swap_elements(state,obs_swp_idx,obs_swp_mask)

    if obs_swp_idx.size()[0] == 0:
        return rotate_elements(state,obs_rot_mask)
    
    r = rotate_elements(state,obs_rot_mask)
    s = swap_elements(state,obs_swp_idx,obs_swp_mask)
    return torch.vstack((s,r))


def rotate_elements(state:Tensor, rot_mask:Tensor):
    """
    Generates the states that are one rotation away

    Input
    ----------
    state: the current state
    rot_mask: mask containing ones at every tile one after the other, repeated
    4 times (one for each rotation angle) of shape [board_size**2*4, board_size, board_size, 4*color_encoding_size ]
    """

    state_batch = repeat(state, 'i j c -> k i j c',k=rot_mask.size()[0])
    rolled = torch.zeros((3,rot_mask.size()[0]//3,*state.size()),device=state.device,dtype=state.dtype)

    for dir in range(3):
        rolled[dir,:] = state.roll(COLOR_ENCODING_SIZE * (dir+1),dims=-1)

    rolled = rearrange(rolled,'dir k i j c -> (dir k) i j c')

    state_batch = torch.where(rot_mask,rolled,state_batch)

    return state_batch
    



def swap_elements(state:Tensor, swp_idx:Tensor, swp_mask:Tensor) -> Tensor:
    """
    Generate all given 2-swaps. The swaps are done out of place.

    Input
    ----------
    state_batch : a batch of states of dimension [batch_size, board_size, board_size, 4 * COLOR_ENCODING_SIZE]
    swp_idx : a batch of coordinates of elements to be swapped of dimension [batch_size, 2, 2]

    TODO: Make inplace
    """

    expanded_state = repeat(state, 'i j c -> b i j c', b=swp_idx.size()[0])

    i = swp_idx[:,0]
    j = swp_idx[:,1]

    t1 = expanded_state[torch.arange(expanded_state.size()[0]), i[:,0], i[:,1]]
    t2 = expanded_state[torch.arange(expanded_state.size()[0]), j[:,0], j[:,1]]

    swapped_batch = expanded_state.masked_scatter(swp_mask[0], t2) # FIXME: inplace error
    swapped_batch.masked_scatter_(swp_mask[1], t1)
    
    # Return the modified input expanded_state of tensors
    return swapped_batch



def eval_sol(state:Tensor, bsize:int,best_score:int) -> int:
    """
    Evaluates the quality of a solution.
    /!\ This only is the true number of connections if the solution was created
    with side_importance = 1 /!\ 

    Input
    --------
    state: evaluated state of size [max_board_size + 2, max_board_size + 2, 4 * color_encoding_size]
    TODO: Shape the reward to help learning that only the sides are grey
    """

    offset = (MAX_BSIZE + 2 - bsize) // 2
    
    board = state[offset:offset+bsize,offset:offset+bsize]
    
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

    if total_connections == max_connections:
        return 100, total_connections

    return (total_connections - best_score) / (max_connections - total_connections + 1), total_connections




# ----------- MAIN CALL -----------

if __name__ == "__main__"  and '__file__' in globals():


    if '-nw' in sys.argv:
        train_model()

    else:
        wandb.init(
            project='Eternity II',
            config=CONFIG
        )

        train_model()

        wandb.finish()


