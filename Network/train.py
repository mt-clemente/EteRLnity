from collections import namedtuple
import gc
import sys

from einops import repeat, rearrange
import torch
from torch import Tensor
from DRQN import *
from torch.profiler import profile, record_function, ProfilerActivity
from math import comb
from pytorch_memlab import MemReporter
from utils import *
from param import *



def train_model(hotstart:str = None):

    """
    Your solver for the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """




    # -------------------- NETWORK INIT -------------------- 

    # torch.cuda.is_available = lambda : False
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    Transition = namedtuple('Transition',
                                     ('state', 'next_state', 'reward','mask','weights','indexes'))



    if hotstart:
        policy_net = DQN(MAX_BSIZE+2, MAX_BSIZE+2, 1, device,COLOR_ENCODING_SIZE)
        policy_net.load_state_dict(torch.load(hotstart))
    else:
        policy_net = DQN(MAX_BSIZE + 2, MAX_BSIZE + 2, 1, device, COLOR_ENCODING_SIZE)
    
    target_net = DQN(MAX_BSIZE+2, MAX_BSIZE+2, 1, device, COLOR_ENCODING_SIZE).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.RMSprop(policy_net.parameters(),lr = LR,eps=1e-5)


    
    # -------------------- GAME INIT --------------------

    state, bsize = initialize_sol(sys.argv[-1])
    
    state = state.to(dtype=UNIT)

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


    obs_neighborhood_size = int(PARTIAL_OBSERVABILITY * neighborhood_size)
    observation_mask = (torch.cat((torch.ones(obs_neighborhood_size),torch.zeros(neighborhood_size - obs_neighborhood_size))) == 1)



    # -------------------- TRAINING LOOP --------------------

    step = 0
    pz = EternityPuzzle(sys.argv[-1])
    observation_mask = observation_mask[torch.randperm(neighborhood_size)]
    neighborhood = gen_neighborhood(state, observation_mask, swp_idx, swp_mask, rot_mask)
    offset = (MAX_BSIZE + 2 - bsize) // 2
    display_solution(0,to_list(state,bsize),"start.png")
    print(state[offset:offset+bsize,offset:offset+bsize])
    print(neighborhood[0,offset:offset+bsize,offset:offset+bsize])
    print(to_list(state,bsize))
    print(pz.verify_solution(to_list(state,bsize)))
    print(pz.verify_solution(to_list(neighborhood[0],bsize)))
    try:
        while 1:

            print(step)

            # TODO: Test the efficacy of either modifying the current neighbor hood vs repeat state -> swap

            #best move
            heur_val = policy_net(neighborhood.unsqueeze(1))
            best_move_idx = torch.argmax(heur_val)
            new_state = neighborhood[best_move_idx]


            #reward
            state_val = eval_sol(state,bsize)
            new_state_val = eval_sol(new_state,bsize)
            reward = state_val - new_state_val

            memory.push(state,new_state,reward,observation_mask)

            state = new_state

            observation_mask = observation_mask[torch.randperm(neighborhood_size)]
            neighborhood = gen_neighborhood(state,observation_mask,swp_idx,swp_mask,rot_mask)

            # -------------------- MODEL OPTIMIZATION --------------------

            
            if len(memory) >= BATCH_SIZE and step % TRAIN_FREQ == 0:
            
                print("aaaa",pz.verify_solution(to_list(state,bsize)))
                for _ in range(BATCH_NB):

                    batch = memory.sample()

                    # TODO: No numpy

                    state_batch =  batch.state.unsqueeze(1).to(device)
                    next_state_batch = batch.next_state.to(device)
                    reward_batch =  batch.reward.to(device)
                    mask_batch = batch.mask.to(device)
                    weights = batch.weights.to(device)

                    state_values = policy_net(state_batch)

                    next_state_values = torch.zeros(BATCH_SIZE, device=device)

                    with torch.no_grad():

                        next_states = torch.cat([gen_neighborhood(next_state_batch[i],mask_batch[i],swp_idx,swp_mask,rot_mask) for i in range(BATCH_SIZE)])

                        target_vals = target_net(next_states.unsqueeze(1))
                        target_vals = rearrange(target_vals.squeeze(-1), '(b n) -> b n ', b=BATCH_SIZE)
                        next_state_values = torch.max(target_vals, dim=1)[0]

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
                    memory.update_priorities(batch.indexes, new_prio)

                    # torch.cuda.empty_cache() 




            step += 1
            #target update
            if step % TARGET_UPDATE == 0:
                
                target_net.load_state_dict(policy_net.state_dict())
    
            # checkpoint the policy net
            # if self.num_episode % (self.TARGET_UPDATE * self.TRAIN_FREQ) == 0:
            if step % CHECKPOINT_PERIOD == 0:
                torch.save(policy_net.state_dict(), f'models/checkpoint/model_{step // CHECKPOINT_PERIOD}.pt')


    except KeyboardInterrupt:
        pass

    reporter = MemReporter()
    # reporter.report()
    l = to_list(state,bsize)
    print("aaaa",pz.verify_solution(to_list(state,bsize)))
    
    display_solution(0,l,"yat.png")

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


    return torch.vstack((swap_elements(state,obs_swp_idx,obs_swp_mask),rotate_elements(state,obs_rot_mask)))


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
    rolled = state_batch.clone()
    
    for dir in range(4):
        rolled[dir] = state_batch[0].roll(COLOR_ENCODING_SIZE * dir,dims=-1)
    
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



def eval_sol(state:Tensor, bsize:int, encoding = 'binary') -> int:
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

    n_offset = extended_board[2:,1:-1,:COLOR_ENCODING_SIZE]
    s_offset = extended_board[:-2,1:-1,COLOR_ENCODING_SIZE:2*COLOR_ENCODING_SIZE]
    w_offset = extended_board[1:-1,:-2,2*COLOR_ENCODING_SIZE:3*COLOR_ENCODING_SIZE]
    e_offset = extended_board[1:-1,2:,3*COLOR_ENCODING_SIZE:4*COLOR_ENCODING_SIZE]

    n_connections = board[:,:,:COLOR_ENCODING_SIZE] - n_offset
    s_connections = board[:,:,COLOR_ENCODING_SIZE: 2*COLOR_ENCODING_SIZE] - s_offset
    w_connections = board[:,:,2*COLOR_ENCODING_SIZE: 3*COLOR_ENCODING_SIZE] - w_offset
    e_connections = board[:,:,3*COLOR_ENCODING_SIZE: 4*COLOR_ENCODING_SIZE] - e_offset

    total_connections = n_connections.count_nonzero() + s_connections.count_nonzero() + e_connections.count_nonzero() + w_connections.count_nonzero()

    
    return total_connections




# ----------- MAIN CALL -----------

if __name__ == "__main__":
    train_model()

