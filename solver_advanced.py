from collections import namedtuple
import gc

from einops import repeat, rearrange
from eternity_puzzle import EternityPuzzle
import torch
from torch import Tensor
from DRQN import *
from solver_random import solve_random
from math import comb
from pytorch_memlab import MemReporter



MAX_BSIZE = 16
NORTH = 0
SOUTH = 1
WEST = 2
EAST = 3

GRAY = 0
BLACK = 23
RED = 24
WHITE = 25
N_COLORS = 23

# -------------------- TRAINING SETTINGS -------------------- 

# TILE_ENCODING_SIZE = 4 * 5
PARTIAL_OBSERVABILITY = 0.5 
BATCH_NB = 50
CHECKPOINT_PERIOD = 10000
BATCH_SIZE = 32
LR = 10**-8
TARGET_UPDATE = 50
MEM_SIZE = 20000
PRIO_EPSILON = 1e-4
ALPHA = 0.4
BETA = 0.9
GAMMA = 0.95
TRAIN_FREQ = 100
# THREADS = 50


def solve_advanced(eternity_puzzle:EternityPuzzle, hotstart:str = None):

    """
    Your solver for the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    TODO: Only 2-swap the inner tiles (not the borders), reduces the neighborhood size from 
          nCr(16*16,2) to nCr(14*14,2) approx.
    """




    # -------------------- NETWORK INIT -------------------- 

    # torch.cuda.is_available = lambda : False
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    Transition = namedtuple('Transition',
                                     ('state', 'next_state', 'reward','weights','indexes'))



    if hotstart:
        policy_net = DQN(MAX_BSIZE+2, MAX_BSIZE+2, 1, device,N_COLORS)
        policy_net.load_state_dict(torch.load(hotstart))
    else:
        policy_net = DQN(MAX_BSIZE + 2, MAX_BSIZE + 2, 1, device, N_COLORS)
    
    target_net = DQN(MAX_BSIZE+2, MAX_BSIZE+2, 1, device, N_COLORS).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(),amsgrad=True,lr = LR,eps=1e-6)

    memory = PrioritizedReplayMemory(
        size=MEM_SIZE,
        Transition=Transition,
        alpha=ALPHA,
        batch_size=BATCH_SIZE,
        max_bsize=MAX_BSIZE,
        n_colors=N_COLORS

    )

    
    # -------------------- GAME INIT --------------------

    step = 0

    bsize = eternity_puzzle.board_size
    neighborhood_size =  comb((bsize-2)**2,2) + comb((bsize - 1) * 4, 2) # + bsize ** 2 * 4 
    obs_neighborhood_size = int(PARTIAL_OBSERVABILITY * neighborhood_size)
    observation_mask = (torch.cat(
        (
        torch.ones(obs_neighborhood_size),
        torch.zeros(neighborhood_size - obs_neighborhood_size)
        )) == 1
        )



    init_sol, _ = solve_random(eternity_puzzle)

    state = to_tensor(init_sol)

    neighborhood = repeat(state, 'i j c -> b i j c', b=obs_neighborhood_size)

    # These help generate the neigborhood, they do not change during training
    swp_idx = _gen_swp_idx(bsize, neighborhood_size)
    swp_mask = _gen_masks(bsize,swp_idx)

    observation_mask = observation_mask[[torch.randperm(neighborhood_size)]]
    obs_swp_idx = swp_idx[observation_mask].to(device=device)
    obs_swp_mask = swp_mask[:,observation_mask].to(device=device)
    neighborhood = swap_elements(neighborhood, obs_swp_idx, obs_swp_mask)



    # -------------------- TRAINING LOOP --------------------
    try:
        while 1:

            print(step)

            # TODO: Test the efficacy of either modifying the current neighbor hood vs repeat state -> swap

            #best move
            heur_val = policy_net(neighborhood.unsqueeze(1))
            best_move_idx = torch.argmax(heur_val)
            new_state = neighborhood[best_move_idx]


            #reward
            state_val = eval_sol(state)
            new_state_val = eval_sol(new_state)
            reward = state_val - new_state_val

            memory.push(state,new_state,reward)

            observation_mask = observation_mask[[torch.randperm(neighborhood_size)]]
            obs_swp_idx = swp_idx[observation_mask].to(device=device)
            obs_swp_mask = swp_mask[:,observation_mask].to(device=device)
            state = new_state
            neighborhood = swap_elements(repeat(state,'i j c -> n i j c', n=obs_neighborhood_size),obs_swp_idx,obs_swp_mask)

            # -------------------- MODEL OPTIMIZATION --------------------

            
            if len(memory) >= BATCH_SIZE and step % TRAIN_FREQ == 0:
                
                s = 0
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                            if obj.device == torch.device('cuda:0'):
                                s += obj.nelement() * obj.element_size()
                    except:
                        pass
                
                print(s / (1024*1024*1024))

                for _ in range(BATCH_NB):

                    batch = memory.sample()

                    # TODO: No numpy

                    state_batch =  batch.state.unsqueeze(1).to(device)
                    reward_batch =  batch.reward.to(device)
                    weights = batch.weights.to(device)

                    state_values = policy_net(state_batch)

                    next_state_values = torch.zeros(BATCH_SIZE, device=device)

                    with torch.no_grad():

                        next_states = torch.cat([swap_elements(repeat(batch.next_state[i],'i j c -> b i j c',b = obs_neighborhood_size).to(device),obs_swp_idx,obs_swp_mask) for i in range(BATCH_SIZE)])

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

                    torch.cuda.empty_cache() 




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
    reporter.report()

    return to_list(state)




def swap_elements(state_batch:Tensor, swp_idx:Tensor, masks:Tensor):
    """
    Generate all given 2-swaps. The swaps are done out of place.

    Input
    ------------
    state_batch : a batch of states of dimension [batch_size, board_size, board_size, n_colors]
    swp_idx : a batch of coordinates of elements to be swapped of dimension [batch_size, 2, 2]
    """
    i = swp_idx[:,0]
    j = swp_idx[:,1]

    t1 = state_batch[torch.arange(state_batch.size()[0]), i[:,0], i[:,1],:]
    t2 = state_batch[torch.arange(state_batch.size()[0]), j[:,0], j[:,1],:]

    swapped_batch = state_batch.masked_scatter(masks[0], t2)
    swapped_batch = state_batch.masked_scatter(masks[1], t1)
    
    # Return the modified input state_batch of tensors
    return swapped_batch



# -------------------- LEARNING FUNCTIONS --------------------




def eval_sol(sol:Tensor) -> int:
    """
    Evaluates the quality of a solution.
    /!\ This only is the true number of connections if the solution was created
    with side_importance = 1 /!\ 

    """


    board = sol[1:-1,1:-1]
    n_offset = sol[:-2,1:-1,:N_COLORS]
    s_offset = sol[2:,1:-1,N_COLORS:2*N_COLORS]
    e_offset = sol[1:-1,2:,2*N_COLORS:3*N_COLORS]
    w_offset = sol[1:-1,:-2,3*N_COLORS:4*N_COLORS]

    n_connections = torch.einsum('i j c , i j c -> i j', n_offset, board[:,:,:N_COLORS])
    s_connections = torch.einsum('i j c , i j c -> i j', s_offset, board[:,:,N_COLORS: 2*N_COLORS])
    e_connections = torch.einsum('i j c , i j c -> i j', e_offset, board[:,:,2*N_COLORS: 3*N_COLORS])
    w_connections = torch.einsum('i j c , i j c -> i j', w_offset, board[:,:,3*N_COLORS: 4*N_COLORS])

    total_connections = n_connections.sum() + s_connections.sum() + e_connections.sum() + w_connections.sum()

    return total_connections



# -------------------- UTILS --------------------

def _gen_swp_idx(size:int, neighborhood_size:int,no_offset:bool = True) -> Tensor:
    """
    Generates the swap indexes for every 2-swap available.
    For efficiency, the borders (tiles containing grey) can only be swapped
    on the border

    output shape : [batch_size, 2, 2] 

    """

    swp_idx = torch.zeros(neighborhood_size,2,2, dtype=int)

    k = 0
    for i in range(size**2 - 1):

        for j in range(i+1, size**2):

            i1 = i // size
            j1 = i % size
            
            i2 = j // size
            j2 = j % size

            on_sides1 = (i1 == 0) or (i1 == MAX_BSIZE - 1) or (j1 == 0) or (j1 == MAX_BSIZE - 1)
            on_sides2 = (i2 == 0) or (i2 == MAX_BSIZE - 1) or (j2 == 0) or (j2 == MAX_BSIZE - 1)

            # only swap borders with borders
            if on_sides1 != on_sides2:
                continue

            swp_idx[k] = torch.tensor([[i1,j1],[i2,j2]])
            k += 1
    print(k)
    # the board does not cover the whole state, we only swap playable tiles
    if no_offset:
        return swp_idx
    
    offset = (MAX_BSIZE - size) // 2 + 1
    return swp_idx + offset

def _gen_masks(size, swp_idx):
    """
    Generates masks according to the given indexes.

    output shape: [2, batch_size, board_size, board_size]

    Where the first dimension represents the first or second element swapped.
    """

    mask_batch = torch.zeros(swp_idx.shape[0], size + 2, size + 2, 4 * N_COLORS ,2, dtype=torch.bool)

    i = swp_idx[:,0]
    j = swp_idx[:,1]

    mask_batch[torch.arange(mask_batch.size()[0]), i[:,0], i[:,1],:,0] = 1
    mask_batch[torch.arange(mask_batch.size()[0]), j[:,0], j[:,1],:,1] = 1

    mask_batch = rearrange(mask_batch,'b i j c k -> k b i j c',k=2)

    return mask_batch == 1

def to_tensor(sol:list, side_importance:int = 5) -> Tensor:
    """
    Converts solutions from list format to a torch Tensor.

    Tensor format:
    [MAX_BSIZE, MAX_BSIZE, N_COLORS * 4]
    Each tile is represented as a vector, consisting of concatenated one hot encoding of the colors
    in the order  N - S - E - W . 
    If there were 4 colors a grey tile would be :
        N       S       E       W
    [1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0]

    """
    tens = torch.zeros((MAX_BSIZE + 2,MAX_BSIZE + 2,4*N_COLORS), device='cuda' if torch.cuda.is_available() else 'cpu')

    # Tiles around the board
    # To make sure the policy learns that the gray tiles are always one the border,
    # the reward for connecting to those tiles is bigger.
    tens[0,:,N_COLORS + GRAY] = side_importance
    tens[:,0,N_COLORS * 2 + GRAY] = side_importance
    tens[MAX_BSIZE+1,:,GRAY] = side_importance
    tens[:,MAX_BSIZE+1,N_COLORS * 3 + GRAY] = side_importance


    b_size = int(len(sol)**0.5)

    # center the playable board as much as possible
    offset = (MAX_BSIZE - b_size) // 2 + 1
    #one hot encode the colors
    for i in range(offset, offset + b_size):
        for j in range(offset, offset + b_size):

            tens[i,j,:] = 0

            for dir in range(4):
                tens[i,j, dir * N_COLORS + sol[(i - offset) * b_size + (j-offset)][dir]] = 1
    
    return tens


def to_list(sol:Tensor) -> list:


    list_sol = []
    for i in range(1, sol.shape[0] - 1):
        for j in range(1, sol.shape[0] - 1):

            temp = torch.where(sol[i,j] == 1)[0]

            for dir in range(4):
                temp[dir] -= dir * N_COLORS
            
            list_sol.append(tuple(temp.tolist()))

    return list_sol