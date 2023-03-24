from collections import namedtuple
from eternity_puzzle import EternityPuzzle
import torch
from DRQN import *
from solver_random import solve_random


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


def solve_advanced(eternity_puzzle:EternityPuzzle, hotstart:str = None):

    """
    Your solver for the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """






    # -------------------- PARAMS -------------------- 

    BSIZE = eternity_puzzle.board_size

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    LR = 10**-4
    MEM_SIZE = 40000
    ALPHA = 0.4
    BATCH_SIZE = 50

    Transition = namedtuple('Transition',
                                     ('state', 'next_state', 'reward','weights','indexes'))


    # -------------------- NETWORK INIT -------------------- 

    if hotstart:
        policy_net = DQN(BSIZE+1, BSIZE+1, 1, device).to(device)
        policy_net.load_state_dict(torch.load(hotstart))
    else:
        policy_net = DQN(BSIZE, BSIZE, 1, device).to(device)
    
    target_net = DQN(BSIZE, BSIZE, 1, device).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(),amsgrad=True,lr = LR)

    memory = PrioritizedReplayMemory(
        size = MEM_SIZE,
        Transition = Transition,
        alpha = ALPHA,
        batch_size = BATCH_SIZE

    )

    
    # -------------------- GAME INIT --------------------

    init_sol, init_score = solve_random(eternity_puzzle)

    state = to_tensor(init_sol)


    raise BaseException



# -------------------- LEARNING FUNCTIONS --------------------


def eval_sol(sol:torch.Tensor) -> int:
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


def to_tensor(sol:list, side_importance:int = 5) -> torch.Tensor:
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
    tens = torch.zeros((MAX_BSIZE + 2,MAX_BSIZE + 2,4*N_COLORS),dtype=int)

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

            print(i,j)
            tens[i,j,:] = 0

            for dir in range(4):
                tens[i,j, dir * N_COLORS + sol[(i - offset) * b_size + (j-offset)][dir]] = 1
    

    return tens
