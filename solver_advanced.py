from collections import namedtuple
import gc
import os
import sys

from einops import repeat, rearrange
from eternity_puzzle import EternityPuzzle
import torch
from solver_random import solve_random
from math import ceil, comb

sys.path.append('./Network')


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
ENCODING = 'ordinal'
COLOR_ENCODING_SIZE = 4 * 5
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

# FIXME:
# FIXME: INIT SOL WITH GRAY TILES ON THE BORDERS
# FIXME: INIT SOL WITH GRAY TILES ON THE BORDERS
# FIXME: INIT SOL WITH GRAY TILES ON THE BORDERS
# FIXME: INIT SOL WITH GRAY TILES ON THE BORDERS
# FIXME: INIT SOL WITH GRAY TILES ON THE BORDERS
# FIXME: INIT SOL WITH GRAY TILES ON THE BORDERS
# FIXME: INIT SOL WITH GRAY TILES ON THE BORDERS
# FIXME: INIT SOL WITH GRAY TILES ON THE BORDERS
# FIXME: INIT SOL WITH GRAY TILES ON THE BORDERS
# FIXME: INIT SOL WITH GRAY TILES ON THE BORDERS
# FIXME: INIT SOL WITH GRAY TILES ON THE BORDERS
# FIXME: INIT SOL WITH GRAY TILES ON THE BORDERS
# FIXME: INIT SOL WITH GRAY TILES ON THE BORDERS
# FIXME: INIT SOL WITH GRAY TILES ON THE BORDERS
# FIXME: INIT SOL WITH GRAY TILES ON THE BORDERS
# FIXME: INIT SOL WITH GRAY TILES ON THE BORDERS
# FIXME: INIT SOL WITH GRAY TILES ON THE BORDERS
# FIXME: INIT SOL WITH GRAY TILES ON THE BORDERS
# FIXME: INIT SOL WITH GRAY TILES ON THE BORDERS
# FIXME: INIT SOL WITH GRAY TILES ON THE BORDERS
# FIXME: INIT SOL WITH GRAY TILES ON THE BORDERS
# FIXME: INIT SOL WITH GRAY TILES ON THE BORDERS
# FIXME:
# FIXME:
# FIXME:
# FIXME:




def solve_advanced(eternity_puzzle:EternityPuzzle, hotstart:str = None):

    """
    Your solver for the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    TODO: Only 2-swap the inner tiles (not the borders), reduces the neighborhood size from 
          nCr(16*16,2) to nCr(14*14,2) approx.
    """

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

    state = to_tensor(init_sol,'binary')
    print(state.size())

    torch.save(state, 'binary')
    state = to_tensor(init_sol,'ordinal')
    print(state.size())

    torch.save(state, 'ordinal')
    state = to_tensor(init_sol,'one_hot')
    print(state.size())
    torch.save(state, 'one_hot')

    raise OSError

def binary(x: torch.Tensor, bits):
    mask = 2**torch.arange(bits)
    return x.unsqueeze(-1).bitwise_and(mask).to(UNIT)


def to_tensor(sol:list, encoding = 'binary') -> torch.Tensor:
    """
    Converts solutions from list format to a torch Tensor.

    Tensor format:
    [MAX_BSIZE, MAX_BSIZE, N_COLORS * 4]
    Each tile is represented as a vector, consisting of concatenated one hot encoding of the colors
    in the order  N - S - E - W . 
    If there were 4 colors a grey tile would be :
        N       S       E       W
    [1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0]

    TODO: convert the list to tensor once 

    """

    if encoding == 'binary':
        color_enc_size = ceil(torch.log2(torch.tensor(N_COLORS)))
        tens = torch.zeros((MAX_BSIZE + 2,MAX_BSIZE + 2,4*color_enc_size), device='cuda' if torch.cuda.is_available() else 'cpu',dtype=UNIT)

        # Tiles around the board
        # To make sure the policy learns that the gray tiles are always one the border,
        # the reward for connecting to those tiles is bigger.
        tens[0,:,color_enc_size:2*color_enc_size] = binary(torch.tensor(GRAY),color_enc_size)
        tens[:,0,2*color_enc_size:color_enc_size*3] = binary(torch.tensor(GRAY),color_enc_size)
        tens[MAX_BSIZE+1,:,:color_enc_size] = binary(torch.tensor(GRAY),color_enc_size)
        tens[:,MAX_BSIZE+1,3*color_enc_size:] = binary(torch.tensor(GRAY),color_enc_size)


        b_size = int(len(sol)**0.5)

        # center the playable board as much as possible
        offset = (MAX_BSIZE - b_size) // 2 + 1
        #one hot encode the colors
        for i in range(offset, offset + b_size):
            for j in range(offset, offset + b_size):

                tens[i,j,:] = 0

                for dir in range(4):
                    tens[i,j, dir * color_enc_size:(dir+1) * color_enc_size] = binary(torch.tensor(sol[(i - offset) * b_size + (j-offset)][dir]),color_enc_size)

    elif encoding == 'ordinal':
        tens = torch.zeros((MAX_BSIZE + 2,MAX_BSIZE + 2,4), device='cuda' if torch.cuda.is_available() else 'cpu',dtype=UNIT)

        # Tiles around the board
        # To make sure the policy learns that the gray tiles are always one the border,
        # the reward for connecting to those tiles is bigger.
        tens[0,:,1] = 0
        tens[:,0,2] = 0
        tens[MAX_BSIZE+1,:,0] = 0
        tens[:,MAX_BSIZE+1,3] = 0


        b_size = int(len(sol)**0.5)

        # center the playable board as much as possible
        offset = (MAX_BSIZE - b_size) // 2 + 1
        #one hot encode the colors
        for i in range(offset, offset + b_size):
            for j in range(offset, offset + b_size):

                tens[i,j,:] = 0

                for dir in range(4):
                    tens[i,j,dir] = torch.tensor(sol[(i - offset) * b_size + (j-offset)][dir])
        
        tens.unsqueeze(-1)


    else:

        tens = torch.zeros((MAX_BSIZE + 2,MAX_BSIZE + 2,4*N_COLORS), device='cuda' if torch.cuda.is_available() else 'cpu',dtype=UNIT)

        tens[0,:,N_COLORS + GRAY] = 1
        tens[:,0,N_COLORS * 2 + GRAY] = 1
        tens[MAX_BSIZE+1,:,GRAY] = 1
        tens[:,MAX_BSIZE+1,N_COLORS * 3 + GRAY] = 1


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


def to_list(sol:torch.Tensor) -> list:


    list_sol = []
    for i in range(1, sol.shape[0] - 1):
        for j in range(1, sol.shape[0] - 1):

            temp = torch.where(sol[i,j] == 1)[0]

            for dir in range(4):
                temp[dir] -= dir * N_COLORS
            
            list_sol.append(tuple(temp.tolist()))

    return list_sol

