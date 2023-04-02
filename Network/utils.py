import argparse
import random
import sys
import torch
from torch import Tensor
from einops import rearrange, repeat
from eternity import EternityPuzzle
from param import *

from init_solver import init_boost_conflicts


# -------------------- UTILS --------------------

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument('--instance', type=str, default='input')
    parser.add_argument('--hotstart', type=str, default=False)

    return parser.parse_args()


def initialize_sol(instance_file:str, boost_conflicts:bool=True):

    pz = EternityPuzzle(instance_file)
    random.shuffle(pz.piece_list)

    if boost_conflicts:
        sol,_ = init_boost_conflicts(pz)
        tens = to_tensor(sol)
    else:
        tens = to_tensor(pz.piece_list.copy(),ENCODING)

    return tens, pz.board_size



def gen_swp_idx(size:int, no_offset:bool = False) -> Tensor:
    """
    Generates the swap indexes for every 2-swap available.
    For efficiency, the borders (tiles containing grey) can only be swapped
    on the border

    output shape : [batch_size, 2, 2] 

    """

    # swp_idx = torch.zeros(neighborhood_size,2,2, dtype=int)
    swp_idx = []

    k = 0
    for i in range(size**2 - 1):

        for j in range(i+1, size**2):


            i1 = i // size
            j1 = i % size
            
            i2 = j // size
            j2 = j % size

            if  max(abs(i1-i2),abs(j1-j2)) > SWAP_RANGE:
                continue
            
            on_sides1 = (i1 == 0) or (i1 == size - 1) or (j1 == 0) or (j1 == size - 1)
            on_sides2 = (i2 == 0) or (i2 == size - 1) or (j2 == 0) or (j2 == size - 1)

            # only swap borders with borders
            if on_sides1 != on_sides2:
                continue
            
            corners={(0,0),(0,size-1),(size-1,0),(size-1,size-1)}

            on_corner1 = (i1,j1) in corners
            on_corner2 = (i2,j2) in corners
            
            if on_corner1 != on_corner2:
                continue

            swp_idx.append([[i1,j1],[i2,j2]])
            k += 1
    swp_idx = torch.tensor(swp_idx,dtype=int,device='cuda' if torch.cuda.is_available() else 'cpu')
    # the board does not cover the whole state, we only swap playable tiles
    

    offset = (MAX_BSIZE + 2 - size) // 2

    return swp_idx + offset

def gen_masks(size, swp_idx):
    """
    Generates masks according to the given indexes.
    FIXME:
    Where the first dimension represents the first or second element swapped.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    swp_mask_batch = torch.zeros(swp_idx.shape[0], MAX_BSIZE + 2, MAX_BSIZE + 2, 4 * COLOR_ENCODING_SIZE ,2, dtype=torch.bool,device=device)

    i = swp_idx[:,0]
    j = swp_idx[:,1]

    swp_mask_batch[torch.arange(swp_mask_batch.size()[0]), i[:,0],i[:,1],:,0] = 1
    swp_mask_batch[torch.arange(swp_mask_batch.size()[0]), j[:,0],j[:,1],:,1] = 1

    swp_mask_batch = rearrange(swp_mask_batch,'b i j c k -> k b i j c',k=2) == 1

    rot_mask_batch  = torch.zeros((3 * size**2 ,MAX_BSIZE+2,MAX_BSIZE+2,4*COLOR_ENCODING_SIZE),device=device)

    offset = (MAX_BSIZE + 2 - size) // 2
    for i in range(offset, offset + size):
        for j in range(offset, offset + size):
            for dir in range(3):
                rot_mask_batch[(i-offset)* size * 3+ (j-offset) * 3 + dir,i,j,:] = 1
            
    rot_mask_batch = rot_mask_batch == 1

    return swp_mask_batch, rot_mask_batch

def binary(x: Tensor, bits):
    mask = 2**torch.arange(bits)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).to(UNIT)

def to_tensor(list_sol:list, encoding = 'binary',gray_borders:bool=True) -> Tensor:
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

    # To be able to rotate tiles easily, it is better to have either NESW or NWSE
    orientation = [0,2,1,3]

    list_sol = list_sol.copy()
    b_size = int(len(list_sol)**0.5)


    if gray_borders:
        sol = []
        l = len(list_sol)

        for k in range(l):

            # if on the border
            if k < b_size or k % b_size == 0 or k % (b_size) == b_size-1 or k > b_size * (b_size - 1):
                border = True
            
            else:
                border = False

            for i in range(len(list_sol)):
                
                tile = list_sol[i]

                if (border and GRAY in tile) or (not border and not (GRAY in tile)):
                    sol.append(tile)
                    list_sol.pop(i)
                    break

    b_size = int(len(sol)**0.5)

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



        # center the playable board as much as possible
        offset = (MAX_BSIZE + 2 - b_size) // 2
        #one hot encode the colors
        for i in range(offset, offset + b_size):
            for j in range(offset, offset + b_size):

                tens[i,j,:] = 0

                for d in range(4):
                    dir = orientation[d]
                    tens[i,j, d * color_enc_size:(d+1) * color_enc_size] = binary(torch.tensor(sol[(i - offset) * b_size + (j-offset)][dir]),color_enc_size)

    elif encoding == 'ordinal':
        tens = torch.zeros((MAX_BSIZE + 2,MAX_BSIZE + 2,4), device='cuda' if torch.cuda.is_available() else 'cpu',dtype=UNIT)

        # Tiles around the board
        # To make sure the policy learns that the gray tiles are always one the border,
        # the reward for connecting to those tiles is bigger.
        tens[0,:,1] = 0
        tens[:,0,2] = 0
        tens[MAX_BSIZE+1,:,0] = 0
        tens[:,MAX_BSIZE+1,3] = 0


        # center the playable board as much as possible
        offset = (MAX_BSIZE - b_size) // 2 + 1
        #one hot encode the colors
        for i in range(offset, offset + b_size):
            for j in range(offset, offset + b_size):

                tens[i,j,:] = 0

                for d in range(4):
                    dir = orientation[d]
                    tens[i,j,d] = torch.tensor(sol[(i - offset) * b_size + (j-offset)][dir])
        
        tens.unsqueeze(-1)


    else:

        tens = torch.zeros((MAX_BSIZE + 2,MAX_BSIZE + 2,4*N_COLORS), device='cuda' if torch.cuda.is_available() else 'cpu',dtype=UNIT)

        tens[0,:,N_COLORS + GRAY] = 1
        tens[:,0,N_COLORS * 2 + GRAY] = 1
        tens[MAX_BSIZE+1,:,GRAY] = 1
        tens[:,MAX_BSIZE+1,N_COLORS * 3 + GRAY] = 1


        # center the playable board as much as possible
        offset = (MAX_BSIZE - b_size) // 2 + 1
        #one hot encode the colors
        for i in range(0,MAX_BSIZE+2):
            for j in range(0,MAX_BSIZE+2):

                if i >= offset and i < offset+b_size and j >= offset and j < offset+b_size:
                    tens[i,j,:] = 0

                    for d in range(4):
                        dir = orientation[d]
                        tens[i,j, d * N_COLORS + sol[(i - offset) * b_size + (j-offset)][dir]] = 1
                    
                else:
                    for dir in range(4):
                        tens[i,j, orientation[dir] * N_COLORS] = 1
                    

            
        

    return tens

def base10(x:Tensor):
    s = 0
    for i in range(x.size()[0]):
        s += x[i] * 2**i
    
    return int(s)

def pprint(state,bsize):
    offset = (MAX_BSIZE + 2 - bsize) // 2

    if state.size()[0] != MAX_BSIZE + 2:
        for s in state:
            print(s[offset:offset+bsize,offset:offset+bsize])

    else:
        print(state[offset:offset+bsize,offset:offset+bsize])
        


def to_list(sol:torch.Tensor,bsize:int) -> list:

    orientation = [0,2,1,3]

    list_sol = []

    sol.int()

    offset = (MAX_BSIZE + 2 - bsize) // 2

    if ENCODING == 'binary':

        for i in range(offset, offset + bsize):
            for j in range(offset, offset + bsize):

                temp = [0]*4
                for d in range(4):
                    dir = orientation[d]
                    temp[d] = base10(sol[i,j,dir*COLOR_ENCODING_SIZE:(dir+1)*COLOR_ENCODING_SIZE])
                
                list_sol.append(tuple(temp))
    
    elif ENCODING == 'ordinal':

        for i in range(offset, offset + bsize):
            for j in range(offset, offset + bsize):

                temp = [0] * 4
                for d in range(4):
                    dir = orientation[d]
                    temp[d] = sol[i,j,dir].item()

                list_sol.append(tuple(temp))

    if ENCODING == 'one_hot':

        for i in range(offset, offset + bsize):
            for j in range(offset, offset + bsize):

                temp = [0] * 4

                for d in range(4):
                    dir = orientation[d]
                    temp[d] = torch.where(sol[i,j,dir*COLOR_ENCODING_SIZE:(dir+1)*COLOR_ENCODING_SIZE] == 1)[0].item()
                list_sol.append(tuple(temp))

    return list_sol


def soft_lj(x):
    return ((1/x)**2- 2 *(1/x)) + 1.5