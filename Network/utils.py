import argparse
import random
import sys
import torch
from torch import Tensor
from einops import rearrange, repeat
from eternity import EternityPuzzle
from param import *


# -------------------- UTILS --------------------

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument('--instance', type=str, default='input')
    parser.add_argument('--hotstart', type=str, default=False)

    return parser.parse_args()


def initialize_sol(instance_file:str, device):

    pz = EternityPuzzle(instance_file)
    n_tiles = len(pz.piece_list)
    tiles = rearrange(to_tensor(pz.piece_list),'h w d -> (h w) d').to(device)
    return torch.zeros((pz.board_size+2,pz.board_size+2,4*COLOR_ENCODING_SIZE),device=device), tiles, n_tiles




def pprint(state,bsize):
    offset = (PADDED_SIZE - bsize) // 2

    if state.size()[0] != PADDED_SIZE:
        for s in state:
            print(s[offset:offset+bsize,offset:offset+bsize])

    else:
        print(state[offset:offset+bsize,offset:offset+bsize])
        



def ucb(q,count,step):

    return q + 0 * torch.sqrt(-torch.log((count + 0.1)/(step + 0.1)))

def binary(x: Tensor, bits):
    mask = 2**torch.arange(bits)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).to(UNIT)


def to_tensor(list_sol:list, encoding = ENCODING,gray_borders:bool=False) -> Tensor:
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
    else:
        sol = list_sol
    b_size = int(len(sol)**0.5)

    if encoding == 'binary':
        color_enc_size = ceil(torch.log2(torch.tensor(N_COLORS)))
        tens = torch.zeros((b_size,b_size,4*color_enc_size), device='cuda' if torch.cuda.is_available() else 'cpu',dtype=UNIT)

        # Tiles around the board
        # To make sure the policy learns that the gray tiles are always one the border,
        # the reward for connecting to those tiles is bigger.
        tens[0,:,color_enc_size:2*color_enc_size] = binary(torch.tensor(GRAY),color_enc_size)
        tens[:,0,2*color_enc_size:color_enc_size*3] = binary(torch.tensor(GRAY),color_enc_size)
        tens[b_size-1,:,:color_enc_size] = binary(torch.tensor(GRAY),color_enc_size)
        tens[:,b_size-1,3*color_enc_size:] = binary(torch.tensor(GRAY),color_enc_size)



        # center the playable board as much as possible
        #one hot encode the colors
        for i in range(b_size):
            for j in range(b_size):

                tens[i,j,:] = 0

                for d in range(4):
                    dir = orientation[d]
                    tens[i,j, d * color_enc_size:(d+1) * color_enc_size] = binary(torch.tensor(sol[i * b_size + j][dir]),color_enc_size)

    elif encoding == 'ordinal':
        tens = torch.zeros((b_size,b_size,4), device='cuda' if torch.cuda.is_available() else 'cpu',dtype=UNIT)

        # Tiles around the board
        # To make sure the policy learns that the gray tiles are always one the border,
        # the reward for connecting to those tiles is bigger.
        tens[0,:,1] = 0
        tens[:,0,2] = 0
        tens[b_size-1,:,0] = 0
        tens[:,b_size-1,3] = 0


        # center the playable board as much as possible
        #one hot encode the colors
        for i in range(b_size):
            for j in range(b_size):

                tens[i,j,:] = 0

                for d in range(4):
                    dir = orientation[d]
                    tens[i,j,d] = torch.tensor(sol[i * b_size + j][dir])
        
        tens.unsqueeze(-1)


    else:

        tens = torch.zeros((b_size,b_size,4*N_COLORS), device='cuda' if torch.cuda.is_available() else 'cpu',dtype=UNIT)

        tens[0,:,N_COLORS + GRAY] = 1
        tens[:,0,N_COLORS * 2 + GRAY] = 1
        tens[b_size-1,:,GRAY] = 1
        tens[:,b_size-1,N_COLORS * 3 + GRAY] = 1


        # center the playable board as much as possible
        #one hot encode the colors
        for i in range(b_size):
            for j in range(b_size):

                if i >= 0 and i < b_size and j >= 0 and j < b_size:
                    tens[i,j,:] = 0

                    for d in range(4):
                        dir = orientation[d]
                        tens[i,j, d * N_COLORS + sol[i * b_size + j][dir]] = 1
                    
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

    offset = 1

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