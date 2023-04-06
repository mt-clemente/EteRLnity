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


def initialize_sol(instance_file:str, device):

    pz = EternityPuzzle(instance_file)
    n_tiles = len(pz.piece_list)
    tiles = torch.tensor(pz.piece_list,device=device).repeat(4,1)
    for dir in range(3):
        tiles[dir*n_tiles:(dir+1)*n_tiles] = tiles[dir*n_tiles:(dir+1)*n_tiles].roll(dir+1,1)
    return torch.zeros((pz.board_size+2,pz.board_size+2,4),device=device), tiles, n_tiles




def pprint(state,bsize):
    offset = (PADDED_SIZE - bsize) // 2

    if state.size()[0] != PADDED_SIZE:
        for s in state:
            print(s[offset:offset+bsize,offset:offset+bsize])

    else:
        print(state[offset:offset+bsize,offset:offset+bsize])
        


def to_list(sol:torch.Tensor,bsize:int) -> list:

    orientation = [0,2,1,3]
    list_sol = []

    sol.int()
    offset = (PADDED_SIZE - bsize) // 2

    for i in range(offset, offset + bsize):
        for j in range(offset, offset + bsize):

            temp = [0] * 4
            for d in range(4):
                dir = orientation[d]
                temp[d] = sol[i,j,dir].item()

            list_sol.append(tuple(temp))

    
    return list_sol


def place_tile(state:Tensor,tile:Tensor,step:int):
    state = state.clone()
    bsize = state.size()[0] - 2
    state[step // bsize + 1, step % bsize + 1,:] = tile
    return state


def ucb(q,count,step):

    return q + 0 * torch.sqrt(-torch.log((count + 0.1)/(step + 0.1)))