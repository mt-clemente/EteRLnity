import torch
from torch import Tensor
from Trajectories import *
from math import exp
from utils import *
from param import *
import wandb

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

    connections = get_connections(state,bsize,step)
    max_connections = (bsize + 1) * bsize * 2

    return max_connections - connections



def get_connections(state:Tensor, bsize:int, step:int = 0) -> int:
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


    return total_connections
