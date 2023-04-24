import argparse
from math import exp
import torch
from torch import Tensor
from einops import rearrange
from Network.eternity import EternityPuzzle
from Network.param import *


# -------------------- UTILS --------------------

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument('--instance', type=str, default='input')
    parser.add_argument('--hotstart', type=str, default=False)

    return parser.parse_args()


def initialize_sol(pz:EternityPuzzle, device):
    """
    Initializes the solution. The first corner is already play to remove symmetries.
    """

    n_tiles = len(pz.piece_list)
    tiles = rearrange(to_tensor(pz.piece_list),'h w d -> (h w) d').to(device)

    # Play the first corner to remove symetries
    first_corner_idx = torch.argmax((tiles == 0).count_nonzero(dim=1))
    first_corner = tiles[first_corner_idx]

    state = torch.zeros((pz.board_size+2,pz.board_size+2,4*COLOR_ENCODING_SIZE),device=device)
    state, _, _ = place_tile(state,first_corner,0)

    tiles = tiles[torch.arange(tiles.size()[0],device=tiles.device) != first_corner_idx]

    return state, tiles, first_corner, n_tiles


def binary(x: Tensor, bits):
    mask = 2**torch.arange(bits)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).to(UNIT)


def to_tensor(list_sol:list, encoding = ENCODING,gray_borders:bool=False) -> Tensor:
    """
    Converts solutions from list format to a torch Tensor.
    Beware, the order of NSEW is not the same, as NSEW tiles cannot be rotated easily.
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
        


def to_list(sol:torch.Tensor,bsize:int) -> list:
    """
    Converts the tensor solution back to a list solution.
    """

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




# ------------- PLAY UTILS -------------


def place_tile(state:Tensor,tile:Tensor,ep_step:int,step_offset:int=0):
    """
    If you start with a prefilled board with k pieces, you need to place tiles at spot
    k + 1, hence the need for a step offset.
    """
    step = ep_step + step_offset
    state = state.clone()
    bsize = state.size()[0] - 2
    best_conflict = -10
    best_connect = -1
    for _ in range(4):
        tile = tile.roll(COLOR_ENCODING_SIZE,-1)
        state[step // bsize + 1, step % bsize + 1,:] = tile
        conflicts, connect = filling_connections(state,bsize,step)

        if connect > best_connect:
            best_state=state.clone()
            best_connect = connect
            best_conflict = conflicts

    return best_state, -best_conflict, best_connect

def streak(streak_length:int, n_tiles):
    return (2 - exp(-streak_length * 3/(0.8 * n_tiles)))



def filling_connections(state:Tensor, bsize:int, step):
    """
    Calculates the created conflicts and connections when placing one tile
    """
    i = step // bsize + 1
    j = step % bsize + 1
    west_tile_color = state[i,j-1,3*COLOR_ENCODING_SIZE:4*COLOR_ENCODING_SIZE]
    south_tile_color = state[i-1,j,:COLOR_ENCODING_SIZE]

    west_border_color = state[i,j,1*COLOR_ENCODING_SIZE:2*COLOR_ENCODING_SIZE]
    south_border_color = state[i,j,2*COLOR_ENCODING_SIZE:3*COLOR_ENCODING_SIZE]

    sides = 0
    connections = 0
    reward = 0

    sides += 1
    if j == 1:
        if torch.all(west_border_color == 0):
            reward += 2
            connections += 1
    
    elif torch.all(west_border_color == west_tile_color):
        connections += 1
        reward += 1

    sides += 1
    if i == 1:
        if torch.all(south_border_color == 0):
            connections += 1
            reward += 2
    
    elif torch.all(south_border_color == south_tile_color):
        connections += 1
        reward += 1
   
   
    if j == bsize:

        east_border_color = state[i,j,3*COLOR_ENCODING_SIZE:4*COLOR_ENCODING_SIZE]
        sides += 1
        if torch.all(east_border_color == 0):
            connections += 1
            reward += 2


    if i == bsize:

        north_border_color = state[i,j,:COLOR_ENCODING_SIZE]
        sides += 1
        if torch.all(north_border_color == 0):
            reward += 2
            connections += 1
    
    return sides - connections, connections





def get_conflicts(state:Tensor, bsize:int) -> int:

    connections = get_connections(state,bsize)
    max_connections = (bsize + 1) * bsize * 2

    return max_connections - connections



def get_connections(state:Tensor, bsize:int) -> int:
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
