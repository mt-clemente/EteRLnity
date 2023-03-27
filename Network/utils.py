import sys
import torch
from torch import Tensor
from einops import rearrange, repeat
from eternity_puzzle import EternityPuzzle
from param import *


import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.lines import Line2D
import numpy as np


# -------------------- UTILS --------------------

def initialize_sol(instance_file:str):

    pz = EternityPuzzle(instance_file)
    
    tens = to_tensor(pz.piece_list.copy(),ENCODING)

    sol = to_list(tens,ENCODING)
    
    raise OSError


def gen_swp_idx(size:int, no_offset:bool = True) -> Tensor:
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
            
            on_sides1 = (i1 == 0) or (i1 == MAX_BSIZE - 1) or (j1 == 0) or (j1 == MAX_BSIZE - 1)
            on_sides2 = (i2 == 0) or (i2 == MAX_BSIZE - 1) or (j2 == 0) or (j2 == MAX_BSIZE - 1)

            # only swap borders with borders
            if on_sides1 != on_sides2:
                continue

            swp_idx.append([[i1,j1],[i2,j2]])
            k += 1
    print(k)
    swp_idx = torch.tensor(swp_idx,dtype=int,device='cuda' if torch.cuda.is_available() else 'cpu')
    # the board does not cover the whole state, we only swap playable tiles
    if no_offset:
        return swp_idx
    
    offset = (MAX_BSIZE - size) // 2 + 1
    return swp_idx + offset

def gen_masks(size, swp_idx):
    """
    Generates masks according to the given indexes.

    output shape: [2, batch_size, board_size, board_size]

    Where the first dimension represents the first or second element swapped.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    swp_mask_batch = torch.zeros(swp_idx.shape[0], size + 2, size + 2, 4 * COLOR_ENCODING_SIZE ,2, dtype=torch.bool,device=device)

    i = swp_idx[:,0]
    j = swp_idx[:,1]

    swp_mask_batch[torch.arange(swp_mask_batch.size()[0]), i[:,0], i[:,1],:,0] = 1
    swp_mask_batch[torch.arange(swp_mask_batch.size()[0]), j[:,0], j[:,1],:,1] = 1

    swp_mask_batch = rearrange(swp_mask_batch,'b i j c k -> k b i j c',k=2) == 1

    rot_mask_batch  = torch.zeros((MAX_BSIZE**2,MAX_BSIZE+2,MAX_BSIZE+2,4*COLOR_ENCODING_SIZE),device=device)

    for i in range(MAX_BSIZE):
        for j in range(MAX_BSIZE):
            rot_mask_batch[i*MAX_BSIZE+j,i,j,:] = 1
            
    rot_mask_batch = repeat(rot_mask_batch,'k i j c -> (a k) i j c',a=4) == 1

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

def base10(x:Tensor):
    s = 0
    for i in range(x.size()[0]):
        s += x[i] * 2**i
    
    return int(s)

def to_list(sol:torch.Tensor, encoding='binary') -> list:

    list_sol = []

    if encoding == 'binary':

        for i in range(1, sol.shape[0] - 1):
            for j in range(1, sol.shape[0] - 1):

                temp = [0]*4
                for dir in range(4):
                    print(sol[i,j])
                    temp[dir] = base10(sol[i,j,dir*COLOR_ENCODING_SIZE:(dir+1)*COLOR_ENCODING_SIZE])
                
                list_sol.append(tuple(temp))
    
    elif encoding == 'ordinal':

        for i in range(1, sol.shape[0] - 1):
            for j in range(1, sol.shape[0] - 1):

                temp = sol[i,j]
                list_sol.append(tuple(temp.tolist()))

    if encoding == 'one_hot':

        for i in range(1, sol.shape[0] - 1):
            for j in range(1, sol.shape[0] - 1):

                temp = torch.where(sol[i,j] == 1)[0]
                for dir in range(4):
                    temp[dir] -= dir * COLOR_ENCODING_SIZE
                
                list_sol.append(tuple(temp.tolist()))

    return list_sol













def display_solution(self, solution, output_file):

    self = EternityPuzzle(sys.argv[-1])

    if len(solution) < self.n_piece:
        solution = solution + [(WHITE, WHITE, WHITE, WHITE)] * (self.n_piece - len(solution))

    origin = 0
    size = self.board_size + 2

    color_dict = self.build_color_dict()

    fig, ax = plt.subplots()

    n_total_conflict = self.get_total_n_conflict(solution)

    n_internal_conflict = 0

    for j in range(size):  # y-axis
        for i in range(size):  # x-axis
            valid_draw = [0, size - 1]
            if i in valid_draw or j in valid_draw:
                ax.add_patch(patches.Rectangle((i, j), i + 1, j + 1, fill=True, facecolor=color_dict[GRAY],
                                                edgecolor=color_dict[BLACK]))
            else:
                # ax.add_patch(patches.Rectangle((i, j), i + 1, j + 1, fill=True, facecolor='white', edgecolor='k'))

                left_bot = (i, j)
                right_bot = (i + 1, j)
                right_top = (i + 1, j + 1)
                left_top = (i, j + 1)
                middle = (i + 0.5, j + 0.5)

                instructions = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]

                triangle_south_path = Path([left_bot, middle, right_bot, left_bot], instructions)
                triangle_east_path = Path([right_top, middle, right_bot, right_top], instructions)
                triangle_north_path = Path([right_top, middle, left_top, right_top], instructions)
                triangle_west_path = Path([left_bot, middle, left_top, left_bot], instructions)

                is_triangle_south_valid = True
                is_triangle_north_valid = True
                is_triangle_east_valid = True
                is_triangle_west_valid = True

                k = self.board_size * (j - 1) + (i - 1)
                k_east = self.board_size * (j - 1) + (i - 2)
                k_south = self.board_size * (j - 2) + (i - 1)

                if i == 1:
                    is_triangle_west_valid = (solution[k][WEST] == GRAY)  # 1 for Gray
                elif i == size - 2:
                    is_triangle_east_valid = (solution[k][EAST] == GRAY)
                    is_triangle_west_valid = solution[k][WEST] == solution[k_east][EAST]
                else:
                    is_triangle_west_valid = solution[k][WEST] == solution[k_east][EAST]

                if j == 1:
                    is_triangle_south_valid = (solution[k][SOUTH] == GRAY)
                elif j == size - 2:
                    is_triangle_north_valid = (solution[k][NORTH] == GRAY)
                    is_triangle_south_valid = solution[k][SOUTH] == solution[k_south][NORTH]
                else:
                    is_triangle_south_valid = solution[k][SOUTH] == solution[k_south][NORTH]

                patch_south = patches.PathPatch(triangle_south_path, facecolor=color_dict[solution[k][SOUTH]],
                                                edgecolor=color_dict[BLACK])

                patch_north = patches.PathPatch(triangle_north_path, facecolor=color_dict[solution[k][NORTH]],
                                                edgecolor=color_dict[BLACK])

                patch_east = patches.PathPatch(triangle_east_path, facecolor=color_dict[solution[k][EAST]],
                                                edgecolor=color_dict[BLACK])

                patch_west = patches.PathPatch(triangle_west_path, facecolor=color_dict[solution[k][WEST]],
                                                edgecolor=color_dict[BLACK])

                if not is_triangle_south_valid:
                    line_zip = list(zip(left_bot, right_bot))
                    line = Line2D(line_zip[0], line_zip[1], color=color_dict[RED], lw=3)
                    ax.add_line(line)

                    if j != 1:
                        n_internal_conflict += 1

                if not is_triangle_north_valid:
                    line_zip = list(zip(left_top, right_top))
                    line = Line2D(line_zip[0], line_zip[1], color=color_dict[RED], lw=3)
                    ax.add_line(line)

                    if j != size - 2:
                        n_internal_conflict += 1

                if not is_triangle_west_valid:
                    line_zip = list(zip(left_bot, left_top))
                    line = Line2D(line_zip[0], line_zip[1], color=color_dict[RED], lw=3)
                    ax.add_line(line)

                    if i != 1:
                        n_internal_conflict += 1

                if not is_triangle_east_valid:
                    line_zip = list(zip(right_bot, right_top))
                    line = Line2D(line_zip[0], line_zip[1], color=color_dict[RED], lw=3)
                    ax.add_line(line)

                    if i != size - 2:
                        n_internal_conflict += 1

                ax.add_patch(patch_south)
                ax.add_patch(patch_north)
                ax.add_patch(patch_east)
                ax.add_patch(patch_west)

                k += 1

    plt.xlim(origin, size)
    plt.ylim(origin, size)

    title = 'Eternity of size %d X %d\n' \
            'Total connections: %d    Internal connections: %d\n' \
            'Total Valid connections: %d     Internal valid internal connections: %d\n' \
            'Total Invalid connections: %d    Internal invalid connections: %d' % \
            (self.board_size, self.board_size,
                self.n_total_connection, self.n_internal_connection,
                self.n_total_connection - n_total_conflict, self.n_internal_connection - n_internal_conflict,
                n_total_conflict, n_internal_conflict,
                )
    ax.set_title(title)

    plt.savefig(output_file)