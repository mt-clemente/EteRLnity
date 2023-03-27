import torch
from torch import Tensor
from math import comb
from einops import rearrange, repeat

def _gen_swp_idx(size:int) -> Tensor:
    """
    Generates the swap indexes for every 2-swap available.
    """

    swp_idx = torch.zeros((comb(size**2,2),2,2), dtype=int)

    k = 0
    for i in range(size**2 - 1):

        for j in range(i+1, size**2):

            i1 = i // size
            j1 = i % size
            
            i2 = j // size
            j2 = j % size

            swp_idx[k] = torch.tensor([[i1,j1],[i2,j2]])
            k += 1
    print(swp_idx.size())
    return swp_idx

def _gen_masks(size, swp_idx):
    """
    Generates masks according to the given indexes
    """

    mask_batch = torch.zeros(swp_idx.shape[0],size,size,2)

    i = swp_idx[:,0]
    j = swp_idx[:,1]



    mask_batch[torch.arange(mask_batch.size()[0]), i[:,0], i[:,1],0] = 1
    mask_batch[torch.arange(mask_batch.size()[0]), j[:,0], j[:,1],1] = 1

    mask_batch = rearrange(mask_batch,'b i j k -> k b i j',k=2)

    return mask_batch == 1



def swap_elements(state_batch:Tensor, swp_idx:Tensor, masks:Tensor, return_prev:bool = False):
    """
    Generate all given 2-swaps.

    Input
    ------------
    state_batch : a batch of states of dimension [batch_size, board_size, board_size, n_colors]
    swp_idx : a batch of coordinates of elements to be swapped of dimension [batch_size, 2, 2]
    """
    i = swp_idx[:,0]
    j = swp_idx[:,1]

    t1 = state_batch[torch.arange(state_batch.size()[0]), i[:,0], i[:,1]]
    t2 = state_batch[torch.arange(state_batch.size()[0]), j[:,0], j[:,1]]

    swapped_batch = state_batch.masked_scatter(masks[0], t2)
    swapped_batch = swapped_batch.masked_scatter(masks[1], t1)
    
    # Return the modified input state_batch of tensors
    return swapped_batch


""" swp_idx = _gen_swp_idx(2)
swap_scatter_mask = _gen_masks(2,swp_idx)

a = torch.arange(4).reshape((2,2))
a = repeat(a,'i j -> b i j', b = comb(4,2))
a = swap_elements(a,swp_idx,swap_scatter_mask)
print("\n\n\n")
print(a) """


def binary(x, bits):
    mask = 2**torch.arange(bits)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

print(binary(torch.tensor(5),5))