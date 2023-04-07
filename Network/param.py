from math import log2,ceil
import torch

# -------------------- PUZZLE SETTINGS -------------------- 

PATH = '/home/wsl/Polymtl/H23/INF6201/Projet/Network'


# global 2-swap gives a size 32640 neighborhood which can be too much
# for the GPU. Capping the swapping range helps reduce the neighborhood
# without losing connectivity.
SWAP_RANGE = 2

MAX_BSIZE = 16

PADDED_SIZE = MAX_BSIZE + 2
NORTH = 0
SOUTH = 1
WEST = 2
EAST = 3

GRAY = 0
BLACK = 23
RED = 24
WHITE = 25
N_COLORS = 23


# -------------------- NETWORK SETTINGS -------------------- 


HIDDEN_SIZES = [512,256]
KERNEL_SIZES = [3,2]

# -------------------- TRAINING SETTINGS -------------------- 
UNIT = torch.float

ENCODING = 'binary'

if ENCODING == 'binary':
    COLOR_ENCODING_SIZE = ceil(log2(N_COLORS))
elif ENCODING == 'ordinal':
    COLOR_ENCODING_SIZE = 1
elif ENCODING == 'one_hot':
    COLOR_ENCODING_SIZE = N_COLORS
else:
    raise ValueError(f"Encoding {ENCODING} not supported")
  
EPOCHS = 20
CHECKPOINT_PERIOD = 10000
BATCH_SIZE = 64
OPT_EPSILON = 1e-6
LR = 1e-6
GAMMA = 0.92
CLIP_RATIO = 0.2

CONFIG = {
    'encoding':ENCODING,
    'unit':UNIT,
    'Batch size':BATCH_SIZE,
    'Gamma':GAMMA,
}