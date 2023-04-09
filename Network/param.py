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

GAE_LAMBDA = 0.988
ENTROPY_WEIGHT = 0.01
VALUE_WEIGHT = .5
CONV_SIZES = [32,64,128]
KERNEL_SIZES = [3,4,5]
HIDDEN_SIZE = 128

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
  
EPOCHS = 10
CHECKPOINT_PERIOD = 256*200
MINIBATCH_SIZE = 64
HORIZON = 2 * 256
OPT_EPSILON = 1e-6
LR = 6e-5
GAMMA = 0.95
CLIP_EPS = 0.2

CONFIG = {
    'encoding':ENCODING,
    'unit':UNIT,
    'Batch size':MINIBATCH_SIZE,
    'Gamma':GAMMA,
}