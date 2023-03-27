from math import log2,ceil
import torch


# -------------------- PUZZLE SETTINGS -------------------- 

PATH = '/home/wsl/Polymtl/H23/INF6201/Projet/Network'


# global 2-swap gives a size 32640 neighborhood which can be too much
# for the GPU. Capping the swapping range helps reduce the neighborhood
# without losing connectivity.
SWAP_RANGE = 2

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
UNIT = torch.half

ENCODING = 'ordinal'

if ENCODING == 'binary':
    COLOR_ENCODING_SIZE = ceil(log2(N_COLORS))
elif ENCODING == 'ordinal':
    COLOR_ENCODING_SIZE = 1
elif ENCODING == 'one_hot':
    COLOR_ENCODING_SIZE = N_COLORS
else:
    raise ValueError(f"Encoding {ENCODING} not supported")
  
PARTIAL_OBSERVABILITY = 0.5
BATCH_NB = 5
CHECKPOINT_PERIOD = 10000
BATCH_SIZE = 8
LR = 10**-6
TARGET_UPDATE = 50
MEM_SIZE = 20000
PRIO_EPSILON = 1e-5
ALPHA = 0.4
BETA = 0.9
GAMMA = 0.95
TRAIN_FREQ = 10