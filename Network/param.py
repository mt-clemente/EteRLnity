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

EMBEDDING_DIM = 16
NUM_LAYERS = 4
NUM_HEADS = 2
DIM_HIDDEN = 128

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
  
BATCH_NB = 20
CHECKPOINT_PERIOD = 10000
BATCH_SIZE = 64
TARGET_UPDATE = 4000
MEM_SIZE = 2**17 # Has to be a power of two // use of segment trees
OPT_EPSILON = 1e-6
PRIO_EPSILON = 1e-7
LR = 1e-6
ALPHA = 0.4
BETA = 0.9
GAMMA = 0.92
TRAIN_FREQ = 800

CONFIG = {
    'encoding':ENCODING,
    'unit':UNIT,
    'Batch size':BATCH_SIZE,
    'Target update':TARGET_UPDATE,
    'Replay size':MEM_SIZE,
    'Prio epsilon':PRIO_EPSILON,
    'Alpha':ALPHA,
    'Beta':BETA,
    'Gamma':GAMMA,
    'Train frequence':TRAIN_FREQ,
}