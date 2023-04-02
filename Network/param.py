from math import log2,ceil
import torch

# -------------------- PUZZLE SETTINGS -------------------- 

PATH = '/home/wsl/Polymtl/H23/INF6201/Projet/Network'


# global 2-swap gives a size 32640 neighborhood which can be too much
# for the GPU. Capping the swapping range helps reduce the neighborhood
# without losing connectivity.
SWAP_RANGE = 40

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


# -------------------- NETWORK SETTINGS -------------------- 

DIM_CONV3D = 256
KER3D = 2
DIM_CONV2D1 = 64
KER2D1 = 2
DIM_CONV2D2 = 32
KER2D2 = 3
DIM_LIN = 32

# -------------------- TRAINING SETTINGS -------------------- 
UNIT = torch.float
SWAP_RANGE = 2

ENCODING = 'binary'

if ENCODING == 'binary':
    COLOR_ENCODING_SIZE = ceil(log2(N_COLORS))
elif ENCODING == 'ordinal':
    COLOR_ENCODING_SIZE = 1
elif ENCODING == 'one_hot':
    COLOR_ENCODING_SIZE = N_COLORS
else:
    raise ValueError(f"Encoding {ENCODING} not supported")
  
PARTIAL_OBSERVABILITY = 1
BATCH_NB = 20
CHECKPOINT_PERIOD = 100000
BATCH_SIZE = 128
LR = 5e-4
TARGET_UPDATE = 10000
MEM_SIZE = 2**17 # Has to be a power of two // use of segment trees
OPT_EPSILON = 1e-6
PRIO_EPSILON = 1e-7
ALPHA = 0.4
BETA = 0.9
GAMMA = 0.975
TRAIN_FREQ = 800
TABU_LENGTH = 0
MAX_REWARD = 10
MIN_REWARD = -5
ATOMS = 10
INIT_TEMP = 1000
TEMP_FACT = 1 - 5e-6
MIN_TEMP = 1
MAX_TEMP = 1000

CONFIG = {
    'Initial temperature':INIT_TEMP,
    'encoding':ENCODING,
    'unit':UNIT,
    'partial_observability':PARTIAL_OBSERVABILITY,
    'Batch size':BATCH_SIZE,
    'Learning rate':LR,
    'Target update':TARGET_UPDATE,
    'Replay size':MEM_SIZE,
    'Prio epsilon':PRIO_EPSILON,
    'Alpha':ALPHA,
    'Beta':BETA,
    'Gamma':GAMMA,
    'Train frequence':TRAIN_FREQ,
    'Dim conv3d':DIM_CONV3D,
    'Size ker3d':KER3D,
    'Dim conv2d1':DIM_CONV2D1,
    'Size ker2d1':KER2D1,
    'Dim conv2d2':DIM_CONV2D2,
    'Size ker2d2':KER2D2,
}