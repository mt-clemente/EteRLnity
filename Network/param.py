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
  
PARTIAL_OBSERVABILITY = 1
BATCH_NB = 10
CHECKPOINT_PERIOD = 2000
BATCH_SIZE = 512
LR = 10**-3
TARGET_UPDATE = 1000
MEM_SIZE = 2**16 # Has to be a power of two // use of segment trees
OPT_EPSILON = 1e-6
PRIO_EPSILON = 1e-7
ALPHA = 0.2
BETA = 0.9
GAMMA = 0.975
TRAIN_FREQ = 100
TABU_LENGTH = 0


CONFIG = {
    'swap_range':SWAP_RANGE,
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
    'Tabu length': TABU_LENGTH
}