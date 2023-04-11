from math import log2,ceil
import torch

DEBUG = True

# -------------------- PUZZLE SETTINGS -------------------- 

PATH = '/home/wsl/Polymtl/H23/INF6201/Projet/Network'


# global 2-swap gives a size 32640 neighborhood which can be too much
# for the GPU. Capping the swapping range helps reduce the neighborhood
# without losing connectivity.

MAX_BSIZE = 16
PADDED_SIZE = MAX_BSIZE + 2

GRAY = 0
N_COLORS = 23


# -------------------- NETWORK SETTINGS -------------------- 

DIM_EMBED=3*16
HIDDEN_SIZE = 32
N_LAYERS = 2
N_HEADS = 2
GAE_LAMBDA = 0.99
ENTROPY_WEIGHT = 0.00001
VALUE_WEIGHT = .1
POLICY_WEIGHT = 10

# --------------------  SETTINGS -------------------- 


# CUDA can be slower for inference so cpu is used, except for training
# set CUDA_ONLY to True to force cuda.
CUDA_ONLY = False

UNIT = torch.float

ENCODING = 'ordinal'

if ENCODING == 'binary':
    COLOR_ENCODING_SIZE = ceil(log2(N_COLORS))
elif ENCODING == 'ordinal':
    COLOR_ENCODING_SIZE = 1
elif ENCODING == 'one_hot':
    COLOR_ENCODING_SIZE = N_COLORS
else:
    raise ValueError(f"Encoding {ENCODING} not supported")
  
EPOCHS = 15
CHECKPOINT_PERIOD = 256*200
MINIBATCH_SIZE = 64
HORIZON = 4 # in number of episodes
OPT_EPSILON = 1e-8
LR = 1e-4
GAMMA = 0.99
CLIP_EPS = 0.1

CONFIG = {
    'encoding':ENCODING,
    'unit':UNIT,
    'Batch size':MINIBATCH_SIZE,
    'Gamma':GAMMA,
}