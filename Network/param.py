from math import log2,ceil
import torch


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

DIM_EMBED= 128 
HIDDEN_SIZE = 256
N_LAYERS = 3
N_HEADS = 4
GAE_LAMBDA = 0.99
ENTROPY_WEIGHT = 0.0001 * 0
VALUE_WEIGHT = 0.05
POLICY_WEIGHT = 2

# --------------------  SETTINGS -------------------- 

DEBUG = True

# CUDA can be slower for inference so cpu is used, except for training
# set CUDA_ONLY to True to force cuda.
CUDA_ONLY = False
CPU_TRAINING = False
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
  
EPOCHS = 5
CHECKPOINT_PERIOD = 256*200
MINIBATCH_SIZE = 128
HORIZON = 4 # in number of steps
MEM_SIZE = 10 # in number of episodes
SEQ_LEN = 4
OPT_EPSILON = 1e-7
LR = 1e-6
GAMMA = 0.99
CLIP_EPS = 0.3

CONFIG = {
    'unit':UNIT,
    'Batch size':MINIBATCH_SIZE,
    'Gamma':GAMMA,
    'DIM_EMBED':DIM_EMBED,
    'HIDDEN_SIZE':HIDDEN_SIZE,
    'N_LAYERS':N_LAYERS,
    'N_HEADS':N_HEADS,
    'GAE_LAMBDA':GAE_LAMBDA,
    'ENTROPY_WEIGHT':ENTROPY_WEIGHT,
    'VALUE_WEIGHT':VALUE_WEIGHT,
    'POLICY_WEIGHT':POLICY_WEIGHT,

}