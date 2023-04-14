from math import log2,ceil
import torch

torch.manual_seed(0)

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

DIM_EMBED= 512
HIDDEN_SIZE = 2048
N_LAYERS = 6
N_DECODE_LAYERS = 3
N_HEADS = 4
GAE_LAMBDA = 0.95
ENTROPY_WEIGHT = 0.0001
VALUE_WEIGHT = 0.5
POLICY_WEIGHT = 1

# --------------------  SETTINGS -------------------- 

DEBUG = False

# CUDA can be slower for inference so cpu is used, except for training
# set CUDA_ONLY to True to force cuda.
CUDA_ONLY = True
CPU_TRAINING = False
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
  
EPOCHS = 1
CHECKPOINT_PERIOD = 256*200
MINIBATCH_SIZE = 20
HORIZON = 20# in number of steps
# MEM_SIZE = 200 # in number of episodes
SEQ_LEN = 20
OPT_EPSILON = 1e-4
LR = 1e-3
GAMMA = 0.99
CLIP_EPS = 0.1

CONFIG = {
    'unit':UNIT,
    'Epoch':EPOCHS,
    'Batch size':MINIBATCH_SIZE,
    'Gamma':GAMMA,
    'DIM_EMBED':DIM_EMBED,
    'HIDDEN_SIZE':HIDDEN_SIZE,
    'Encoder layers':N_LAYERS,
    'Decoder layers':N_DECODE_LAYERS,
    'Horizon':HORIZON,
    'Transformer sequence length':SEQ_LEN,
    'N_HEADS':N_HEADS,
    'GAE_LAMBDA':GAE_LAMBDA,
    'ENTROPY_WEIGHT':ENTROPY_WEIGHT,
    'VALUE_WEIGHT':VALUE_WEIGHT,
    'POLICY_WEIGHT':POLICY_WEIGHT,

}