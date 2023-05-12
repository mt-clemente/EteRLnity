from math import log2,ceil
import torch

torch.manual_seed(0)

# -------------------- PUZZLE SETTINGS -------------------- 

MAX_BSIZE = 16
PADDED_SIZE = MAX_BSIZE + 2

GRAY = 0
N_COLORS = 23


# -------------------- NETWORK SETTINGS -------------------- 

DIM_EMBED= 20
HIDDEN_SIZE = 512
N_ENCODER_LAYERS = 4
N_DECODER_LAYERS = 4
N_HEADS = 40
GAE_LAMBDA = 0.9
ENTROPY_WEIGHT = 0.03
VALUE_WEIGHT = 0.5
POLICY_WEIGHT = 1

GUIDE_PROB = 0.3

POINTER = True

# --------------------  SETTINGS -------------------- 

DEBUG = False

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
NUM_WORKERS = 8
CHECKPOINT_PERIOD = 100
MINIBATCH_SIZE = 9
# in number of steps, remember that the first tile is always placed
# so for a 100 steps game, your max horizon will be 99. Horizon
# needs to be smaller than an episode.
HORIZON = 99

OPT_EPSILON = 1e-4
LR = 7e-4
GAMMA = 0.94
CLIP_EPS = 0.16

CONFIG = {
    'unit':UNIT,
    'Epoch':EPOCHS,
    'Batch size':MINIBATCH_SIZE,
    'Gamma':GAMMA,
    'DIM_EMBED':DIM_EMBED,
    'HIDDEN_SIZE':HIDDEN_SIZE,
    'Encoder layers':N_ENCODER_LAYERS,
    'Decoder layers':N_DECODER_LAYERS,
    'Horizon':HORIZON,
    'N_HEADS':N_HEADS,
    'GAE_LAMBDA':GAE_LAMBDA,
    'ENTROPY_WEIGHT':ENTROPY_WEIGHT,
    'VALUE_WEIGHT':VALUE_WEIGHT,
    'POLICY_WEIGHT':POLICY_WEIGHT,
    'Pointer':POINTER,
    'Clip epsilon':CLIP_EPS

}