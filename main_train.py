import torch
import wandb
from Network.train import train_model
from Network.utils import parse_arguments
from Network.param import CONFIG

import sys
from pathlib import Path

project_path = Path(__file__).resolve().parent
sys.path.append(str(project_path))


# ----------- MAIN CALL -----------

if __name__ == "__main__"  and '__file__' in globals():

    args = parse_arguments()
    CONFIG['Instance'] = args.instance.replace('instances/','')
    torch.autograd.set_detect_anomaly(True)
    wandb.init(
        project='EteRLnity',
        entity='mateo-clemente',
        group='Tests',
        config=CONFIG
    )

    train_model()

    wandb.finish()
