import os
from einops import repeat, rearrange
import torch
from torch import Tensor
from train_monoagent import get_conflicts, place_tile
from Trajectories import *
from math import exp
from Transformer import DecisionTransformerAC, PPOAgent
from utils import *
from param import *
import wandb

LOG_EVERY = 1

def train_model(hotstart:str = None):

    """
    Your solver for the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    args = parse_arguments()
    pz = EternityPuzzle(args.instance)
    n_tiles = len(pz.piece_list)
    bsize = pz.board_size
    hotstart = args.hotstart
    # torch.cuda.is_available = lambda : False
    

    # -------------------- NETWORK INIT -------------------- 

    if torch.cuda.is_available() and CUDA_ONLY: #FIXME:
        device = 'cuda'
    else:
        device = 'cpu'

    print()
    if n_tiles % HORIZON:
        raise UserWarning(f"Episode length ({n_tiles}) is not a multiple of horizon ({HORIZON})")


    cfg = {
    'n_tiles' : n_tiles ,
    'gamma' : GAMMA ,
    'clip_eps' : CLIP_EPS ,
    'lr' : LR ,
    'epochs' : EPOCHS ,
    'minibatch_size' : MINIBATCH_SIZE,
    'horizon' : HORIZON,
    'seq_len' : SEQ_LEN,
    'state_dim' : bsize+2,
    'n_encoder_layers':N_ENCODER_LAYERS,
    'n_decoder_layers':N_DECODER_LAYERS,
    'n_heads':N_HEADS,
    'dim_embed': DIM_EMBED,
    'hidden_size':HIDDEN_SIZE,
    'entropy_weight':ENTROPY_WEIGHT,
    'value_weight':VALUE_WEIGHT,
    'gae_lambda':GAE_LAMBDA,
    'device' : device 
    }


    agent = PPOAgent(cfg)
    agent.model.train()



    # -------------------- TRAINING LOOP --------------------

    torch.cuda.empty_cache()

    episode = 0
    step = 0

    print("start")

    while 1:


        state, tiles, n_tiles = initialize_sol(args.instance,device)
        state = state.to(dtype=UNIT)
    
        for w in range(NUM_WORKERS):
            rollout(worker=agent.workers[w],
                    state=state,
                    tiles=tiles,
                    n_tiles=n_tiles,
                    step=step,
                    horizon=HORIZON
                    )
            


        
        if torch.cuda.is_available() and not CPU_TRAINING:
            # agent = agent.cuda() #FIXME:
            training_device = 'cuda'
        else:
            training_device = 'cpu'

        state_buf = torch.vstack([worker.ep_buf['state_buf'] for worker in agent.workers]).to(training_device)
        act_buf = torch.hstack([worker.ep_buf['act_buf'] for worker in agent.workers]).to(training_device)
        tile_seq = torch.vstack([worker.ep_buf['tile_seq'] for worker in agent.workers]).to(training_device)
        mask_buf = torch.vstack([worker.ep_buf['mask_buf'] for worker in agent.workers]).to(training_device)
        adv_buf = torch.hstack([worker.ep_buf['adv_buf'] for worker in agent.workers]).to(training_device)
        value_buf = torch.hstack([worker.ep_buf['value_buf'] for worker in agent.workers]).to(training_device)
        rew_buf = torch.hstack([worker.ep_buf['rew_buf'] for worker in agent.workers]).to(training_device)
        policy_buf = torch.vstack([worker.ep_buf['policy_buf'] for worker in agent.workers]).to(training_device)
        rtg_buf = torch.hstack([worker.ep_buf['rtg_buf'] for worker in agent.workers]).to(training_device)
        timestep_buf = torch.hstack([worker.ep_buf['timestep_buf'] for worker in agent.workers]).to(training_device)

        wandb.log({
            'Mean batch reward' : rew_buf.mean(),
            'Advantage repartition': adv_buf,
            'Return to go repartition': rtg_buf,
        })

        dataset = TensorDataset(
            state_buf,
            act_buf,
            tile_seq,
            mask_buf,
            adv_buf,
            policy_buf,
            rtg_buf,
            timestep_buf,
        )

        if (HORIZON) % MINIBATCH_SIZE < MINIBATCH_SIZE / 2 and HORIZON % MINIBATCH_SIZE != 0:
            print("dropping last ",(HORIZON) % MINIBATCH_SIZE)
            drop_last = True
        else:
            drop_last = False


        loader = DataLoader(dataset, batch_size=MINIBATCH_SIZE, shuffle=True, drop_last=drop_last)

        agent.update(
            loader
        )


        if step + HORIZON == n_tiles + 1:
            step = 0
            for worker in agent.workers:
                worker.ep_buf.reset()
            
            print(f"END EPISODE {episode}")
            
        elif step == 0:
            step = HORIZON+1

        elif step < n_tiles - 1:
            step += HORIZON
        else:
            raise OSError(step)

                # checkpoint the policy net
        if episode % CHECKPOINT_PERIOD == 0:
            try:
                torch.save(agent.model.state_dict(), f'models/checkpoint/{episode}.pt')
            except BaseException as e:
                os.mkdir(f"models/checkpoint/")
                torch.save(agent.model.state_dict(), f'models/checkpoint/{episode}.pt')


    return 



def rollout(worker:DecisionTransformerAC,
            state:Tensor,
            tiles:Tensor,
            n_tiles:int,
            step:int,
            horizon:int,
            ):

    device = state.device

    try:

        mask = (torch.zeros_like(tiles[:,0]) == 0)
        horizon_end = False

        # print(f"NEW EPISODE : - {episode:>5}")

        while not horizon_end:
            
            with torch.no_grad():
                policy, value = worker(
                    worker.ep_buf.state_buf[worker.ep_buf.ptr-1],
                    worker.ep_buf.tile_seq[worker.ep_buf.ptr-1],
                    torch.tensor(step,device=device),
                    mask,
                )
                
            selected_tile_idx = worker.get_action(policy)
            selected_tile = tiles[selected_tile_idx]
            new_state, reward, _ = place_tile(state,selected_tile,step)

            worker.ep_buf.push(
                state=state,
                action=selected_tile_idx,
                tile=selected_tile,
                tile_mask=mask,
                policy=policy,
                value=value,
                reward=reward,
                ep_step=step,
                final= (step == (n_tiles-1))
            )
        

            if step == n_tiles-1 or (step) % horizon == 0 and step != 0:
                horizon_end = True
            
            mask[selected_tile_idx] = False
            state = new_state
            step += 1



    except KeyboardInterrupt:
        pass






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
