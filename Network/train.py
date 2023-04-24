import copy
import torch
from torch import Tensor
from Network.Trajectories import *
from Network.Transformer import DecisionTransformerAC, PPOAgent
from Network.utils import *
from Network.param import *
import wandb

from solver_advanced import infer_model_size

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
    

    # -------------------- NETWORK INIT -------------------- 

    if torch.cuda.is_available() : #FIXME:
        device = 'cuda'
    else:
        device = 'cpu'

    init_state, tiles, first_corner,n_tiles = initialize_sol(pz,device)

    cfg = {
    'n_tiles' : n_tiles ,
    'gamma' : GAMMA ,
    'clip_eps' : CLIP_EPS ,
    'lr' : LR ,
    'epochs' : EPOCHS ,
    'minibatch_size' : MINIBATCH_SIZE,
    'horizon' : HORIZON,
    'state_dim' : bsize+2,
    'n_encoder_layers':N_ENCODER_LAYERS,
    'n_decoder_layers':N_DECODER_LAYERS,
    'n_heads':N_HEADS,
    'dim_embed': DIM_EMBED,
    'hidden_size':HIDDEN_SIZE,
    'entropy_weight':ENTROPY_WEIGHT,
    'value_weight':VALUE_WEIGHT,
    'gae_lambda':GAE_LAMBDA,
    'device' : device,
    'first_corner':first_corner
    }

    load_list = [
        'actor_dt',
        'critic_dt',
        'embed_tiles',
        'embed_state',
    ]

    cfg['n_encoder_layers'], cfg['n_decoder_layers'], cfg['hidden_size'] = infer_model_size("models/Inference/D")

    cfg['first_corner'] = first_corner
    cfg['n_tiles'] = n_tiles
    cfg['state_dim'] = bsize + 2

    load_list = [
        'actor_dt',
        'critic_dt',
        'embed_tiles',
        'embed_state',
    ]

    agent = PPOAgent(
        config=cfg,
        eval_model_dir="models/Inference/D",
        tiles=tiles,
        init_state=init_state,
        load_list=load_list,
        device=device
    ).model


    agent = PPOAgent(cfg,tiles,init_state)
    agent.model.train()



    # -------------------- TRAINING LOOP --------------------

    torch.cuda.empty_cache()
    dir = f'models/checkpoint/{datetime.now().strftime("%d%M_%h_%m")}'
    episode = 0
    step = 0
    best_conf = 1000
    best_sol = None
    states = init_state.repeat(NUM_WORKERS,1,1,1)
    masks = (torch.zeros_like(tiles[:,0]) == 0).repeat(NUM_WORKERS,1)

    print("start")

    try:
        while 1:

            avg_conf = 0
            for w in range(NUM_WORKERS):
                new_state, new_mask = rollout(worker=agent.workers[w],
                                state=states[w],
                                mask=masks[w],
                                tiles=tiles,
                                n_tiles=n_tiles,
                                step=step,
                                horizon=HORIZON
                                )
                states[w] = new_state
                masks[w] = new_mask
                conf = get_conflicts(new_state,bsize=agent.workers[w].bsize)
                avg_conf += conf

                if best_conf > conf:
                    best_conf = conf
                    if step +  HORIZON >= n_tiles or HORIZON > n_tiles / 2:
                        best_sol = new_state
                        pz.print_solution(to_list(best_sol,bsize),f"B_{episode}.txt")


            avg_conf = avg_conf / NUM_WORKERS
            
            if torch.cuda.is_available() and not CPU_TRAINING:
                training_device = 'cuda'
            else:
                training_device = 'cpu'

            state_buf = torch.vstack([worker.ep_buf['state_buf'] for worker in agent.workers]).to(training_device)
            act_buf = torch.hstack([worker.ep_buf['act_buf'] for worker in agent.workers]).to(training_device)
            mask_buf = torch.vstack([worker.ep_buf['mask_buf'] for worker in agent.workers]).to(training_device)
            adv_buf = torch.hstack([worker.ep_buf['adv_buf'] for worker in agent.workers]).to(training_device)
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
                mask_buf,
                adv_buf,
                policy_buf,
                rtg_buf,
                timestep_buf,
            )

            # if MINIBATCH_SIZE%HORIZON < MINIBATCH_SIZE / 2 and HORIZON % MINIBATCH_SIZE != 0:
            #     print("dropping last ",(HORIZON) % MINIBATCH_SIZE)
            #     drop_last = True
            # else:
            #     drop_last = False


            loader = DataLoader(dataset, batch_size=MINIBATCH_SIZE, shuffle=True, drop_last=True)
            agent.update(
                loader
            )

            if step +  HORIZON >= n_tiles or HORIZON > n_tiles / 2:
                step = 0
                for worker in agent.workers:
                    worker.ep_buf.reset()

                print(f"END EPISODE {episode} - {avg_conf}")

                states = init_state.repeat(NUM_WORKERS,1,1,1)
                masks = (torch.zeros_like(tiles[:,0]) == 0).repeat(NUM_WORKERS,1)
                wandb.log({'Conflicts':avg_conf})

                episode += 1
                
            elif step == 0:
                step = HORIZON+1

            elif step < n_tiles - 2:
                step += HORIZON
            else:
                raise OSError(step)

                    # checkpoint the policy net
            if episode % CHECKPOINT_PERIOD == 0:

                agent.save_models(dir,episode)

    except KeyboardInterrupt:
        pass

    list_sol = to_list(best_sol,bsize)
    print(pz.verify_solution(list_sol))
    pz.print_solution(list_sol,"run_solution")


def rollout(worker:DecisionTransformerAC,
            state:Tensor,
            mask:Tensor,
            tiles:Tensor,
            n_tiles:int,
            step:int,
            horizon:int,
            ):

    bsize = int(n_tiles**0.5)
    device = state.device

    horizon_end = False

    while not horizon_end:

        with torch.no_grad():
            policy, value = worker(
                state,
                torch.tensor(step,device=device),
                mask,
            )
            
        selected_tile_idx = worker.get_action(policy)
        selected_tile = tiles[selected_tile_idx]
        new_state, reward, _ = place_tile(state,selected_tile,step,step_offset=1)

        # if step == n_tiles - 2:
        #      reward = get_connections(new_state,bsize,step) * 0.5
            

        worker.ep_buf.push(
            state=state,
            action=selected_tile_idx,
            tile=selected_tile,
            tile_mask=mask,
            policy=policy,
            value=value,
            reward=reward,
            ep_step=step,
            final= (step == (n_tiles-2))
        )

    
        if step == n_tiles-2 or (step) % horizon == 0 and step != 0:
            horizon_end = True
        
        mask[selected_tile_idx] = False
        state = new_state
        step += 1

    return new_state,mask

