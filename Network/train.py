from collections import namedtuple

from einops import repeat, rearrange
import torch
from torch import Tensor
from DRQN import *
from math import comb
from pytorch_memlab import MemReporter
from utils import *
from param import *
import wandb

LOG_EVERY = 600

def train_model(hotstart:str = None):

    """
    Your solver for the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    args = parse_arguments()
    pz = EternityPuzzle(args.instance)
    n_tiles = len(pz.piece_list) * 4
    bsize = pz.board_size
    hotstart = args.hotstart
    # torch.cuda.is_available = lambda : False
    
    # -------------------- GAME INIT --------------------

    # -------------------- NETWORK INIT -------------------- 

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    Transition = namedtuple('Transition',
                                     ('state',
                                      'next_state',
                                      'reward',
                                      'action',
                                      'final',
                                      'weights',
                                      'indices'
                                      )
                                      )



    policy_net = DTQN(n_tiles,EMBEDDING_DIM,NUM_LAYERS,NUM_HEADS,DIM_HIDDEN,device=device)
    
    policy_net.train()

    target_net = DTQN(n_tiles,EMBEDDING_DIM,NUM_LAYERS,NUM_HEADS,DIM_HIDDEN,device=device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()


    optimizer = torch.optim.RMSprop(policy_net.parameters(),lr = LR,eps=1e-6)

    move_buffer = MoveBuffer()
    
    ucb_count = torch.zeros(len(pz.piece_list)*4,device=device)

    memory = PrioritizedReplayMemory(
        size=MEM_SIZE,
        Transition=Transition,
        alpha=ALPHA,
        batch_size=BATCH_SIZE,
        encoding_size=COLOR_ENCODING_SIZE,
        n_tiles=n_tiles
    )


    cpu_buf = CPUBuffer(
        capacity=TRAIN_FREQ,
        linked_mem=memory,
        n_tiles=n_tiles
    )

    stopping_crit = StoppingCriterion(10000)

    max_entropy = torch.log2(torch.tensor(MAX_BSIZE**2))
    policy_entropy = max_entropy

    # -------------------- TRAINING LOOP --------------------

    torch.cuda.empty_cache()

    step = 0

    print("start")

    best_score = 0
    episode = 0

    try:
        
        while 1:

            state, remaining_tiles, n_tiles = initialize_sol(args.instance,device)
            state = state.to(dtype=UNIT)
            mask = (torch.zeros_like(remaining_tiles[:,0]) == 0)
            episode_end = False
            prev_conflicts = get_conflicts(state,bsize)

            print(f"NEW EPISODE : - {episode:>5}")
            conflicts = 0
            ep_reward = 0
            ep_good_moves = 0
            episode_best_score = 0
            ep_start = step
            ep_step = 0

            episode_tiles = set()

            while not episode_end:

                if step % 1 == 0:
                    print(f"{step} - {conflicts} - {policy_entropy}",end='\r')


                tile = torch.tensor([ep_step // bsize + 1, ep_step % bsize + 1],device=device)

                with torch.no_grad():
                    state_val = policy_net(state,tile,mask)
                
                explorer = ucb(state_val,ucb_count,step)
                selected_tile_index = torch.argmax(explorer)

                selected_tile = remaining_tiles[selected_tile_index]
                ucb_count[selected_tile_index] += 1

                idx = selected_tile_index % n_tiles
                for dir in range(4):
                    mask[dir * 256 +idx] = False

                new_state = place_tile(state,selected_tile,ep_step)

                conflicts = get_conflicts(new_state,bsize)

                reward = (2 + prev_conflicts - conflicts)
                
                stopping_crit.update(conflicts,episode_best_score)


                if ep_step != 0:
                    cpu_buf.push(
                        move_buffer.state,
                        
                        mask,
                        move_buffer.reward,
                        move_buffer.action,
                        final=False
                    )


                # end on either max conflicts or trajectory does not lead to anything
                if ep_step == 255 or selected_tile_index.item()%256 in episode_tiles:

                    if selected_tile_index.item()%256 in episode_tiles:
                        reward = -10
                    episode += 1
                    episode_end = True
                    stopping_crit.reset()
                    cpu_buf.push(
                        mask,
                        torch.zeros_like(mask) == 1,
                        reward,
                        selected_tile_index,
                        final=True
                    )

                    print(f"END EPISODE {episode} - Conflicts {conflicts}/543")

                    if step % LOG_EVERY == 0:
                        wandb.log({"Actuator episode reward":ep_reward/(step - ep_start + 1e-5),'Mean good moves per ep':ep_good_moves/(step - ep_start + 1e-5)})


                episode_tiles.add(selected_tile_index.item()%256)
                
                prev_conflicts = conflicts
                move_buffer.state = mask
                move_buffer.reward = reward
                move_buffer.action = selected_tile_index


                state = new_state
                with torch.no_grad():
                    policy_prob = torch.softmax(state_val,dim=-1).squeeze(-1)
                    policy_prob = policy_prob[policy_prob != 0]
                    policy_entropy = -(policy_prob * torch.log2(policy_prob)).sum()
                    
                

                if step%LOG_EVERY==0:

                    wandb.log(
                        {   
                            'Score':conflicts,
                            'Relative policy entropy': policy_entropy/max_entropy,
                            'Q':state_val[selected_tile_index:],
                            'Reward': reward,
                            'Episode':episode,
                            'Select min count.':ucb_count.min(),
                        }
                    )
                # -------------------- MODEL OPTIMIZATION --------------------

                
                if len(memory) >= BATCH_SIZE and step % TRAIN_FREQ == 0:

                    

                    optimize(
                        memory,
                        policy_net,
                        target_net,
                        optimizer,
                        device
                    )

                # torch.cuda.empty_cache() 

                step += 1
                ep_step += 1
                #target update
                if step % TARGET_UPDATE == 0:
                    pass
                    # target_net.load_state_dict(meta_net.state_dict())
        
                # checkpoint the policy net
                if step % CHECKPOINT_PERIOD == 0:
                    torch.save(policy_net.state_dict(), f'models/checkpoint/{ENCODING}/{step // CHECKPOINT_PERIOD}.pt')


            if episode_best_score > best_score:
                best_score = episode_best_score

    except KeyboardInterrupt:
        pass

    reporter = MemReporter(policy_net)
    reporter.report()
    print("STILL VALID :",pz.verify_solution(to_list(state,bsize)))
    print(best_score)
    return 






def optimize(memory:PrioritizedReplayMemory, policy_net:DTQN, target_net:DTQN, optimizer:torch.optim.Optimizer, device):

    for _ in range(BATCH_NB):

        # print("-----------------------------OPTI-----------------------------")

        batch = memory.sample()

        mask_batch = batch.state.unsqueeze(1).to(device)
        next_mask_batch = batch.next_state.unsqueeze(1).to(device)
        reward_batch = batch.reward.to(UNIT).to(device)
        final_mask = batch.final.to(device)
        not_final_mask = torch.logical_not(final_mask)
        weights = torch.from_numpy(batch.weights).to(UNIT).to(device)

        action_batch = batch.action.to(device).unsqueeze(-1)
        state_values = policy_net(mask_batch.squeeze(1)).gather(1,action_batch).squeeze(-1)


        expected_state_values = torch.zeros_like(state_values).to(UNIT)

        if torch.any(not_final_mask):
            with torch.no_grad():
                next_state_values = target_net(next_mask_batch[not_final_mask].squeeze(1)).max(dim=-1).values
                expected_state_values[not_final_mask] = (next_state_values * GAMMA) + reward_batch[not_final_mask]



        expected_state_values[final_mask] = reward_batch[final_mask]

        criterion = nn.HuberLoss(reduction='none')
        eltwise_loss = criterion(state_values, expected_state_values)

        loss = torch.mean(eltwise_loss * weights)
        
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        g = 0
        for name,param in policy_net.named_parameters():
            print(f"{name:<58} + {torch.norm(param.grad).item()}")
            g += torch.norm(param.grad)
            param.grad.clamp_(-1,1)

        optimizer.step()

        # PER prio update
        prio_loss = eltwise_loss
        new_prio = prio_loss + PRIO_EPSILON
        memory.update_priorities(batch.indices, new_prio.cpu().detach().numpy())



        wandb.log(
            {
                'Train mean reward':reward_batch.mean(),
                'Train Loss':loss,
                'Meta Gradient cumul norm':g,
            }
        )
 



def get_conflicts(state:Tensor, bsize:int) -> int:
    """
    Evaluates the quality of a solution.
    /!\ This only is the true number of connections if the solution was created
    with side_importance = 1 /!\ 

    Input
    --------
    state: evaluated state of size [max_board_size + 2, max_board_size + 2, 4 * color_encoding_size]
    TODO: Shape the reward to help learning that only the sides are grey
    """

    offset = (PADDED_SIZE - bsize) // 2

    board = state[offset:offset+bsize,offset:offset+bsize]
    
    extended_board = state[offset-1:offset+bsize+1,offset-1:offset+bsize+1]

    n_offset = extended_board[2:,1:-1,2*COLOR_ENCODING_SIZE:3*COLOR_ENCODING_SIZE]
    s_offset = extended_board[:-2,1:-1,:COLOR_ENCODING_SIZE]
    w_offset = extended_board[1:-1,:-2,3*COLOR_ENCODING_SIZE:4*COLOR_ENCODING_SIZE]
    e_offset = extended_board[1:-1,2:,COLOR_ENCODING_SIZE:2*COLOR_ENCODING_SIZE]

    n_connections = board[:,:,:COLOR_ENCODING_SIZE] == n_offset
    s_connections = board[:,:,2*COLOR_ENCODING_SIZE:3*COLOR_ENCODING_SIZE] == s_offset
    w_connections = board[:,:,COLOR_ENCODING_SIZE: 2*COLOR_ENCODING_SIZE] == w_offset
    e_connections = board[:,:,3*COLOR_ENCODING_SIZE: 4*COLOR_ENCODING_SIZE] == e_offset



    redundant_ns = torch.logical_and(n_connections[:-1,:],s_connections[1:,:])
    redundant_we = torch.logical_and(w_connections[:,1:],e_connections[:,:-1])

    redundant_connections = torch.all(redundant_we,dim=-1).sum() + torch.all(redundant_ns,dim=-1).sum()

    all = (torch.all(n_connections,dim=-1).sum() + torch.all(s_connections,dim=-1).sum() + torch.all(e_connections,dim=-1).sum() + torch.all(w_connections,dim=-1).sum())
    
    total_connections = all - redundant_connections

    max_connections = (bsize + 1) * bsize * 2

    if total_connections == max_connections:
        return 10

    return max_connections - total_connections




# ----------- MAIN CALL -----------

if __name__ == "__main__"  and '__file__' in globals():

    args = parse_arguments()
    CONFIG['Instance'] = args.instance.replace('instances/','')

    wandb.init(
        project='Eternity II',
        group='Distributional approach',
        config=CONFIG
    )

    train_model()

    wandb.finish()


