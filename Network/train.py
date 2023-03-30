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



def train_model(hotstart:str = None):

    """
    Your solver for the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """

    # -------------------- NETWORK INIT -------------------- 

    args = parse_arguments()
    hotstart = args.hotstart
    # torch.cuda.is_available = lambda : False
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    Transition = namedtuple('Transition',
                                     ('state',
                                      'next_state',
                                      'reward',
                                      'state_val',
                                      'final',
                                      'weights',
                                      'indices'
                                      )
                                      )


    action_nb = comb((MAX_BSIZE-2)**2,2) + comb(4*(MAX_BSIZE-2),2) + comb(4,2) + MAX_BSIZE ** 2 * 3


    if hotstart:
        policy_net = DQN(MAX_BSIZE+2, MAX_BSIZE+2, action_nb, device,COLOR_ENCODING_SIZE)
        policy_net.load_state_dict(torch.load(hotstart))
    else:
        policy_net = DQN(MAX_BSIZE + 2, MAX_BSIZE + 2, action_nb, device, COLOR_ENCODING_SIZE)
    
    policy_net.train()

    target_net = DQN(MAX_BSIZE+2, MAX_BSIZE+2, action_nb, device, COLOR_ENCODING_SIZE).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(),lr = LR,eps=1e-6)

    move_buffer = MoveBuffer()
    
    # -------------------- GAME INIT --------------------

    state, bsize = initialize_sol(args.instance)
    
    state = state.to(dtype=UNIT)

    swp_idx = gen_swp_idx(MAX_BSIZE)
    swp_mask, rot_mask = gen_masks(MAX_BSIZE,swp_idx)
    neighborhood_size = swp_idx.size()[0] + rot_mask.size()[0]
    

    memory = PrioritizedReplayMemory(
        size=MEM_SIZE,
        Transition=Transition,
        alpha=ALPHA,
        batch_size=BATCH_SIZE,
        max_bsize=MAX_BSIZE,
        encoding_size=COLOR_ENCODING_SIZE,
        neighborhood_size=neighborhood_size
    )

    cpu_buf = CPUBuffer(
        capacity=TRAIN_FREQ,
        neighborhood_size=neighborhood_size,
        linked_mem=memory
    )

    tabu = TabuList(TABU_LENGTH)
    stopping_crit = StoppingCriterion(5000)

    obs_neighborhood_size = int(PARTIAL_OBSERVABILITY * neighborhood_size)
    observation_mask = (torch.cat((torch.ones(obs_neighborhood_size),torch.zeros(neighborhood_size - obs_neighborhood_size))) == 1)

    max_entropy = torch.log2(torch.tensor(obs_neighborhood_size))

    # -------------------- TRAINING LOOP --------------------

    torch.cuda.empty_cache()

    step = 0
    pz = EternityPuzzle(args.instance)
    pz.display_solution(to_list(state,bsize),"start.png")

    temp = state.clone()
    print("start")

    best_score = 0
    best_state = state
    episode = 0

    try:
        
        while 1:

            state, bsize = initialize_sol(args.instance)
            state = state.to(dtype=UNIT)
            state = scramlbe(state,rot_mask)

            """
            TODO: IRLS
            """
            move_buffer.state = state
            move_buffer.reward = 0
            move_buffer.state_val = 0
            episode_best_score = 0
            episode_best_state = state
            _, prev_state_score = eval_sol(state,bsize,episode_best_score)
            episode_end = False
            print(f"NEW EPISODE : - {episode:>5}")
            score = 0
            while not episode_end:

                if step % 1 == 0:
                    print(f"{step} - {score}",end='\r')

                tabu.filter(step)

                with torch.no_grad():
                    heur_val = policy_net(state.unsqueeze(0).unsqueeze(0)).squeeze(0)

                best_move_idxs= torch.topk(heur_val,TABU_LENGTH + 1).indices
                new_state = None

                best_neighbors = gen_neighborhood(state,best_move_idxs,swp_idx,swp_mask,rot_mask)

                if TABU_LENGTH != 0:
                    new_state, new_idx = tabu.get_update(best_neighbors,step)
                
                    if new_state is None:
                        tabu.fast_foward()
                        continue
                    
                    tabu.push(new_state,step)

                else:
                    new_state = gen_neighbor(state,best_move_idxs,swp_idx,swp_mask,rot_mask)
                    new_idx = best_move_idxs

                print(new_idx)

                #reward
                new_state_val, score = eval_sol(new_state,bsize,episode_best_score)

                if torch.all(new_state == state):
                    reward = - 2
                    raise OSError
                
                else:
                    reward =  new_state_val + 2 * max(score - prev_state_score,0) + min(score - prev_state_score,0)


                """ if torch.all(new_state == state):
                    pass #raise OSError
                if ((bsize + 1) * bsize * 2 - score) != pz.get_total_n_conflict(to_list(new_state,bsize)):
                    pz.display_solution(to_list(state,bsize),f'old')
                    pz.display_solution(to_list(new_state,bsize),f'new')
                    raise OSError
                """
                
                if score > episode_best_score:
                    print(f"Ep {episode:>7} - score : {score}")
                    episode_best_score = score
                    episode_best_state = state                    

                stopping_crit.update(score,episode_best_score)



                cpu_buf.push(
                    move_buffer.state,
                    state,
                    move_buffer.reward,
                    move_buffer.state_val,
                    final=False
                )


                # end on either max score or trajectory does not lead to anything
                if ((bsize + 1) * bsize * 2 - score) == 0 or stopping_crit.is_stale():
                    if stopping_crit.is_stale():
                        print("STALE")
                    else:
                        print("FOUND SOLUTION!")

                    episode += 1
                    episode_end = True
                    stopping_crit.reset()
                    cpu_buf.push(
                        state,
                        torch.zeros_like(state),
                        reward,
                        heur_val[new_idx],
                        final=True
                    )


                
                prev_state_score = score
                move_buffer.state = state
                move_buffer.reward = reward
                move_buffer.state_val = heur_val[new_idx]


                wandb.log({'Score':score})
                state = new_state

                with torch.no_grad():
                    policy_prob = torch.softmax(heur_val,dim=0).squeeze(-1)
                    policy_entropy = -(policy_prob * torch.log2(policy_prob)).sum()

                wandb.log(
                    {   
                        'Relative policy entropy': policy_entropy/max_entropy,
                        'Q values':heur_val[new_idx],
                        'reward':reward,
                        'Episode':episode
                    }
                )
                # -------------------- MODEL OPTIMIZATION --------------------

                
                if len(memory) >= BATCH_SIZE and step % TRAIN_FREQ == 0:

                    
                    if random.random() > 1 - 10e-5:
                        ok = pz.verify_solution(to_list(state,bsize))

                        if ok == False:
                            
                            print(state)
                            print(to_list(state,bsize))
                            pz.display_solution(to_list(state,bsize))
                            raise ValueError


                    optimize_model(
                        memory,
                        policy_net,
                        target_net,
                        optimizer,
                        device
                    )
                    

                # torch.cuda.empty_cache() 

                step += 1
                #target update
                if step % TARGET_UPDATE == 0:
                    
                    target_net.load_state_dict(policy_net.state_dict())
        
                # checkpoint the policy net
                # if self.num_episode % (self.TARGET_UPDATE * self.TRAIN_FREQ) == 0:
                if step % CHECKPOINT_PERIOD == 0:
                    torch.save(policy_net.state_dict(), f'models/checkpoint/{ENCODING}/{step // CHECKPOINT_PERIOD}.pt')


            if episode_best_score > best_score:
                best_score = episode_best_score
                best_state = episode_best_state

    except KeyboardInterrupt:
        pass

    # reporter = MemReporter()
    # reporter.report()
    print("STILL VALID :",pz.verify_solution(to_list(best_state,bsize)))
    print(best_score)
    return 









def optimize_model(memory:PrioritizedReplayMemory, policy_net:DQN, target_net:DQN, optimizer:torch.optim.Optimizer, device):

    for _ in range(BATCH_NB):

        torch.cuda.empty_cache()

        batch = memory.sample()

        state_batch = batch.state.unsqueeze(1).to(UNIT).to(device)
        next_state_batch = batch.next_state.unsqueeze(1).to(UNIT).to(device)
        reward_batch = batch.reward.to(UNIT).to(device)
        final_mask = batch.final.to(device)
        not_final_mask = torch.logical_not(final_mask)
        old_state_vals = batch.state_val.to(UNIT).to(device)
        weights = torch.from_numpy(batch.weights).to(UNIT).to(device)

        state_values = policy_net(state_batch).max(dim=-1).values

        with torch.no_grad():
            next_state_values = target_net(next_state_batch[not_final_mask]).max(dim=-1).values


        expected_state_values = torch.zeros_like(state_values)

        expected_state_values[not_final_mask] = (next_state_values* GAMMA) + reward_batch[not_final_mask]
        expected_state_values[final_mask] = reward_batch[final_mask]

        criterion = nn.HuberLoss(reduction='none')
        eltwise_loss = criterion(state_values, expected_state_values)

        loss = torch.mean(eltwise_loss * weights)

        optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        for param in policy_net.parameters():
            param.grad.clamp_(-1,1)

        optimizer.step()

        # PER prio update
        prio_loss = eltwise_loss
        new_prio = prio_loss + PRIO_EPSILON
        memory.update_priorities(batch.indices, new_prio.cpu().detach().numpy())

        with torch.no_grad():
            log_prob = torch.log(torch.softmax(state_values,-1))
            old_log_prob = torch.log(torch.softmax(old_state_vals,-1))
        wandb.log(
            {
                'KL div': torch.nn.functional.kl_div(log_prob,old_log_prob,reduction='batchmean',log_target=True),
                'Train mean Q values':state_values.mean(),
                'Train mean next Q values':next_state_values.mean(),
                'Train mean expected Q vlaues':expected_state_values.mean(),
                'Train mean reward':reward_batch.mean(),
                'Train Loss':loss,
            }
        )






def gen_neighbor(state:Tensor, idx:int, swp_idx:Tensor, swp_mask:Tensor, rot_mask:Tensor):
    """
    Generate one neighbor
    """
    if idx >= swp_idx.size()[0]:
        idx = swp_idx.size()[0] + random.randint(0,rot_mask.size()[0]-1)
        neighbor = rotate_elements(state,rot_mask[idx-swp_idx.size()[0]])

    else:
        idx = 0
        neighbor = swap_elements(state,swp_idx[idx],swp_mask[:,idx])
    return neighbor




def gen_neighborhood(state:Tensor, observation_mask:Tensor, swp_idx:Tensor, swp_mask:Tensor, rot_mask:Tensor):
    """
    Generate state's an observed part of the full neighborhood including ranged 2-swaps depending
    on SWAP_RANGE specified in param.py, plus the rotations of each tile.

    Input
    ---------
    state: the current state
    observation_mask: a mask describing the part of the neihborhood that is observed
    swp_idx, swp_mask, rot_mask are fixed masks that represent swapped and rotated elements.

    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    obs_swp_idx = swp_idx[observation_mask[observation_mask < swp_idx.size()[0]]].to(device=device)
    obs_swp_mask = swp_mask[:,observation_mask[observation_mask < swp_idx.size()[0]]].to(device=device)
    obs_rot_mask = rot_mask[observation_mask[observation_mask > swp_idx.size()[0]] - swp_idx.size()[0]].to(device=device)


    if obs_rot_mask.size()[0] == 0:
        return swap_elements(state,obs_swp_idx,obs_swp_mask)

    if obs_swp_idx.size()[0] == 0:
        return rotate_elements(state,obs_rot_mask)
    
    r = rotate_elements(state,obs_rot_mask)
    s = swap_elements(state,obs_swp_idx,obs_swp_mask)
    return torch.vstack((s,r))


def rotate_elements(state:Tensor, rot_mask:Tensor, idx:int=0):
    """
    Generates the states that are one rotation away

    Input
    ----------
    state: the current state
    rot_mask: mask containing ones at every tile one after the other, repeated
    4 times (one for each rotation angle) of shape [board_size**2*4, board_size, board_size, 4*color_encoding_size ]
    """

    # Only one element
    if rot_mask.dim() == 3:

        rolled = state.clone()

        rolled = state.roll(COLOR_ENCODING_SIZE * (idx % 3 + 1),dims=-1)

        rolled = torch.where(rot_mask,rolled,state)

    else:

        state_batch = repeat(state, 'i j c -> k i j c',k=rot_mask.size()[0])
        rolled = torch.zeros((3,rot_mask.size()[0]//3,*state.size()),device=state.device,dtype=state.dtype)

        for dir in range(3):
            rolled[dir,:] = state.roll(COLOR_ENCODING_SIZE * (dir+1),dims=-1)

        rolled = rearrange(rolled,'dir k i j c -> (dir k) i j c')

        rolled = torch.where(rot_mask,rolled,state_batch)

    return rolled
    
def scramlbe(state:torch.Tensor, rot_mask:torch.Tensor, n_rounds:int = 2000):

    for _ in range(n_rounds):
        
        k = random.randint(0,rot_mask.size()[0]-1)
        d = random.randint(1,4)

        state = rotate_elements(state,rot_mask[k],d)

    return state

def swap_elements(state:Tensor, swp_idx:Tensor, swp_mask:Tensor) -> Tensor:
    """
    Generate all given 2-swaps. The swaps are done out of place.

    Input
    ----------
    state_batch : a batch of states of dimension [batch_size, board_size, board_size, 4 * COLOR_ENCODING_SIZE]
    swp_idx : a batch of coordinates of elements to be swapped of dimension [batch_size, 2, 2]

    TODO: Make inplace
    """
    # One swap only
    if swp_idx.dim() == 2:
        swapped = state.clone()
        i = swp_idx[0]
        j = swp_idx[1]
        swapped[[i[0],j[0]],[i[1],j[1]],:] = swapped[[j[0],i[0]],[j[1],i[1]],:]
        

    else:
        expanded_state = repeat(state, 'i j c -> b i j c', b=swp_idx.size()[0])

        i = swp_idx[:,0]
        j = swp_idx[:,1]

        t1 = expanded_state[torch.arange(expanded_state.size()[0]), i[:,0], i[:,1]]
        t2 = expanded_state[torch.arange(expanded_state.size()[0]), j[:,0], j[:,1]]

        swapped = expanded_state.masked_scatter(swp_mask[0], t2) # FIXME: inplace error
        swapped.masked_scatter_(swp_mask[1], t1)
        
    # Return the modified input expanded_state of tensors
    return swapped



def eval_sol(state:Tensor, bsize:int,best_score:int) -> int:
    """
    Evaluates the quality of a solution.
    /!\ This only is the true number of connections if the solution was created
    with side_importance = 1 /!\ 

    Input
    --------
    state: evaluated state of size [max_board_size + 2, max_board_size + 2, 4 * color_encoding_size]
    TODO: Shape the reward to help learning that only the sides are grey
    """

    offset = (MAX_BSIZE + 2 - bsize) // 2
    
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
        return 50, total_connections

    return (total_connections - max_connections) / (max_connections), total_connections




# ----------- MAIN CALL -----------

if __name__ == "__main__"  and '__file__' in globals():

    args = parse_arguments()
    CONFIG['Instance'] = args.instance.replace('instances/','')

    wandb.init(
        project='Eternity II',
        config=CONFIG
    )

    train_model()

    wandb.finish()


