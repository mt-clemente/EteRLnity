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
    hotstart = args.hotstart
    torch.cuda.is_available = lambda : False
    
    # -------------------- GAME INIT --------------------

    state, bsize = initialize_sol(args.instance)
    
    state = state.to(dtype=UNIT)

    swp_idx = gen_swp_idx(2*SWAP_RANGE+1)
    swp_mask, rot_mask = gen_masks(2*SWAP_RANGE+1,swp_idx)
    neighborhood_size = swp_idx.size()[0] + rot_mask.size()[0]

    print(neighborhood_size)


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



    if hotstart:
        meta_net = MetaDQN(PADDED_SIZE, PADDED_SIZE, neighborhood_size, device,COLOR_ENCODING_SIZE)
        meta_net.load_state_dict(torch.load(hotstart))
        actuator = Actuator(SWAP_RANGE+1,(2 * SWAP_RANGE + 1)**2 - 1 + 3, device,COLOR_ENCODING_SIZE)
        actuator.load_state_dict(torch.load(hotstart))
    else:
        meta_net = MetaDQN(PADDED_SIZE, PADDED_SIZE, neighborhood_size, device, COLOR_ENCODING_SIZE)
        actuator = Actuator(SWAP_RANGE,(2 * SWAP_RANGE + 1)**2 - 1 + 3, device,COLOR_ENCODING_SIZE)
    
    meta_net.train()
    actuator.train()

    meta_target = MetaDQN(PADDED_SIZE, PADDED_SIZE, neighborhood_size, device, COLOR_ENCODING_SIZE)
    meta_target.load_state_dict(meta_net.state_dict())
    meta_target.eval()
    
    actuator_target = Actuator(SWAP_RANGE,(2 * SWAP_RANGE + 1)**2 - 1 + 3, device,COLOR_ENCODING_SIZE)
    actuator_target.load_state_dict(actuator.state_dict())
    actuator_target.eval()


    meta_optimizer = torch.optim.Adam(meta_net.parameters(),lr = META_LR,eps=1e-6)
    actuator_optimizer = torch.optim.Adam(actuator.parameters(),lr = ACT_LR,eps=1e-6)

    move_buffer = MoveBuffer()
    
    meta_ucb_count = torch.zeros(MAX_BSIZE,MAX_BSIZE,device=device)
    act_ucb_count = torch.zeros((2*SWAP_RANGE+1)**2-1+3,device=device)

    meta_memory = PrioritizedReplayMemory(
        size=MEM_SIZE,
        Transition=Transition,
        alpha=ALPHA,
        batch_size=BATCH_SIZE,
        encoding_size=COLOR_ENCODING_SIZE,
        meta = True
    )

    meta_cpu_buf = CPUBuffer(
        capacity=TRAIN_FREQ,
        linked_mem=meta_memory,
        meta=True
    )

    actuator_memory = PrioritizedReplayMemory(
        size=MEM_SIZE,
        Transition=Transition,
        alpha=ALPHA,
        batch_size=BATCH_SIZE,
        encoding_size=COLOR_ENCODING_SIZE,

    )

    actuator_cpu_buf = CPUBuffer(
        capacity=TRAIN_FREQ,
        linked_mem=actuator_memory
    )

    stopping_crit = StoppingCriterion(10000)

    meta_max_entropy = torch.log2(torch.tensor(MAX_BSIZE**2))
    act_max_entropy = torch.log2(torch.tensor((2*SWAP_RANGE+1)**2-1+3))
    policy_entropy = meta_max_entropy

    # -------------------- TRAINING LOOP --------------------

    torch.cuda.empty_cache()

    step = 0
    pz = EternityPuzzle(args.instance)
    pz.display_solution(to_list(state,bsize),"start.png")

    print("start")

    best_score = 0
    best_state = state
    episode = 0

    try:
        
        while 1:

            state, bsize = initialize_sol(args.instance)
            state = state.to(dtype=UNIT)
            state = scramlbe(state,rot_mask)
            episode_end = False
            _, prev_state_score = eval_sol(state,bsize)

            print(f"NEW EPISODE : - {episode:>5}")
            score = 0
            ep_reward = 0
            ep_good_moves = 0
            episode_best_score = 0
            ep_start = step

            while not episode_end:

                if step % 10 == 0:
                    print(f"{step} - {score} - {policy_entropy}",end='\r')


                with torch.no_grad():
                    meta_val = meta_net(state.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                
                explorer = ucb(meta_val,meta_ucb_count,step,True)
                selected_tile = (explorer == torch.max(explorer)).nonzero()

                selected_tile = selected_tile[random.randint(0,selected_tile.size()[0]-1)] + (SWAP_RANGE)
                selected_tile = torch.tensor([15,15])
                if torch.any((selected_tile+SWAP_RANGE) >= PADDED_SIZE):
                    raise OSError
                meta_ucb_count[selected_tile - SWAP_RANGE] += 1


                with torch.no_grad():
                    action_val = actuator(state.unsqueeze(0).unsqueeze(0),selected_tile.unsqueeze(0)).squeeze(0)

                action_explorer = ucb(action_val,act_ucb_count,step)
                action = torch.argmax(action_explorer)
                act_ucb_count[action] += 1
                new_state = gen_neighbor(state,selected_tile,action)
                new_state_val, score = eval_sol(new_state,bsize)

                if torch.all(new_state == state):
                    a_reward = - 3
                
                else:
                    a_reward =  (4 * max(score - prev_state_score,0) +0.25 * min(score - prev_state_score,0)) / 4

                ep_reward += a_reward


                
                if score > episode_best_score:
                    print(f"Ep {episode:>7} - score : {score}")
                    m_reward = 2
                    episode_best_score = score
                    episode_best_state = state                    
                    ep_good_moves += 1

                elif score - prev_state_score > 0:
                    m_reward = 0.5 * score / episode_best_score
                    ep_good_moves += 1
                else:
                    m_reward = 0

                stopping_crit.update(score,episode_best_score)


                if step - ep_start != 0:

                    meta_cpu_buf.push(
                        move_buffer.state,
                        state,
                        move_buffer.m_reward,
                        move_buffer.tile,
                        final=False
                    )

                    
                    actuator_cpu_buf.push(
                        move_buffer.state,
                        state,
                        move_buffer.a_reward,
                        torch.hstack((move_buffer.action,move_buffer.tile)),
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
                    meta_cpu_buf.push(
                        state,
                        torch.zeros_like(state),
                        m_reward,
                        selected_tile,
                        final=True
                    )
                    actuator_cpu_buf.push(
                        state,
                        torch.zeros_like(state),
                        a_reward,
                        torch.hstack((action,selected_tile)),
                        final=True
                    )
                    if step % LOG_EVERY == 0:
                        wandb.log({"Actuator episode reward":ep_reward/(step - ep_start + 1e-5),'Mean good moves per ep':ep_good_moves/(step - ep_start + 1e-5)})


                
                prev_state_score = score
                move_buffer.state = state
                move_buffer.m_reward = m_reward
                move_buffer.a_reward = a_reward
                move_buffer.action = action
                move_buffer.tile = selected_tile


                state = new_state
                with torch.no_grad():
                    policy_prob = torch.softmax(meta_val,dim=-1).squeeze(-1)
                    policy_prob = policy_prob[policy_prob != 0]
                    policy_entropy = -(policy_prob * torch.log2(policy_prob)).sum()
                    
                    act_policy_prob = torch.softmax(action_val,dim=-1).squeeze(-1)
                    act_policy_prob = act_policy_prob[act_policy_prob != 0]
                    act_policy_entropy = -(act_policy_prob * torch.log2(act_policy_prob)).sum()

                
                log_tile = selected_tile-SWAP_RANGE

                if step%LOG_EVERY==0:

                    wandb.log(
                        {   
                            'Score':score,
                            'A Relative policy entropy': act_policy_entropy/act_max_entropy,
                            'M Relative policy entropy': policy_entropy/meta_max_entropy,
                            'Meta Q':meta_val[log_tile[0],log_tile[1]],
                            'Actuator Q':action_val[action],
                            'Actuator Reward':a_reward,
                            'Meta Reward': m_reward,
                            'Episode':episode,
                            'Select min count.':meta_ucb_count.min(),
                            'Action min count.':act_ucb_count.min(),
                        }
                    )
                # -------------------- MODEL OPTIMIZATION --------------------

                
                if len(meta_memory) >= BATCH_SIZE and step % TRAIN_FREQ == 0:

                    
                    if random.random() > 1 - 10e-5:
                        ok = pz.verify_solution(to_list(state,bsize))

                        if ok == False:
                            
                            print(state)
                            print(to_list(state,bsize))
                            pz.display_solution(to_list(state,bsize))
                            raise ValueError


                    optimize_meta(
                        meta_memory,
                        meta_net,
                        meta_target,
                        meta_optimizer,
                        device
                    )

                    optimize_actuator(
                        actuator_memory,
                        actuator,
                        actuator_target,
                        meta_net,
                        actuator_optimizer,
                    )
                    

                # torch.cuda.empty_cache() 

                step += 1
                #target update
                if step % TARGET_UPDATE == 0:
                    pass
                    # target_net.load_state_dict(meta_net.state_dict())
        
                # checkpoint the policy net
                # if self.num_episode % (self.TARGET_UPDATE * self.TRAIN_FREQ) == 0:
                if step % CHECKPOINT_PERIOD == 0:
                    torch.save(meta_net.state_dict(), f'models/checkpoint/{ENCODING}/{step // CHECKPOINT_PERIOD}.pt')


            if episode_best_score > best_score:
                best_score = episode_best_score
                best_state = episode_best_state

    except KeyboardInterrupt:
        pass

    # reporter = MemReporter(meta_net)
    # reporter.report()
    print("STILL VALID :",pz.verify_solution(to_list(state,bsize)))
    print(best_score)
    return 









def optimize_actuator(memory:PrioritizedReplayMemory, policy_net:Actuator, target_net:Actuator, meta_net:MetaDQN,optimizer:torch.optim.Optimizer):

    device = 'cuda'
    policy_net = policy_net.to(device)
    target_net = target_net.to(device)
    meta_net = meta_net.to(device)

    for _ in range(BATCH_NB):

        # print("-----------------------------OPTI-----------------------------")
        torch.cuda.empty_cache()

        batch = memory.sample()

        state_batch = batch.state.unsqueeze(1).to(UNIT).to(device)
        next_state_batch = batch.next_state.unsqueeze(1).to(UNIT).to(device)
        reward_batch = batch.reward.to(UNIT).to(device)
        final_mask = batch.final.to(device)
        not_final_mask = torch.logical_not(final_mask)
        weights = torch.from_numpy(batch.weights).to(UNIT).to(device)

        action_batch = batch.action.to(device)
        
        if action_batch.size()[-1] == 3:
            tile_batch = action_batch[:,[1,2]]
            action_batch = action_batch[:,[0]]


        state_values = policy_net(state_batch,tile_batch).gather(1,action_batch)

        with torch.no_grad():
            next_meta_val = meta_net(next_state_batch[not_final_mask])
            next_tile_batch = (next_meta_val == torch.amax(next_meta_val,(2,3)).unsqueeze(-1).unsqueeze(-1)).nonzero()[:,2:4]#FIXME: Q val equality
            next_state_values = target_net(next_state_batch[not_final_mask],next_tile_batch[not_final_mask]).max(dim=-1).values


        expected_state_values = torch.zeros_like(state_values).squeeze(-1)
        expected_state_values[not_final_mask] = (next_state_values * GAMMA) + reward_batch[not_final_mask]
        expected_state_values[final_mask] = reward_batch[final_mask]

        criterion = nn.HuberLoss(reduction='none')
        eltwise_loss = criterion(state_values.squeeze(1), expected_state_values)

        loss = torch.mean(eltwise_loss * weights)
        
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        g = 0
        for name,param in policy_net.named_parameters():
            g += torch.norm(param.grad)
            # print(f"{name:<28} - {torch.norm(param.grad).item()}")
            param.grad.clamp_(-1,1)

        optimizer.step()

        # PER prio update
        prio_loss = eltwise_loss
        new_prio = prio_loss + PRIO_EPSILON
        memory.update_priorities(batch.indices, new_prio.cpu().detach().numpy())

        """ with torch.no_grad():
            log_prob = torch.log(torch.softmax(state_values,-1)+1e-5)
            log_prob[log_prob == -torch.inf] = 0
            old_log_prob = torch.log(torch.softmax(old_state_vals,-1)+1e-5)
            old_log_prob[old_log_prob == -torch.inf] = 0

        wandb.log(
            {
                'KL div': torch.nn.functional.kl_div(log_prob,old_log_prob,reduction='batchmean',log_target=True),
                'Train mean Q values':state_values.mean(),
                'Train mean next Q values':next_state_values.mean(),
                'Train mean expected Q vlaues':expected_state_values.mean(),
                'Train mean reward':reward_batch.mean(),
                'Train Loss':loss,
            }
        ) """

        wandb.log(
            {
                'Train mean reward':reward_batch.mean(),
                'Train Loss':loss,
                'Gradient cumul norm':g,
            }
        )

    policy_net = policy_net.cpu()
    target_net = target_net.cpu()
    meta_net = meta_net.cpu()

def optimize_meta(memory:PrioritizedReplayMemory, policy_net:MetaDQN, target_net:MetaDQN, optimizer:torch.optim.Optimizer, device):

    device = 'cuda'
    policy_net = policy_net.to(device)
    target_net = target_net.to(device)
    
    for _ in range(BATCH_NB):

        # print("-----------------------------OPTI-----------------------------")
        torch.cuda.empty_cache()

        batch = memory.sample()

        state_batch = batch.state.unsqueeze(1).to(UNIT).to(device)
        next_state_batch = batch.next_state.unsqueeze(1).to(UNIT).to(device)
        reward_batch = batch.reward.to(UNIT).to(device)
        final_mask = batch.final.to(device)
        not_final_mask = torch.logical_not(final_mask)
        weights = torch.from_numpy(batch.weights).to(UNIT).to(device)

        tile_batch = batch.action.to(device) - SWAP_RANGE
        
        state_values = policy_net(state_batch).squeeze(1)[torch.arange(state_batch.size()[0]),tile_batch[:,0],tile_batch[:,1]]

        with torch.no_grad():
            next_state_values = torch.amax(target_net(next_state_batch[not_final_mask]),dim=(2,3)).squeeze(-1)




        expected_state_values = torch.zeros_like(state_values)
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
            # print(f"{name:<28} + {torch.norm(param.grad).item()}")
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
 
    policy_net = policy_net.cpu()
    target_net = target_net.cpu()




def gen_neighbor(state:Tensor, tile:Tensor, action:int):
    """
    Generate one neighbor
    """

    if action < 3:
        neighbor = rotate(state,tile,action.item() + 1)

    else:
        target_tile = torch.tensor([(action-3)//(2*SWAP_RANGE+1),(action-3)%(2*SWAP_RANGE+1)],device=state.device) + tile - SWAP_RANGE
        if torch.all(target_tile == tile):
            target_tile = tile + SWAP_RANGE
        
        if state[tile[0],tile[1]].count_nonzero() == 0 or state[target_tile[0],target_tile[1]].count_nonzero() == 0:
            return state

        neighbor = swap(state,tile,target_tile)

    return neighbor



def rotate(state:Tensor, tile:Tensor, dir:int=0):
    rolled = state.clone()
    rolled[tile] = rolled[tile].roll(dir,dims=-1)
    return rolled

def swap(state:Tensor, tile:Tensor, target_tile:Tensor):
    swapped = state.clone()
    temp = swapped[target_tile[0],target_tile[1]].clone()
    swapped[target_tile[0],target_tile[1]] = swapped[tile[0],tile[1]]
    swapped[tile[0],tile[1]] = temp
    return swapped


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
        
        state = rotate(state,torch.randint(0+SWAP_RANGE,MAX_BSIZE+SWAP_RANGE,(2,)),random.randint(0,2))

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



def eval_sol(state:Tensor, bsize:int) -> int:
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
        return 10, total_connections

    return 0.4*(total_connections - max_connections) / (max_connections), total_connections




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


