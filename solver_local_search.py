from datetime import datetime, timedelta
from eternity_puzzle import EternityPuzzle
from solver_heuristic import solve_heuristic, choose_piece
import numpy as np

def solve_local_search(eternity_puzzle:EternityPuzzle):
    """
    Local search solution of the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    init_sol, ann_cost= solve_heuristic(eternity_puzzle)
    print(init_sol)
    best_cost = eternity_puzzle.get_total_n_conflict(init_sol)
    board_size = eternity_puzzle.board_size
    init_sol = np.array(init_sol).reshape((board_size,board_size,4))
    best_sol = init_sol
    ann_sol = init_sol

    T_0 = 10
    ALPHA = 0.95
    DURATION = 20
    MAX_TIMEOUT = 1
    BETA = 10
    BETA_VAR = 0.1
    START = datetime.now()
    t = T_0

    step = 0
    beta_adjust = 0
    tabu = []

    while datetime.now() < timedelta(seconds = DURATION) + START and best_cost != 0:

        step += 1
        neighbors = []
        # non edges
        for i in range(board_size):
            for j in range(board_size):
                

                for ip in range(0,board_size):
                    for jp in range(0,board_size):
                        
                        for k in range(1,4):
                            n = ann_sol.copy()
                            n[i,j] = eternity_puzzle.generate_rotation(n[i,j])[k]
                            neighbors.append(n)

                        if (i*board_size +j >= ip * board_size + jp):
                            continue

                        for k in range(4):
                            for kp in range(4):

                                n = ann_sol.copy()
                                n[i,j] = eternity_puzzle.generate_rotation(n[i,j])[k]
                                n[ip,jp] = eternity_puzzle.generate_rotation(n[ip,jp])[kp]

                                n[i,j], n[ip,jp] = n[ip,jp].copy(), n[i,j].copy()
                                neighbors.append(n)

        print(0)
        new_sol = neighbors[np.random.randint(len(neighbors))]

        tabu = [t for t in tabu if t[1] > step]
        if  any([np.array_equal(new_sol,t[0]) for t in tabu]):
            beta_adjust -= BETA_VAR
            continue
        else:
            tabu.append((new_sol,step + MAX_TIMEOUT))


        cost = eternity_puzzle.get_total_n_conflict(arr_to_list(new_sol))
        delta = min(ann_cost - cost,0)


        if np.random.random() < np.exp(max(BETA + beta_adjust,0.5)* delta / t):

            ann_sol = new_sol


            if  cost < best_cost:
                best_sol = new_sol
                best_cost = cost
                print("YAY")
                print(best_cost)
                eternity_puzzle.display_solution(arr_to_list(best_sol),f"{step}")

            beta_adjust = min(BETA, beta_adjust + BETA_VAR)
        
        else:
            beta_adjust -= BETA_VAR

        
        t *= ALPHA

    print(best_sol)
    return arr_to_list(best_sol), best_cost



def arr_to_list(array:np.ndarray):

    arr = array.reshape((array.shape[0] * array.shape[1],array.shape[2]))
    l = [ tuple(arr[i]) for i in range(arr.shape[0])]

    return l