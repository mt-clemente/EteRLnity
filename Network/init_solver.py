import copy
import numpy as np
from eternity import EternityPuzzle

GRAY = 0
BLACK = 23
RED = 24
WHITE = 25

NORTH = 0
SOUTH = 1
WEST = 2
EAST = 3

def init_boost_conflicts(eternity_puzzle:EternityPuzzle):
    """
    Heuristic solution of the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """

    solution = [(23,23,23,23)] * eternity_puzzle.board_size**2
    remaining_piece = copy.deepcopy(eternity_puzzle.piece_list)

    # shuffle the pieces
    # take the orientation that works the best
    for i in range(eternity_puzzle.n_piece):

        piece = choose_piece(i//eternity_puzzle.board_size,i%eternity_puzzle.board_size,solution,remaining_piece,eternity_puzzle)

        best_n_conflicts = 2**32
        new_piece = None

        for k in range(0,4):

            piece_permuted = eternity_puzzle.generate_rotation(piece)[k]
            temp = solution.copy()
            temp[i] = piece_permuted
            cost = eternity_puzzle.get_total_n_conflict(temp)
            if  cost < best_n_conflicts:
                new_piece = piece_permuted
                best_n_conflicts = cost


        solution[i] = new_piece

        remaining_piece.remove(piece)

    return solution, eternity_puzzle.get_total_n_conflict(solution)


def choose_piece(i,j,solution,pieces,puzzle:EternityPuzzle):

    board_size = puzzle.board_size

    if len(pieces) == 1:
        return pieces[0]

    k = board_size * i + j
    k_east = board_size * i + (j + 1)
    k_west = board_size * i + (j - 1)
    k_south = board_size * (i + 1) + j
    k_north = board_size * (i - 1) + j

    if i == 0 or j == 0 or i == board_size -1 or j == board_size -1:
        with_gray = [p for p in pieces if GRAY in p]
        return with_gray[0]

    else:
        pieces = [p for p in pieces if GRAY not in p]


    scores = []
    for p in pieces:


        if i == 0 :
            c_north = GRAY
        
        else:
            c_north = solution[k_north][SOUTH]
            
        if  j == 0:
            c_west = GRAY
        
        else:
            c_west = solution[k_west][EAST]

        if i == board_size-1:
            c_south = GRAY
        
        else:
            c_south = solution[k_south][NORTH]

        if j == board_size-1:
            c_east = GRAY
        
        else:
            c_east = solution[k_east][WEST]

        adjacent_colors = set((c_south,c_north,c_east,c_west))

        scores.append(len(adjacent_colors.intersection(set(p))))

    worst = np.argmin(scores)
    if isinstance(worst,np.ndarray):
        p = pieces[worst[0]]
    else:
        p = pieces[worst]

    return p

