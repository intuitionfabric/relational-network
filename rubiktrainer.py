import numpy as np
from rubik import *

def RunEpisodes(dataset, n_eps, n_runs, cpuct, temp, maxmoves, neuralnet):
    """
    Run episodes with one CPU core. Each episode picks a random rubiks cube starting state from the dataset.

    Args:
    dataset - list of 54-length strings representing rubik's cube states
    n_eps - number of independent episodes to run
    n_runs - number of MCTS runs per step/turn in an episode
    cpuct&temp - exploration hyperparameters. <1 to exploit
    maxmoves - maximum moves the player can take before game stops. this is the resignation threshold

    Returns
    list of training examples, each of the form (game state, probability vector, game final score)
    """

    examples = []
    l = len(dataset)
    assert(l>=n_eps)
    indices = np.arange(l)

    print("Running episodes in serial.", end="")
    for i in range(n_eps):
        drawn_index = np.random.choice(indices, replace=False)
        result, _ = PlayEpisode(dataset[int(drawn_index)], n_runs, cpuct, temp, neuralnet, maxmoves)
        examples += [(matrix,p,v) for cube,p,v in result for matrix in cube.generate_symmetries() ]
        print(".", end="")
    print("Done!")
    return examples

def EvaluateSolver(dataset, n_eps, n_runs, cpuct, temp, maxmoves, neuralnet):
    """
    Same as RunEpisodes, but returns instead a list of game final scores and the corresponding number of moves taken per game played.
    Since you are evaluating the solver, low values of cpuct and temp are recommended.
    """
    scores = []
    num_moves = []
    l = len(dataset)
    assert(l>=n_eps)
    indices = np.arange(l)

    print("Running episodes in serial.", end="")
    for i in range(n_eps):
        drawn_index = np.random.choice(indices, replace=False)
        _ , gameinstance = PlayEpisode(dataset[int(drawn_index)], n_runs, cpuct, temp, neuralnet, maxmoves)
        scores.append(gameinstance.get_score())
        num_moves.append(gameinstance.num_moves_taken())
        print(".", end="")
    print("Done!")
    return scores, num_moves

