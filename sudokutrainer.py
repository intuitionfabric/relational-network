import numpy as np
import multiprocessing as mp
import torch.multiprocessing as mp
from sudoku import *
from datagen import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

def RunEpisodesSerial(dataset, n_eps, n_runs, cpuct, temp, neuralnet, no_replacements=False, scoring_scheme=1, maxmoves=100):
    """
    Run episodes with one CPU core. Each episode picks a random sudoku problem from the dataset.

    Args:
    dataset - list of (question, answer) tuples of 81-length strings
    n_eps - number of independent episodes to run
    n_runs - number of MCTS runs per step/turn in an episode
    cpuct&temp - exploration hyperparameters. <1 to exploit
    no_replacements - False if the game allows the player to replace previously filled cells
    scoring_scheme - 1, 2 or 3. see documentation for explanation of the schemes
    maxmoves - maximum moves the player can take before game stops. this is the resignation threshold

    Returns
    list of training examples, each of the form (game state, probability vector, game final score)
    """

    examples = []
    indices = np.arange(len(dataset))

    print("Running episodes in serial.", end="")
    for i in range(n_eps):
        drawn_index = np.random.choice(indices)
        result, _ = PlayEpisode(dataset[int(drawn_index)][0], dataset[int(drawn_index)][1], n_runs, cpuct, temp, neuralnet, no_replacements, scoring_scheme, maxmoves)
        examples += result
        print(".", end="")
    print("Done!")
    return examples

def EvaluateSolverSerial(dataset, n_eps, n_runs, cpuct, temp, neuralnet, no_replacements=False, scoring_scheme=1, maxmoves=100):
    """
    Same as RunEpisodesSerial, but returns instead a list of game final scores. Since you are evaluating the solver,
    low values of cpuct and temp are recommended. 
    """
    scores = []
    indices = np.arange(len(dataset))

    print("Running episodes in serial.", end="")
    for i in range(n_eps):
        drawn_index = np.random.choice(indices)
        _ , gameinstance = PlayEpisode(dataset[int(drawn_index)][0], dataset[int(drawn_index)][1], n_runs, cpuct, temp, neuralnet, no_replacements, scoring_scheme, maxmoves)
        scores.append(gameinstance.getFinalScore())
        print(".", end="")
    print("Done!")
    return scores


def RunEpisodes(dataset, n_eps, n_runs, n_procs, cpuct, temp, neuralnet):
    """
    parallelized over n_procs processes (runs independent episodes on many cores)
    Does not work if neuralnet is a Pytorch module wrapper
    """
    examples = []
    indices = np.arange(len(dataset))
    output = mp.Queue()

    def EpisodePlayerWrapper(startstate, solution):
        startboard = SudokuBoard(startstate, solution)
        result, gameins = PlayEpisode(startboard, n_runs, cpuct, temp, neuralnet)
        output.put(result)

    for i in range(n_eps):
        drawn_indices = np.random.choice(indices, size=n_procs)
        processes = [mp.Process(target=EpisodePlayerWrapper, args=(dataset[int(drawn_indices[j])][0],
                                                                  dataset[int(drawn_indices[j])][1])) for j in range(n_procs)]
        for p in processes:
            p.start()

        while 1:
            time.sleep(0.01*n_runs)
            running = any(p.is_alive() for p in processes)
            while not output.empty():
                examples += output.get()
            if not running:
                break

        for p in processes:
            p.join()

    return examples

def EvaluateSolver(testset, n_eps, n_runs, n_procs, neuralnet, cpuct=0.05, temp=0.05):
    """
    parallelized over n_procs processes (runs independent episodes on many cores)
    Does not work if neuralnet is a Pytorch module wrapper
    """
    scores = []
    indices = np.arange(len(testset))
    output = mp.Queue()

    def EpisodePlayerWrapper(startstate, soln):
        startboard = SudokuBoard(startstate, soln)
        examples, gameins = PlayEpisode(startboard, n_runs, cpuct, temp, neuralnet)
        output.put(examples[-1][2])

    for i in range(n_eps):
        drawn_nums = np.random.choice(indices, size=n_procs)
        processes = [mp.Process(target=EpisodePlayerWrapper, args=(testset[int(drawn_nums[j])][0],
                                                                   testset[int(drawn_nums[j])][1])) for j in range(n_procs)]
        for p in processes:
            p.start()

        while 1:
            time.sleep(0.01*n_runs)
            running = any(p.is_alive() for p in processes)
            while not output.empty():
                scores.append(output.get())
            if not running:
                break

        for p in processes:
            p.join()

    return scores
