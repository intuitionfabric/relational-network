import numpy as np
#import multiprocessing as mp
import torch.multiprocessing as mp
from sudoku import SudokuBoard, SudokuGame, MCTSnode, MCTSrun, PlayEpisode

def randomprobs(matrix):
    return np.random.random_sample((9,9,9))

def fakeneuralnetwork(matrix):
    value = np.random.random_sample()
    return (randomprobs(matrix), value)

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

def RunEpisodesSerial(dataset, n_eps, n_runs, cpuct, temp, neuralnet):
    examples = []
    indices = np.arange(len(dataset))

    print("Running episodes in serial.", end="")
    for i in range(n_eps):
        drawn_index = np.random.choice(indices)
        startboard = SudokuBoard(dataset[int(drawn_index)][0], dataset[int(drawn_index)][1])
        result, _ = PlayEpisode(startboard, n_runs, cpuct, temp, neuralnet)
        examples += result
        print(".", end="")
    print("Done!")
    return examples

def EvaluateSolverSerial(testset, n_eps, n_runs, neuralnet, cpuct=0.1, temp=0.2):
    scores = []
    indices = np.arange(len(testset))

    print("Running episodes in serial.", end="")
    for i in range(n_eps):
        drawn_index = np.random.choice(indices)
        startboard = SudokuBoard(testset[int(drawn_index)][0], testset[int(drawn_index)][1])
        _ , gameinstance = PlayEpisode(startboard, n_runs, cpuct, temp, neuralnet)
        scores.append(gameinstance.getFinalScore())
        print(".", end="")
    print("Done!")
    return scores
