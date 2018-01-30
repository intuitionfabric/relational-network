import random

from rubik import *

raw = "RRRRRRRRRBBBBBBBBBOOOOOOOOOGGGGGGGGGWWWWWWWWWYYYYYYYYY"
problem5 = 'YOORRWRRBGGRBBYYYYGYROOROOWBBBBGWOGWWBYOWWOWWGGRYYRBGG' # distance-5 problem
problem3 = 'BBBBBBRRWYOOYOOBBBGGGGGGYOORRWRRWGGGWWWWWWOOOYYRYYRYYR' # distance-3 problem
fname = "sudoku17.txt"

def extract(filename):
    with open(filename) as f:
        lines = f.readlines()
    dataset = [line.strip().split(',') for line in lines]
    random.shuffle(dataset)
    return dataset

def add_hints(problem, num_hints):
    q, a = problem
    for i in range(num_hints):
        empty_places = [k for k, v in enumerate(q) if v == '0']
        ind = random.choice(empty_places)
        q = q[:ind] + a[ind] + q[(ind + 1):]
    return q, a

def generate(dataset, num_new_hints, num_probs):
    """
    generate a set containing num_probs sudoku problems all with num_new_hints extra added hints
    """
    newset = []
    for i in range(num_probs):
        problem = random.choice(dataset)
        problem = add_hints(problem, num_new_hints)
        newset.append(problem)
    return newset

def generate_cubes(num_states, startstate, distance):
    """
    Generates a list of randomly generated rubik's cube states
    """
    states = []
    for i in range(num_states):
        x = RubiksCube(startstate)
        for j in range(distance):
            x.apply_move(random.randint(0,20))
        states.append(x.get_state())
    states = list(set(states))
    return states
