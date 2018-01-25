import random

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
