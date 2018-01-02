import numpy as np

class SudokuBoard():
    def __init__(self, initboard, solnboard, hintsmask=None):
        """ BOTH ARGS must be length-81 STRINGS"""
        self._board = initboard
        self._solution = solnboard
        if hintsmask == None: self._initialize_hintsmask()
        else: self._hintsmask = hintsmask

    def _initialize_hintsmask(self):
        self._hintsmask = "" # NOTE: 1 for changeable values, 0 for hints
        for i in range(81):
            if (self._board[i] == "0"): self._hintsmask += "1"
            else: self._hintsmask += "0"

    def __str__(self):
        final = "-"*13 + "\n"
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    final += "|"+self._board[i*27+j*9+k*3:i*27+j*9+k*3+3]
                final += "|\n"
            final += "-"*13 + "\n"
        return final

    def changevalue(self, indices):
        """ indices must be a tuple (number, i, j)"""
        num, i, j = indices
        if self._hintsmask[i*9+j]=="1":
            # NOTE: we input num+1 because indices are of the range [0,8]
            self._board = self._board[:i*9+j]+str(num+1)+self._board[i*9+j+1:]

    def generate_matrix(self):
        return np.array([[[int(self._board[i*9+j]) for j in range(9)] for i in range(9)],
            [[int(self._hintsmask[i*9+j]) for j in range(9)] for i in range(9)]])

    def will_terminate(self, indices):
        num, i, j = indices
        x = self._board
        if self._hintsmask[i*9+j] == "1": x = self._board[:i*9+j]+str(num+1)+self._board[i*9+j+1:]
        for i in range(81):
            if x[i] == "0": return False
        return True

    def is_terminal(self):
        for i in range(81):
            if self._board[i] == "0": return False
        return True

    def generate_next_board(self,indices):
        num, i, j = indices
        x = SudokuBoard(self._board, self._solution, self._hintsmask)
        if x._hintsmask[i*9+j]=="1":
            # NOTE: we input num+1 because indices are of the range [0,8]
            x._board = x._board[:i*9+j]+str(num+1)+x._board[i*9+j+1:]
        return x

    def completion(self):
        """ computes the percentage of fillable cells correctly filled so far"""
        num_errors = 0
        for i in range(81):
            if self._board[i] != self._solution[i]: num_errors += 1
        num_fillable_cells = 0
        for i in range(81):
            if self._hintsmask[i]=="1": num_fillable_cells += 1
        return 1-float(num_errors)/num_fillable_cells

class SudokuGame():
    def __init__(self, startboard, maxmoves=100, keeptrack=False):
        self._board = SudokuBoard(startboard._board, startboard._solution, startboard._hintsmask)
        self._gameSequence = []
        self._keeptrack = keeptrack
        self._nummoves = 0
        self._maxmoves = maxmoves

    def inputAction(self, action):
        self._board.changevalue(action)
        self._nummoves += 1
        if self._keeptrack: self._gameSequence.append(self._board._board)

    def gameHasEnded(self):
        # NOTE: we will stop the game if it has already taken MAXMOVES steps
        return self._board.is_terminal() or self._nummoves > self._maxmoves

    def getFinalScore(self):
        return self._board.completion()

    def num_actions_taken(self):
        return self._nummoves

    def printGameSequence(self):
        # NOTE: We can write another method that gives a more readable trace of moves
        for c in range(len(self._gameSequence)):
            final = "-"*13 + "\n"
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        final += "|"+self._gameSequence[c][i*27+j*9+k*3:i*27+j*9+k*3+3]
                    final += "|\n"
                final += "-"*13 + "\n"
            print(final)

class MCTSnode():
    def __init__(self, board, parent=0, lastAction=0):
        self._children = {}
        self._parent = parent
        self._lastAction = lastAction
        self._P = np.zeros((9,9,9))
        self._Q = np.zeros((9,9,9))
        self._N = np.zeros((9,9,9))
        self._board = board

    def __str__(self):
        return str(self._board)+str(self._children)+"\nParent: "+str(self._parent)

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent == 0

    def get_best_action(self, cpuct):
        n_sum = np.sum(self._N)
        # in case N is full of zeros, choose a random valid action to prevent bias to (0,0,0)
        if n_sum == 0:
            i, j, num = np.random.randint(0,9,3)
            i, j = int(i), int(j)
            while self._board._hintsmask[i*9+j] == "0":
                i, j, num = np.random.randint(0,9,3)
                i, j = int(i), int(j)
            return (num, i, j)
        values = self._Q + cpuct*np.sqrt(n_sum)*self._P/(1+self._N)
        return np.unravel_index(np.argmax(values), (9,9,9))

    def get_next_node(self, nextAction):
        """ nextAction is an (i,j,number) tuple """
        try:
            return self._children[nextAction]
        except:
            return None

    def get_parent(self):
        return self._parent

    def get_last_action(self):
        if self._lastAction != 0: return self._lastAction
        else: return None

    def get_N_vector(self):
        return self._N

    def make_child_node(self, nextAction):
        x = MCTSnode(self._board.generate_next_board(nextAction), self, nextAction)
        self._children[nextAction] = x
        return x

    def update_mcts_stats(self, new_v, targetAction):
        num, i, j = targetAction
        self._Q[num,i,j] = (self._Q[num,i,j]*self._N[num,i,j] + new_v)/(self._N[num,i,j]+1)
        self._N[num,i,j] += 1

    def update_prob_vector(self, probvec):
        self._P = probvec

    def is_terminating_edge(self,targetAction):
        return (not targetAction in self._children) or self._board.will_terminate(targetAction)

    def generate_matrix(self):
        return self._board.generate_matrix()

    def get_max_depth(self):
        if self.is_leaf(): return 0
        else: return max([node.get_max_depth() for action,node in self._children.items()]) + 1

    def set_as_root(self):
        self._parent = 0

def MCTSrun(node, cpuct, neuralnet):
    currentnode = node
    nextAction = currentnode.get_best_action(cpuct)
    while not currentnode.is_terminating_edge(nextAction):
        currentnode = currentnode.get_next_node(nextAction)
        nextAction = currentnode.get_best_action(cpuct)
    newnode = currentnode.make_child_node(nextAction)
    probvec, newvalue = neuralnet(newnode.generate_matrix())
    newnode.update_prob_vector(probvec)
    currentnode.update_mcts_stats(newvalue, nextAction)
    while not currentnode.is_root():
        nextAction = currentnode.get_last_action()
        currentnode = currentnode.get_parent()
        currentnode.update_mcts_stats(newvalue, nextAction)

def PlayEpisode(startboard, numRuns, cpuct, temp, neuralnet):
    gameinstance = SudokuGame(startboard)
    currentnode = MCTSnode(startboard)
    probvec, v = neuralnet(currentnode.generate_matrix())
    currentnode.update_prob_vector(probvec)
    examples = []
    states = []
    probvectors = []
    indices = np.arange(729)
    superroot = currentnode
    while True:
        for i in range(numRuns):
            MCTSrun(currentnode, cpuct, neuralnet)
        states.append(currentnode.generate_matrix())
        N = currentnode.get_N_vector()
        pi = N**(1/temp)/np.sum(N**(1/temp))
        probvectors.append(pi)
        nextAction = np.unravel_index(np.random.choice(indices,p=pi.flatten()), (9,9,9))
        currentnode = currentnode.get_next_node(nextAction)
        currentnode.set_as_root()
        gameinstance.inputAction(nextAction)
        if gameinstance.gameHasEnded():
            z = gameinstance.getFinalScore()
            for i in range(len(states)):
                examples.append((states[i],probvectors[i],z))
            return examples,gameinstance

def randomprobs(matrix):
    return np.random.random_sample((9,9,9))

def fakeneuralnet(matrix):
    value = np.random.random_sample()
    return (randomprobs(matrix), value)

sample = "000500006020010000030000000900600050010020000000000000500804000100000270000000300"
soln = "791542836625318947834769521947683152316425798258197463572834619183956274469271385"
