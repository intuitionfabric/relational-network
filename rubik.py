import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

MOVES_DICT = {0:"L", 1:"L'", 2:"R", 3:"R'", 4:"U", 5:"U'", 6:"D", 7:"D'",
                8:"F", 9:"F'", 10:"B", 11:"B'", 12:"M", 13:"M'", 14:"E", 15:"E'",
                16:"S", 17:"S'", 18:"X", 19:"Y", 20:"Z", 21:"X'", 22:"Y'", 23:"Z'"}

SYMMETRY_OPS = [[],[19],[19,20], [19,20,20],[19,23],[20],[20,20],[23],
                [19,19],[19,19,20],[19,19,20,20],[19,19,23],
                [22],[22,20],[22,20,20],[22,23],[21],[21,20],[21,20,20],
                [21,23], [18], [18,20],[18,20,20],[18,23]]

SHIFTL1 = [(3,3), (4,3), (5,3), (6,3), (7,3), (8,3), (8,8),(7,8),(6,8),(0,3), (1,3), (2,3)]
SHIFTL2 = [(3,2), (4,2), (5,2), (5,1), (5,0), (4,0), (3,0), (3,1)]
SHIFTR1 = [(5,5),(4,5),(3,5),(2,5),(1,5),(0,5),(6,6),(7,6),(8,6),(8,5),(7,5),(6,5)]
SHIFTR2 = [(3,6),(3,7),(3,8),(4,8),(5,8),(5,7),(5,6),(4,6)]
SHIFTU1 = [(3,5),(3,4),(3,3),(3,2),(3,1),(3,0),(6,8),(6,7),(6,6),(3,8),(3,7),(3,6)]
SHIFTU2 = [(2,5),(2,4),(2,3),(1,3),(0,3),(0,4),(0,5),(1,5)]
SHIFTD1 = [(5,3),(5,4),(5,5),(5,6),(5,7),(5,8),(8,6),(8,7),(8,8),(5,0),(5,1),(5,2)]
SHIFTD2 = [(6,3),(6,4),(6,5),(7,5),(8,5),(8,4),(8,3),(7,3)]
SHIFTF1 = [(3,6),(4,6),(5,6),(6,5),(6,4),(6,3),(5,2),(4,2),(3,2),(2,3),(2,4),(2,5)]
SHIFTF2 = [(3,3),(3,4),(3,5),(4,5),(5,5),(5,4),(5,3),(4,3)]
SHIFTB1 = [(5,8),(4,8),(3,8),(0,5),(0,4),(0,3),(3,0),(4,0),(5,0),(8,3),(8,4),(8,5)]
SHIFTB2 = [(6,6),(6,7),(6,8),(7,8),(8,8),(8,7),(8,6),(7,6)]
SHIFTY1 = [(4,3),(4,4),(4,5),(4,6),(4,7),(4,8),(7,6),(7,7),(7,8),(4,0),(4,1),(4,2)]
SHIFTX1 = [(5,4),(4,4),(3,4),(2,4),(1,4),(0,4),(6,7),(7,7),(8,7),(8,4),(7,4),(6,4)]
SHIFTZ1 = [(3,7),(4,7),(5,7),(7,5),(7,4),(7,3),(5,1),(4,1),(3,1),(1,3),(1,4),(1,5)]

raw_terminal_states = ['RRRRRRRRRYYYYYYYYYOOOOOOOOOWWWWWWWWWBBBBBBBBBGGGGGGGGG',
                   'BBBBBBBBBYYYYYYYYYGGGGGGGGGWWWWWWWWWOOOOOOOOORRRRRRRRR',
                   'YYYYYYYYYBBBBBBBBBWWWWWWWWWGGGGGGGGGRRRRRRRRROOOOOOOOO',
                   'BBBBBBBBBWWWWWWWWWGGGGGGGGGYYYYYYYYYRRRRRRRRROOOOOOOOO',
                   'OOOOOOOOOGGGGGGGGGRRRRRRRRRBBBBBBBBBWWWWWWWWWYYYYYYYYY',
                   'GGGGGGGGGOOOOOOOOOBBBBBBBBBRRRRRRRRRYYYYYYYYYWWWWWWWWW',
                   'GGGGGGGGGRRRRRRRRRBBBBBBBBBOOOOOOOOOWWWWWWWWWYYYYYYYYY',
                   'GGGGGGGGGYYYYYYYYYBBBBBBBBBWWWWWWWWWRRRRRRRRROOOOOOOOO',
                   'RRRRRRRRRWWWWWWWWWOOOOOOOOOYYYYYYYYYGGGGGGGGGBBBBBBBBB',
                   'OOOOOOOOOWWWWWWWWWRRRRRRRRRYYYYYYYYYBBBBBBBBBGGGGGGGGG',
                   'WWWWWWWWWRRRRRRRRRYYYYYYYYYOOOOOOOOOBBBBBBBBBGGGGGGGGG',
                   'BBBBBBBBBOOOOOOOOOGGGGGGGGGRRRRRRRRRWWWWWWWWWYYYYYYYYY',
                   'BBBBBBBBBRRRRRRRRRGGGGGGGGGOOOOOOOOOYYYYYYYYYWWWWWWWWW',
                   'RRRRRRRRRGGGGGGGGGOOOOOOOOOBBBBBBBBBYYYYYYYYYWWWWWWWWW',
                   'YYYYYYYYYRRRRRRRRRWWWWWWWWWOOOOOOOOOGGGGGGGGGBBBBBBBBB',
                   'YYYYYYYYYOOOOOOOOOWWWWWWWWWRRRRRRRRRBBBBBBBBBGGGGGGGGG',
                   'GGGGGGGGGWWWWWWWWWBBBBBBBBBYYYYYYYYYOOOOOOOOORRRRRRRRR',
                   'WWWWWWWWWGGGGGGGGGYYYYYYYYYBBBBBBBBBRRRRRRRRROOOOOOOOO',
                   'OOOOOOOOOYYYYYYYYYRRRRRRRRRWWWWWWWWWGGGGGGGGGBBBBBBBBB',
                   'RRRRRRRRRBBBBBBBBBOOOOOOOOOGGGGGGGGGWWWWWWWWWYYYYYYYYY',
                   'WWWWWWWWWBBBBBBBBBYYYYYYYYYGGGGGGGGGOOOOOOOOORRRRRRRRR',
                   'WWWWWWWWWOOOOOOOOOYYYYYYYYYRRRRRRRRRGGGGGGGGGBBBBBBBBB',
                   'OOOOOOOOOBBBBBBBBBRRRRRRRRRGGGGGGGGGYYYYYYYYYWWWWWWWWW',
                   'YYYYYYYYYGGGGGGGGGWWWWWWWWWBBBBBBBBBOOOOOOOOORRRRRRRRR']

def GenerateRubiksMatrix(rawstr):
    def convert(value):
        if value=='R': return 1./6
        elif value=='B': return 2./6
        elif value=='O': return 3./6
        elif value=='G': return 4./6
        elif value=='W': return 5./6
        elif value=='Y': return 1.0
    rubiks = [convert(char) for char in rawstr]
    rubiksnew = [[0 for i in range(3)] + [rubiks[36],rubiks[37],rubiks[38]] + [0 for i in range(3)],
                 [0 for i in range(3)] + [rubiks[39],rubiks[40],rubiks[41]] + [0 for i in range(3)],
                 [0 for i in range(3)] + [rubiks[42],rubiks[43],rubiks[44]] + [0 for i in range(3)],
                 [rubiks[27],rubiks[28],rubiks[29],rubiks[0],rubiks[1],rubiks[2],rubiks[9],rubiks[10],rubiks[11]],
                 [rubiks[30],rubiks[31],rubiks[32],rubiks[3],rubiks[4],rubiks[5],rubiks[12],rubiks[13],rubiks[14]],
                 [rubiks[33],rubiks[34],rubiks[35],rubiks[6],rubiks[7],rubiks[8],rubiks[15],rubiks[16],rubiks[17]],
                 [0 for i in range(3)] + [rubiks[45],rubiks[46],rubiks[47]] + [rubiks[18],rubiks[19],rubiks[20]],
                 [0 for i in range(3)] + [rubiks[48],rubiks[49],rubiks[50]] + [rubiks[21],rubiks[22],rubiks[23]],
                 [0 for i in range(3)] + [rubiks[51],rubiks[52],rubiks[53]] + [rubiks[24],rubiks[25],rubiks[26]]]
    return np.array(rubiksnew)

TERMINAL_STATES = [GenerateRubiksMatrix(rawstr) for rawstr in raw_terminal_states]

class RubiksCube():
    __slots__ = ['_states', '_moves_taken', '_max_moves']
    HISTO_MAX = 5

    def __init__(self, rawstate="", states=None, moves_taken=0, max_moves=100):
        if rawstate != "":
            self._save_rawstate(rawstate)
        else: self._states = states
        self._moves_taken = moves_taken
        self._max_moves = max_moves

    def __str__(self):
        return str(self._states[0])

    @staticmethod
    def _convert(value):
        if value=='R': return 1./6
        elif value=='B': return 2./6
        elif value=='O': return 3./6
        elif value=='G': return 4./6
        elif value=='W': return 5./6
        elif value=='Y': return 1.0
        else: raise ValueError('Encountered an invalid character in the input.')

    def _save_rawstate(self, rawstate):
        rubiks = [self._convert(char) for char in rawstate]
        rubiksnew = [[[0 for i in range(3)] + [rubiks[36],rubiks[37],rubiks[38]] + [0 for i in range(3)],
                     [0 for i in range(3)] + [rubiks[39],rubiks[40],rubiks[41]] + [0 for i in range(3)],
                     [0 for i in range(3)] + [rubiks[42],rubiks[43],rubiks[44]] + [0 for i in range(3)],
                     [rubiks[27],rubiks[28],rubiks[29],rubiks[0],rubiks[1],rubiks[2],rubiks[9],rubiks[10],rubiks[11]],
                     [rubiks[30],rubiks[31],rubiks[32],rubiks[3],rubiks[4],rubiks[5],rubiks[12],rubiks[13],rubiks[14]],
                     [rubiks[33],rubiks[34],rubiks[35],rubiks[6],rubiks[7],rubiks[8],rubiks[15],rubiks[16],rubiks[17]],
                     [0 for i in range(3)] + [rubiks[45],rubiks[46],rubiks[47]] + [rubiks[18],rubiks[19],rubiks[20]],
                     [0 for i in range(3)] + [rubiks[48],rubiks[49],rubiks[50]] + [rubiks[21],rubiks[22],rubiks[23]],
                     [0 for i in range(3)] + [rubiks[51],rubiks[52],rubiks[53]] + [rubiks[24],rubiks[25],rubiks[26]]]]
        self._states = np.concatenate((np.array(rubiksnew), np.zeros((RubiksCube.HISTO_MAX,9,9))), 0)

    def plot_cube(self, layer=0, save=False, fname="rubikscube", title='Rubiks cube'):
        """
        Plot the rubik's cube as a 9x12 matrix as if the faces of the cube are unwrapped.
        By default, the current cube state is plotted, but you can also plot a stored previous state
        by specifying a layer (0th axis of np array) of the _states matrix

        Use no arguments - a plt.figure() shows up
        Use save = True to save the figure instead into the "figures" folder in the same directory
        """
        r = self._states[layer,:,:]
        rubiksnew = [[0 for i in range(3)] + [r[0,3],r[0,4], r[0,5]] + [0 for i in range(6)],
                     [0 for i in range(3)] + [r[1,3],r[1,4], r[1,5]] + [0 for i in range(6)],
                     [0 for i in range(3)] + [r[2,3],r[2,4], r[2,5]] + [0 for i in range(6)],
                     [r[3,0],r[3,1], r[3,2],r[3,3],r[3,4], r[3,5],r[3,6],r[3,7], r[3,8],r[6,6],r[6,7], r[6,8]],
                     [r[4,0],r[4,1], r[4,2],r[4,3],r[4,4], r[4,5],r[4,6],r[4,7], r[4,8],r[7,6],r[7,7], r[7,8]],
                     [r[5,0],r[5,1], r[5,2],r[5,3],r[5,4], r[5,5],r[5,6],r[5,7], r[5,8],r[8,6],r[8,7], r[8,8]],
                     [0 for i in range(3)] + [r[6,3],r[6,4], r[6,5]] + [0 for i in range(6)],
                     [0 for i in range(3)] + [r[7,3],r[7,4], r[7,5]] + [0 for i in range(6)],
                     [0 for i in range(3)] + [r[8,3],r[8,4], r[8,5]] + [0 for i in range(6)]]
        x = np.array(rubiksnew)
        rubikcmap = colors.ListedColormap(['k','r','b','m','g','w','y'])
        bounds=[0,1,2,3,4,5,6]
        norm = colors.BoundaryNorm(bounds, rubikcmap.N)
        plt.imshow(x, interpolation='nearest', cmap=rubikcmap)
        plt.title(title)
        plt.tight_layout()
        if save: plt.savefig("figures/"+fname+".png")
        else: plt.show()

    @classmethod
    def apply_move_static(cls, state, move):
        if move == 0:
            cls.apply_shift(state,SHIFTL1,3)
            cls.apply_shift(state,SHIFTL2,2)
        elif move == 1:
            cls.apply_shift(state,SHIFTL1,-3)
            cls.apply_shift(state,SHIFTL2,-2)
        elif move == 2:
            cls.apply_shift(state,SHIFTR1,3)
            cls.apply_shift(state,SHIFTR2,2)
        elif move == 3:
            cls.apply_shift(state,SHIFTR1,-3)
            cls.apply_shift(state,SHIFTR2,-2)
        elif move == 4:
            cls.apply_shift(state,SHIFTU1,3)
            cls.apply_shift(state,SHIFTU2,2)
        elif move == 5:
            cls.apply_shift(state,SHIFTU1,-3)
            cls.apply_shift(state,SHIFTU2,-2)
        elif move == 6:
            cls.apply_shift(state,SHIFTD1,3)
            cls.apply_shift(state,SHIFTD2,2)
        elif move == 7:
            cls.apply_shift(state,SHIFTD1,-3)
            cls.apply_shift(state,SHIFTD2,-2)
        elif move == 8:
            cls.apply_shift(state,SHIFTF1,3)
            cls.apply_shift(state,SHIFTF2,2)
        elif move == 9:
            cls.apply_shift(state,SHIFTF1,-3)
            cls.apply_shift(state,SHIFTF2,-2)
        elif move == 10:
            cls.apply_shift(state,SHIFTB1,3)
            cls.apply_shift(state,SHIFTB2,2)
        elif move == 11:
            cls.apply_shift(state,SHIFTB1,-3)
            cls.apply_shift(state,SHIFTB2,-2)
        elif move == 12:
            cls.apply_shift(state,SHIFTX1,-3)
        elif move == 13:
            cls.apply_shift(state,SHIFTX1, 3)
        elif move == 14:
            cls.apply_shift(state,SHIFTY1, 3)
        elif move == 15:
            cls.apply_shift(state,SHIFTY1,-3)
        elif move == 16:
            cls.apply_shift(state,SHIFTZ1, 3)
        elif move == 17:
            cls.apply_shift(state,SHIFTZ1,-3)
        elif move == 18:
            cls.apply_shift(state,SHIFTL1,-3)
            cls.apply_shift(state,SHIFTX1, 3)
            cls.apply_shift(state,SHIFTR1, 3)
            cls.apply_shift(state,SHIFTL2,-2)
            cls.apply_shift(state,SHIFTR2, 2)
        elif move == 19:
            cls.apply_shift(state,SHIFTU1, 3)
            cls.apply_shift(state,SHIFTY1,-3)
            cls.apply_shift(state,SHIFTD1,-3)
            cls.apply_shift(state,SHIFTU2, 2)
            cls.apply_shift(state,SHIFTD2,-2)
        elif move == 20:
            cls.apply_shift(state,SHIFTB1,-3)
            cls.apply_shift(state,SHIFTZ1, 3)
            cls.apply_shift(state,SHIFTF1, 3)
            cls.apply_shift(state,SHIFTF2, 2)
            cls.apply_shift(state,SHIFTB2, -2)
        elif move == 21:
            cls.apply_shift(state,SHIFTL1, 3)
            cls.apply_shift(state,SHIFTX1,-3)
            cls.apply_shift(state,SHIFTR1,-3)
            cls.apply_shift(state,SHIFTL2, 2)
            cls.apply_shift(state,SHIFTR2,-2)
        elif move == 22:
            cls.apply_shift(state,SHIFTU1,-3)
            cls.apply_shift(state,SHIFTY1, 3)
            cls.apply_shift(state,SHIFTD1, 3)
            cls.apply_shift(state,SHIFTU2,-2)
            cls.apply_shift(state,SHIFTD2, 2)
        elif move == 23:
            cls.apply_shift(state,SHIFTB1, 3)
            cls.apply_shift(state,SHIFTZ1,-3)
            cls.apply_shift(state,SHIFTF1,-3)
            cls.apply_shift(state,SHIFTF2,-2)
            cls.apply_shift(state,SHIFTB2, 2)
        return state

    @staticmethod
    def apply_shift(states, shift_index_array, num_right_shifts):
        templist = []
        for pair in shift_index_array:
            templist.append(states[pair])
        templist = templist[-num_right_shifts:] + templist[:-num_right_shifts]
        for i, pair in enumerate(shift_index_array):
            states[pair] = templist[i]
        return states

    def apply_move(self, move):
        self._states[-1] = self._states[0]
        self._states = np.roll(self._states,1,0)
        self._states[0] = self.apply_move_static(self._states[0],move)
        self._moves_taken += 1

    def generate_symmetries(self):
        for move_sequence in SYMMETRY_OPS:
            new_state = self.get_matrix()
            for move in move_sequence:
                for i in range(RubiksCube.HISTO_MAX):
                    new_state[i] = self.apply_move_static(new_state[i], move)
            yield(new_state)

    def generate_next_cube(self, move):
        x = RubiksCube(states = self._states.copy(), moves_taken=self._moves_taken, max_moves=self._max_moves)
        x.apply_move(move)
        return x

    @staticmethod
    def is_solved(state):
        # NOTE: make sure that state is a 9x9 array (strip away the other layers)
        for npmatrix in TERMINAL_STATES:
            if np.array_equal(state, npmatrix): return True
        return False

    def will_terminate(self, move):
        if self._moves_taken + 1 == self._max_moves: return True
        if move < 0 and move > 20:
            print("Invalid move entered: ", move)
        return self.is_solved(self.apply_move_static(self._states.copy()[0], move)[0])

    def is_terminal(self):
        if self._moves_taken == self._max_moves: return True
        return self.is_solved(self._states[0])

    def get_matrix(self):
        return self._states.copy()

    def get_score(self):
        return 1 if self.is_solved(self._states[0]) else 0

    def get_moves_taken(self):
        return self._moves_taken


class RubikGame():
    __slots__ = ['_cube', '_state_sequence', '_move_sequence', '_keep_track']
    def __init__(self, start_state, max_moves=100, keep_track=False):
        self._cube = RubiksCube(start_state, max_moves=max_moves)
        self._state_sequence = []
        self._move_sequence = ""
        self._keep_track = keep_track

    def input_move(self, move):
        if move < 0 and move > 20:
            print("Invalid move entered: ", move)
            return None
        self._cube.apply_move(move)
        self._move_sequence += MOVES_DICT[move]
        if self._keep_track:
            self._state_sequence.append(self._cube.get_matrix())
            # TODO: implement auto-printing of cube state history, perhaps animation?

    def game_has_ended(self):
        return self._cube.is_terminal()

    def get_score(self):
        return self._cube.get_score()

    def num_moves_taken(self):
        return self._cube.get_moves_taken()

    def print_move_sequence(self):
        print(self._move_sequence)

    def show_cube(self):
        self._cube.plot_cube()

class MCTSnode():
    __slots__ = ['_children','_parent', '_last_move', '_P', '_Q', '_N', '_cube']
    def __init__(self, cube, parent=0, last_move=None):
        self._children = {}
        self._parent = parent
        self._last_move = last_move
        self._P = np.zeros(18)
        self._Q = np.zeros(18)
        self._N = np.zeros(18)
        self._cube = cube

    def __str__(self):
        return str(self._cube)+"N = "+str(self._N)+"\nQ = "+str(self._Q)+"\nno. of children:"+ \
            str(len(self._children.keys()))+"\nsum(N) = "+str(np.sum(self._N))

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent == 0

    def get_best_move(self, cpuct):
        n_sum = np.sum(self._N)
        if n_sum == 0: return np.random.randint(0,18)
        values = self._Q + cpuct*np.sqrt(n_sum)*self._P/(1.0+self._N)
        return np.argmax(values)

    def get_next_node(self, next_move):
        try: return self._children[next_move]
        except:
            return self.make_child_node(next_move)

    def get_parent(self):
        return self._parent

    def get_last_move(self):
        if self._last_move != None: return self._last_move
        return None

    def get_N_vector(self):
        return np.copy(self._N)

    def generate_matrix(self):
        return self._cube.get_matrix()

    def make_child_node(self, next_move):
        x = MCTSnode(self._cube.generate_next_cube(next_move), self, next_move)
        self._children[next_move] = x
        return x

    def update_mcts_stats(self, new_v, move):
        self._Q[move] = (self._Q[move]*self._N[move] + new_v)/(self._N[move]+1)
        self._N[move] += 1

    def update_prob_vector(self, probvec):
        self._P = probvec

    def is_terminal_state(self):
        return self._cube.is_terminal()

    def is_terminating_edge(self, target_move):
        return (not target_move in self._children) or self._cube.will_terminate(target_move)

    def get_score(self):
        return self._cube.get_score()

    def get_cube(self):
        return self._cube

    def get_max_depth(self):
        if self.is_leaf(): return 0
        return max([node.get_max_depth() for action, node in self._children.items()]) + 1

    def set_as_root(self):
        self._parent = 0

def MCTSrun(node, cpuct, neuralnet):
    currentnode = node
    nextmove = currentnode.get_best_move(cpuct)
    while not currentnode.is_terminating_edge(nextmove):
        currentnode = currentnode.get_next_node(nextmove)
        nextmove = currentnode.get_best_move(cpuct)
    newnode = currentnode.make_child_node(nextmove)
    if newnode.is_terminal_state():
        newval = newnode.get_score()
    else:
        probvec, newval = neuralnet(newnode.generate_matrix())
        newnode.update_prob_vector(probvec)
    currentnode.update_mcts_stats(newval, nextmove)
    while not currentnode.is_root():
        nextmove = currentnode.get_last_move()
        currentnode = currentnode.get_parent()
        currentnode.update_mcts_stats(newval, nextmove)

def PlayEpisode(startstate, numRuns, cpuct, temp, neuralnet, max_moves=100):
    gameinstance = RubikGame(startstate, max_moves=max_moves)
    startcube = RubiksCube(startstate, max_moves=max_moves)
    currentnode = MCTSnode(startcube)
    probvec, v = neuralnet(currentnode.generate_matrix())
    currentnode.update_prob_vector(probvec)
    examples = []
    states = []
    probvectors = []
    indices = np.arange(18)
    # superroot = currentnode # NOTE: for debugging
    while True:
        for i in range(numRuns):
            MCTSrun(currentnode, cpuct, neuralnet)
        states.append(currentnode.get_cube())
        N = currentnode.get_N_vector()
        pi = N**(1/temp)/np.sum(N**(1/temp))
        probvectors.append(pi)
        nextmove = np.random.choice(indices,p=pi)
        currentnode = currentnode.get_next_node(nextmove)
        currentnode.set_as_root()
        gameinstance.input_move(nextmove)
        if gameinstance.game_has_ended():
            z = gameinstance.get_score()
            for i, cube in enumerate(states):
                examples.append((cube,probvectors[i],z))
            return examples,gameinstance
