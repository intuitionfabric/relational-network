import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def convert(value):
    if value=='R': return 1
    elif value=='B': return 2
    elif value=='O': return 3
    elif value=='G': return 4
    elif value=='W': return 5
    elif value=='Y': return 6

moves_dict = {0:"L", 1:"L'", 2:"R", 3:"R'", 4:"U", 5:"U'", 6:"D", 7:"D'",
                8:"F", 9:"F'", 10:"B", 11:"B'", 12:"M", 13:"M'", 14:"E", 15:"E'",
                16:"S", 17:"S'", 18:"X", 19:"Y", 20:"Z"}

SHIFTL1 = [0,3,6,45,48,51,26,23,20,36,39,42]
SHIFTL2 = [29,32,35,34,33,30,27,28]
SHIFTR1 = [8,5,2,44,41,38,18,21,24,53,50,47]
SHIFTR2 = [9,10,11,14,17,16,15,12]
SHIFTU1 = [2,1,0,29,28,27,20,19,18,11,10,9]
SHIFTU2 = [44,43,42,39,36,37,38,41]
SHIFTD1 = [6,7,8,15,16,17,24,25,26,33,34,35]
SHIFTD2 = [45,46,47,50,53,52,51,48]
SHIFTF1 = [9,12,15,47,46,45,35,32,29,42,43,44]
SHIFTF2 = [0,1,2,5,8,7,6,3]
SHIFTB1 = [17,14,11,38,37,36,27,30,33,51,52,53]
SHIFTB2 = [18,19,20,23,26,25,24,21]
SHIFTY1 = [3,4,5,12,13,14,21,22,23,30,31,32]
SHIFTX1 = [7,4,1,43,40,37,19,22,25,52,49,46]
SHIFTZ1 = [10,13,16,50,49,48,34,31,28,39,40,41]
TERMINAL_STATES = {'RRRRRRRRRYYYYYYYYYOOOOOOOOOWWWWWWWWWBBBBBBBBBGGGGGGGGG',
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
                   'YYYYYYYYYGGGGGGGGGWWWWWWWWWBBBBBBBBBOOOOOOOOORRRRRRRRR'}

rubikcmap = colors.ListedColormap(['k','r','b','m','g','w','y'])
bounds=[0,1,2,3,4,5,6]
norm = colors.BoundaryNorm(bounds, rubikcmap.N)

class RubiksCube():
    def __init__(self, rawstate, moves_taken=0, max_moves=100, recent_history=[]):
        self._state = rawstate
        self._moves_taken = moves_taken
        self._max_moves = max_moves
        if recent_history != []: self._recent_history = recent_history
        else: self._recent_history = [rawstate]

    def generate_matrix(self):
        fullmatrix = []
        for state in self._recent_history[-1::-1]:
            rubiks = []
            for i in range(54): rubiks.append(convert(self._state[i])/6.0)
            rubiksnew = [[0 for i in range(3)] + [rubiks[36],rubiks[37],rubiks[38]] + [0 for i in range(3)],
                         [0 for i in range(3)] + [rubiks[39],rubiks[40],rubiks[41]] + [0 for i in range(3)],
                         [0 for i in range(3)] + [rubiks[42],rubiks[43],rubiks[44]] + [0 for i in range(3)],
                         [rubiks[27],rubiks[28],rubiks[29],rubiks[0],rubiks[1],rubiks[2],rubiks[9],rubiks[10],rubiks[11]],
                         [rubiks[30],rubiks[31],rubiks[32],rubiks[3],rubiks[4],rubiks[5],rubiks[12],rubiks[13],rubiks[14]],
                         [rubiks[33],rubiks[34],rubiks[35],rubiks[6],rubiks[7],rubiks[8],rubiks[15],rubiks[16],rubiks[17]],
                         [0 for i in range(3)] + [rubiks[45],rubiks[46],rubiks[47]] + [rubiks[18],rubiks[19],rubiks[20]],
                         [0 for i in range(3)] + [rubiks[48],rubiks[49],rubiks[50]] + [rubiks[21],rubiks[22],rubiks[23]],
                         [0 for i in range(3)] + [rubiks[51],rubiks[52],rubiks[53]] + [rubiks[24],rubiks[25],rubiks[26]]]
            fullmatrix.append(rubiksnew)
        current_len = len(fullmatrix)
        for i in range(5-current_len): fullmatrix.append([[0 for i in range(9)] for j in range(9)])
        return np.array(fullmatrix)

    def plot_cube_alt(self, save=False, fname="rubikscube", title='Rubiks cube'):
        """
        Plot the rubik's cube as a 9x9 matrix. See documentation for why

        Use no arguments - a plt.figure() shows up
        Use save = True to save the figure instead into the "figures" folder in the same directory
        """
        rubiks = []
        for i in range(54): rubiks.append(convert(self._state[i]))
        rubiksnew = [[0 for i in range(3)] + [rubiks[36],rubiks[37],rubiks[38]] + [0 for i in range(3)],
                     [0 for i in range(3)] + [rubiks[39],rubiks[40],rubiks[41]] + [0 for i in range(3)],
                     [0 for i in range(3)] + [rubiks[42],rubiks[43],rubiks[44]] + [0 for i in range(3)],
                     [rubiks[27],rubiks[28],rubiks[29],rubiks[0],rubiks[1],rubiks[2],rubiks[9],rubiks[10],rubiks[11]],
                     [rubiks[30],rubiks[31],rubiks[32],rubiks[3],rubiks[4],rubiks[5],rubiks[12],rubiks[13],rubiks[14]],
                     [rubiks[33],rubiks[34],rubiks[35],rubiks[6],rubiks[7],rubiks[8],rubiks[15],rubiks[16],rubiks[17]],
                     [0 for i in range(3)] + [rubiks[45],rubiks[46],rubiks[47]] + [rubiks[18],rubiks[19],rubiks[20]],
                     [0 for i in range(3)] + [rubiks[48],rubiks[49],rubiks[50]] + [rubiks[21],rubiks[22],rubiks[23]],
                     [0 for i in range(3)] + [rubiks[51],rubiks[52],rubiks[53]] + [rubiks[24],rubiks[25],rubiks[26]]]
        x = np.array(rubiksnew)
        plt.imshow(x, interpolation='nearest', cmap=rubikcmap)
        plt.title(title)
        plt.tight_layout()
        if save: plt.savefig("figures/"+fname+".png")
        else: plt.show()

    def plot_cube(self, save=False, fname="rubikscube", title='Rubiks cube'):
        """
        Plot the rubik's cube as a 9x12 matrix as if the faces of the cube are unwrapped

        Use no arguments - a plt.figure() shows up
        Use save = True to save the figure instead into the "figures" folder in the same directory
        """
        rubiks = []
        for i in range(54): rubiks.append(convert(self._state[i]))
        rubiksnew = [[0 for i in range(3)] + [rubiks[36],rubiks[37],rubiks[38]] + [0 for i in range(6)],
                     [0 for i in range(3)] + [rubiks[39],rubiks[40],rubiks[41]] + [0 for i in range(6)],
                     [0 for i in range(3)] + [rubiks[42],rubiks[43],rubiks[44]] + [0 for i in range(6)],
                     [rubiks[27],rubiks[28],rubiks[29],rubiks[0],rubiks[1],rubiks[2],rubiks[9],rubiks[10],rubiks[11],rubiks[18],rubiks[19],rubiks[20]],
                     [rubiks[30],rubiks[31],rubiks[32],rubiks[3],rubiks[4],rubiks[5],rubiks[12],rubiks[13],rubiks[14],rubiks[21],rubiks[22],rubiks[23]],
                     [rubiks[33],rubiks[34],rubiks[35],rubiks[6],rubiks[7],rubiks[8],rubiks[15],rubiks[16],rubiks[17],rubiks[24],rubiks[25],rubiks[26]],
                     [0 for i in range(3)] + [rubiks[45],rubiks[46],rubiks[47]] + [0 for i in range(6)],
                     [0 for i in range(3)] + [rubiks[48],rubiks[49],rubiks[50]] + [0 for i in range(6)],
                     [0 for i in range(3)] + [rubiks[51],rubiks[52],rubiks[53]] + [0 for i in range(6)]]
        x = np.array(rubiksnew)
        plt.imshow(x, interpolation='nearest', cmap=rubikcmap)
        plt.title(title)
        plt.tight_layout()
        if save: plt.savefig("figures/"+fname+".png")
        else: plt.show()

    def __str__(self):
        output = ""
        output += "___|" + self._state[36:39] + "|___|___\n"
        output += "___|" + self._state[39:42] + "|___|___\n"
        output += "___|" + self._state[42:45] + "|___|___\n"
        output += self._state[27:30]+"|"+ self._state[:3]+"|" + self._state[9:12]+"|" + self._state[18:21] + "\n"
        output += self._state[30:33]+"|" + self._state[3:6]+"|" + self._state[12:15]+"|" + self._state[21:24] + "\n"
        output += self._state[33:36]+"|" + self._state[6:9]+"|" + self._state[15:18]+"|" + self._state[24:27] + "\n"
        output += "___|" + self._state[45:48] + "|___|___\n"
        output += "___|" + self._state[48:51] + "|___|___\n"
        output += "___|" + self._state[51:54] + "|___|___\n"
        return output

    def apply_shift(self, shift_index_array, num_right_shifts):
        templist = []
        for index in shift_index_array:
            templist.append(self._state[index])
        templist = templist[-num_right_shifts:] + templist[:-num_right_shifts]
        for i in range(len(shift_index_array)):
            self._state = self._state[:shift_index_array[i]]+templist[i] + self._state[shift_index_array[i]+1:]

    def apply_move_sequence(self, move_list):
        for move in move_list: self.apply_move(move)

    def apply_move(self, move):
        """
        Apply one of the defined 15 possible moves (0<= move <= 14) on this RubiksCube object
        """
        if move == 0:
            self.apply_shift(SHIFTL1,3)
            self.apply_shift(SHIFTL2,2)
        elif move == 1:
            self.apply_shift(SHIFTL1,-3)
            self.apply_shift(SHIFTL2,-2)
        elif move == 2:
            self.apply_shift(SHIFTR1,3)
            self.apply_shift(SHIFTR2,2)
        elif move == 3:
            self.apply_shift(SHIFTR1,-3)
            self.apply_shift(SHIFTR2,-2)
        elif move == 4:
            self.apply_shift(SHIFTU1,3)
            self.apply_shift(SHIFTU2,2)
        elif move == 5:
            self.apply_shift(SHIFTU1,-3)
            self.apply_shift(SHIFTU2,-2)
        elif move == 6:
            self.apply_shift(SHIFTD1,3)
            self.apply_shift(SHIFTD2,2)
        elif move == 7:
            self.apply_shift(SHIFTD1,-3)
            self.apply_shift(SHIFTD2,-2)
        elif move == 8:
            self.apply_shift(SHIFTF1,3)
            self.apply_shift(SHIFTF2,2)
        elif move == 9:
            self.apply_shift(SHIFTF1,-3)
            self.apply_shift(SHIFTF2,-2)
        elif move == 10:
            self.apply_shift(SHIFTB1,3)
            self.apply_shift(SHIFTB2,2)
        elif move == 11:
            self.apply_shift(SHIFTB1,-3)
            self.apply_shift(SHIFTB2,-2)
        elif move == 12:
            self.apply_shift(SHIFTX1,-3)
        elif move == 13:
            self.apply_shift(SHIFTX1, 3)
        elif move == 14:
            self.apply_shift(SHIFTY1, 3)
        elif move == 15:
            self.apply_shift(SHIFTY1,-3)
        elif move == 16:
            self.apply_shift(SHIFTZ1, 3)
        elif move == 17:
            self.apply_shift(SHIFTZ1,-3)
        elif move == 18:
            # NOTE: this is the X rotation
            self.apply_shift(SHIFTL1,-3)
            self.apply_shift(SHIFTX1,3)
            self.apply_shift(SHIFTR1,3)
            self.apply_shift(SHIFTL2,-2)
            self.apply_shift(SHIFTR2,2)
        elif move == 19:
            # NOTE: this is the Y rotation
            self.apply_shift(SHIFTU1,3)
            self.apply_shift(SHIFTY1,-3)
            self.apply_shift(SHIFTD1,-3)
            self.apply_shift(SHIFTU2,2)
            self.apply_shift(SHIFTD2,-2)
        elif move == 20:
            # NOTE: this is the Z rotation
            self.apply_shift(SHIFTB1,-3)
            self.apply_shift(SHIFTZ1, 3)
            self.apply_shift(SHIFTF1, 3)
            self.apply_shift(SHIFTF2, 2)
            self.apply_shift(SHIFTB2, -2)
        else:
            print("Invalid move entered: ", move)
            return
        self._moves_taken += 1
        self._recent_history.append(self._state)
        # NOTE: this number below (here, 4) sets the number of most recent past cube states saved
        if self._moves_taken > 4: self._recent_history.pop(0)

    def generate_next_cube(self, move):
        if move < 0 and move > 20:
            print("Invalid move entered: ", move)
            return None
        x = RubiksCube(self._state, self._moves_taken, self._max_moves, self._recent_history[:])
        x.apply_move(move)
        return x

    def will_terminate(self, move):
        if self._moves_taken + 1 == self._max_moves: return True
        if move < 0 and move > 20:
            print("Invalid move entered: ", move)
        x = self.generate_next_cube(move)
        return x.is_solved()

    def is_terminal(self):
        if self._moves_taken == self._max_moves: return True
        return self.is_solved()

    def is_solved(self):
        return self._state in TERMINAL_STATES

    def get_score(self):
        return 1 if self.is_solved() else 0

    def get_state(self):
        return self._state

    def get_moves_taken(self):
        return self._moves_taken

class RubikGame():
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
        if self._keep_track:
            self._state_sequence.append(self._cube.get_state())
            self._move_sequence += moves_dict[move]

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
        # NOTE: I am not sure if this is the right way to deal with an unexpanded next_move
        except:
            return self.make_child_node(next_move)

    def get_parent(self):
        return self._parent

    def get_last_move(self):
        if self._last_move != None: return self._last_move
        return None

    def get_N_vector(self):
        return self._N[:]

    def generate_matrix(self):
        return self._cube.generate_matrix()

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
        #print("nextmove = ",next_move," and current node is ",current_node)
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
    gameinstance = RubikGame(startstate, max_moves=max_moves, keep_track=True)
    startcube = RubiksCube(startstate, max_moves=max_moves)
    currentnode = MCTSnode(startcube)
    probvec, v = neuralnet(currentnode.generate_matrix())
    currentnode.update_prob_vector(probvec)
    examples = []
    states = []
    probvectors = []
    indices = np.arange(18)
    superroot = currentnode
    while True:
        for i in range(numRuns):
            MCTSrun(currentnode, cpuct, neuralnet)
        states.append(currentnode.generate_matrix())
        N = currentnode.get_N_vector()
        pi = N**(1/temp)/np.sum(N**(1/temp))
        probvectors.append(pi)
        nextmove = np.random.choice(indices,p=pi)
        currentnode = currentnode.get_next_node(nextmove)
        currentnode.set_as_root()
        gameinstance.input_move(nextmove)
        if gameinstance.game_has_ended():
            z = gameinstance.get_score()
            for i in range(len(states)):
                examples.append((states[i],probvectors[i],z))
            return examples,gameinstance,superroot

def randomprobs(matrix):
    return np.random.random_sample(18)

def fakeneuralnetwork(matrix):
    value = np.random.random_sample()
    return (randomprobs(matrix), value)

raw = "RRRRRRRRRBBBBBBBBBOOOOOOOOOGGGGGGGGGWWWWWWWWWYYYYYYYYY"
problem3 = 'BBBBBBRRWYOOYOOBBBGGGGGGYOORRWRRWGGGWWWWWWOOOYYRYYRYYR'
problem1 = 'RRRRRRRRRBWBBWBBWBOOOOOOOOOGYGGYGGYGWWWGGGWWWYYYBBBYYY'
problem5 = 'YOORRWRRBGGRBBYYYYGYROOROOWBBBBGWOGWWBYOWWOWWGGRYYRBGG'
cube = RubiksCube(problem3)
root = MCTSnode(cube)

print("Starting cube state: ")
cube.plot_cube()
examples, game, root = PlayEpisode(problem3,100000,0.1,0.1,fakeneuralnetwork, 10)
print(root)
print("Game score = ",game.get_score())
game.show_cube()
