import numpy as np

class RubiksCubeOld():
    def __init__(self, rawstate):
        self._state = rawstate

    def apply_shift(self, shift_index_array, num_right_shifts):
        templist = []
        for index in shift_index_array:
            templist.append(self._state[index])
        templist = templist[-num_right_shifts:] + templist[:-num_right_shifts]
        for i in range(len(shift_index_array)):
            self._state = self._state[:shift_index_array[i]]+templist[i] + self._state[shift_index_array[i]+1:]

    def apply_move(self, move):
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

    def get_state(self):
        return self._state

def generate_cubes(num_states, distance, startstate='RRRRRRRRRBBBBBBBBBOOOOOOOOOGGGGGGGGGWWWWWWWWWYYYYYYYYY'):
    """
    Generates a list of randomly generated rubik's cube states
    """
    terminal_states = {'RRRRRRRRRYYYYYYYYYOOOOOOOOOWWWWWWWWWBBBBBBBBBGGGGGGGGG',
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
    states = []
    while len(states) < num_states:
        x = RubiksCubeOld(startstate)
        for j in range(distance):
            x.apply_move(np.random.randint(0,18))
        newstate = x.get_state()
        if newstate not in terminal_states: states.append(newstate)
        states = list(set(states))

    return states

#print(generate_cubes(1,"RRRRRRRRRBBBBBBBBBOOOOOOOOOGGGGGGGGGWWWWWWWWWYYYYYYYYY",3))
