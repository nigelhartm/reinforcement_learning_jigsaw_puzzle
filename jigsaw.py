# conda create --name test10
# conda activate test10

# Import libraries
import numpy as np
import random
import gym # open AI

# CLASS jigsaw_game
#
# Task: information about a jigsaw game. (Field)
#
class jigsaw_game:
    board = None
    rows = None
    cols = None
    
    # Constructor -> empty field
    def __init__(self):
        self.board = np.zeros((4,6), dtype=int)
        self.rows = self.board.shape[0]
        self.cols = self.board.shape[1]

    # Check if game is solved return True if it is
    def solved(self):
        if(np.array_equal(self.board, np.ones((4,6), dtype=int))):
            return True
        else:
            return False
    
    # Check if putting piece at x, y is valid True if it is
    def valid_move(self, act_piece, x_origin, y_origin):
        board_piece_area = self.board[y_origin:y_origin+act_piece.rows, x_origin:x_origin+act_piece.cols]
        if(np.max(np.add(board_piece_area, act_piece.form)) == 1):
            return True
        else:
            return False
    
    # check if move is valid and move -> True if worked
    def move(self, act_piece, x_origin, y_origin):
        try:
            possible_move = self.valid_move(act_piece, x_origin, y_origin)
            if(possible_move):
                self.board[y_origin:y_origin+act_piece.rows, x_origin:x_origin+act_piece.cols] = np.add(self.board[y_origin:y_origin+act_piece.rows, x_origin:x_origin+act_piece.cols], act_piece.form)
                return True
            else:
                return False
        except:
            return False





# CLASS jigsaw_piece
#
# Task: Hold information about a puzzle piece, its form/color/etc. it also creates a random piece by using its constructor
#
class jigsaw_piece:
    form = None
    color = None
    rows = None
    cols = None

    # Constructor
    def __init__(self):
        piece_number = random.randint(0, 5)
        if(piece_number == 0):
            self.form = np.array([[1, 0], [1, 1]], dtype=int)
            self.color = "green"
        elif(piece_number == 1):
            self.form = np.array([[1, 1], [0, 1]], dtype=int)
            self.color = "yellow"
        elif(piece_number == 2):
            self.form = np.array([[1, 1], [1, 1]], dtype=int)
            self.color = "brightblue"
        elif(piece_number == 3):
            self.form = np.array([[1], [1], [1]], dtype=int)
            self.color = "darkblue"
        elif(piece_number == 4):
            self.form = np.array([[1, 1, 0], [0, 1, 1]], dtype=int)
            self.color = "red"
        elif(piece_number == 5):
            self.form = np.array([[1]], dtype=int)
            self.color = "orange"
        self.rows = self.form.shape[0]
        self.cols = self.form.shape[1]

def run_brute_force():
    # Init game
    act_jigsaw_game = jigsaw_game()
    #print(act_jigsaw_game.board)

    # Until solved
    turn = 0
    not_used = 0
    used = 0
    while(act_jigsaw_game.solved() == False):
        turn += 1
        
        act_jigsaw_piece = jigsaw_piece() # get new piece
        worked = False
        
        for y in range(0, act_jigsaw_game.rows):
            for x in range(0, act_jigsaw_game.cols):
                if(act_jigsaw_game.move(act_jigsaw_piece, x, y) == True):
                    worked = True
                    used +=1
                    #print("Piece added.")
                    #print(act_jigsaw_piece.form)
                    #print(act_jigsaw_game.board)
                    break
            if worked == True:
                break
        if(worked == False):
            not_used += 1
            #print("Piece not added.")
            #print(act_jigsaw_piece.form)
    #print(act_jigsaw_game.board)
    return(turn,used,not_used)

"""
class ai_jigsaw_game(gym.Env):
    board = None
    rows = None
    cols = None
    
    # Constructor -> empty field
    def __init__(self):
        self.board = np.zeros((4,6), dtype=int)
        self.rows = self.board.shape[0]
        self.cols = self.board.shape[1]

        self.action_space = gym.spaces.Discrete(self.rows*self.cols)
        self.observation_space = 
        return observation
    def step(self, action):
        return  self._get_obs(), reward, done, info
    def render(self):
"""     

def main():
    #statistic brute force
    g_turn = 0
    g_used = 0
    g_not_used = 0
    for i in range(0, 100):
        turn, used, not_used = run_brute_force()
        g_turn += turn
        g_used += used
        g_not_used += not_used
    print("Average Brute Force (Turn: " + str(g_turn/100) + " Used: " + str(g_used/100) + " Not used: " + str(g_not_used/100))


main()
