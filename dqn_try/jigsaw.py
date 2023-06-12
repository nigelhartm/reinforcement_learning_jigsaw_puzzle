# conda create --name test10
# conda activate test10

# Import libraries
import numpy as np
import random

# CLASS jigsaw_game
#
# Task: information about a jigsaw game. (Field)
#
class jigsaw_game:
    board = None
    rows = None
    cols = None
    piece = None
    reward = None
    finished = None
    
    # Constructor -> empty field
    def __init__(self):
        self.board = np.zeros((4,6), dtype=int)
        self.rows = self.board.shape[0]
        self.cols = self.board.shape[1]
        self.new_piece()
        self.reward = 0
        self.finished = False

    # Check if game is solved return True if it is
    def solved(self):
        if(np.array_equal(self.board, np.ones((4,6), dtype=int))):
            self.reward = 100
            self.finished = True
            return True
        else:
            return False
    
    # Check if putting piece at x, y is valid True if it is
    def valid_move(self, x_origin, y_origin):
        board_piece_area = self.board[y_origin:y_origin+self.piece.rows, x_origin:x_origin+self.piece.cols]
        if(np.max(np.add(board_piece_area, self.piece.form)) == 1):
            return True
        else:
            return False
    
    # check if move is valid and move -> True if worked
    def move(self, x_origin, y_origin):
        try:
            possible_move = self.valid_move(x_origin, y_origin)
            if(possible_move):
                self.board[y_origin:y_origin+self.piece.rows, x_origin:x_origin+self.piece.cols] = np.add(self.board[y_origin:y_origin+self.piece.rows, x_origin:x_origin+self.piece.cols], self.piece.form)
                self.reward = 10
                print("Piece added at position x="+ str(x_origin) + " y=" + str(y_origin) + ")")
                self.new_piece()
                return True
            else:
                print("Piece not added (Reason already other piece at position x="+ str(x_origin) + " y=" + str(y_origin) + ")")
                self.reward = -0.1
                return False
        except:
            print("Piece not added (Reason out of Field at position x="+ str(x_origin) + " y=" + str(y_origin) + ")")
            self.reward = -0.1
            return False
    
    def new_piece(self):
        self.piece = jigsaw_piece()
        #self.reward = -0.1 wenn hier dann wird auch erfolg bestraft
    
    def reset(self):
        self.board = np.zeros((4,6), dtype=int)
        self.rows = self.board.shape[0]
        self.cols = self.board.shape[1]
        self.new_piece()
        self.reward = 0
        self.finished = False

    def get_state(self):
        piece_buffer = np.zeros((4,4), dtype=int)
        piece_buffer[0:0+self.piece.rows, 0:0+self.piece.cols] = np.add(piece_buffer[0:0+self.piece.rows, 0:0+self.piece.cols], self.piece.form)
        reward_last_action = self.reward
        state = np.concatenate((self.board, piece_buffer), axis=1, out=None, dtype=int, casting="no")
        print(state)
        self.reward = 0
        finish = False
        if(self.solved()):
            finish = True
            self.reset()
        return [state, reward_last_action, finish]
    
    def action_converter(self, action):
        action_index = int(np.where(action == 1)[0])
        #print(action_index)
        # Move to position
        if(action_index<24):
            row = int(action_index/6)
            col = int(action_index%6)
            self.move(col, row)
        # do something else
        else:
            # get new piece
            if(action_index == 24):
                self.new_piece()
                self.reward = -0.1




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
        
        worked = False
        
        for y in range(0, act_jigsaw_game.rows):
            for x in range(0, act_jigsaw_game.cols):
                if(act_jigsaw_game.move(x, y) == True):
                    worked = True
                    used +=1
                    break
            if worked == True:
                break
        if(worked == False):
            not_used += 1
        act_jigsaw_game.new_piece()
    act_jigsaw_game.action_converter()
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
"""
def main():
    #statistic brute force
    g_turn = 0
    g_used = 0
    g_not_used = 0
    for i in range(0, 1):
        turn, used, not_used = run_brute_force()
        g_turn += turn
        g_used += used
        g_not_used += not_used
    print("Average Brute Force (Turn: " + str(g_turn/1) + " Used: " + str(g_used/1) + " Not used: " + str(g_not_used/1))


main()
"""