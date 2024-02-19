import numpy as np
import random
import torch
from math import inf

class jigsaw_game:
    board = None
    rows = 4
    cols = 6
    piece = None
    PIECESQUARE = 4
    step = 0
    reward = 0

    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((self.rows,self.cols), dtype=int)
        self.new_piece()
        self.reward = 0
        self.step = 0

    def solved(self):
        if(np.array_equal(self.board, np.ones((self.rows, self.cols), dtype=int))):
            if self.step <= 10:
                self.reward = 100
            else:
                if self.step <= 24:
                    self.reward = 10
                else:
                    self.reward = -10
            return True
        else:
            if self.step >= 30:
                self.reward = -50
                return True
            return False

    def valid_move(self, x_origin, y_origin):
        board_piece_area = self.board[y_origin:y_origin+self.piece.rows, x_origin:x_origin+self.piece.cols]
        #check out of area
        if(board_piece_area.shape[0] != self.piece.rows or board_piece_area.shape[1] != self.piece.cols):
          return False
        #check in board valid
        if(np.max(np.add(board_piece_area, self.piece.form)) == 1):
            return True
        else:
            return False
    
    def getMask(self):
        ret = np.zeros((25), dtype=float)
        for i in range(0, 24):
            row = int(i / self.cols)
            col = int(i % self.cols)
            if(self.valid_move(col, row)):
                ret[i] = 0
            else:
                ret[i] = inf
        ret[24] = 0
        return torch.from_numpy(ret)

    def move(self, x_origin, y_origin):
        if(self.valid_move(x_origin, y_origin)):
            self.board[y_origin:y_origin+self.piece.rows, x_origin:x_origin+self.piece.cols] = np.add(self.board[y_origin:y_origin+self.piece.rows, x_origin:x_origin+self.piece.cols], self.piece.form)
            return True
        else:
            return False
    
    def new_piece(self):
        self.piece = jigsaw_piece()

    def get_state(self, action):
        is_moved = False
        action = int(np.where(action == 1)[0])
        if(action < self.rows * self.cols):
            row = int(action / self.cols)
            col = int(action % self.cols)
            is_moved = self.move(col, row)
            if is_moved:
                self.step +=  +1
                self.new_piece()
            else:
                exit("ERROR action not allowed")
        else:
            if(action == self.rows * self.cols):
                self.new_piece()
                self.step += 1
        piece_buffer = np.zeros((self.PIECESQUARE, self.PIECESQUARE), dtype=int)
        piece_buffer[0:0+self.piece.rows, 0:0+self.piece.cols] = np.add(piece_buffer[0:0+self.piece.rows, 0:0+self.piece.cols], self.piece.form)
        state = np.concatenate((self.board, piece_buffer), axis=1, out=None, dtype=int, casting="no")
        finish = self.solved()
        reward_last_action = self.reward
        stepp = self.step
        if(finish):
            self.reset()
        return [state, reward_last_action, finish, stepp]

class jigsaw_piece:
    id = None
    form = None
    color = None
    rows = None
    cols = None

    def __init__(self):
        piece_number = random.randint(0, 5)
        self.id = piece_number
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
