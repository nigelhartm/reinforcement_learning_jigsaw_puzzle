from tkinter import *
from PIL import ImageTk, Image #png
import threading
import time
import numpy as np
import random

# Global initialization
#
SCALE = 32
WIDTH = 6
HEIGHT = 4
TIMER = 0.01
CHEST_S = 0
CHEST_M = 0
CHEST_L = 0
TEXT_CHEST_S = None
TEXT_CHEST_M = None
TEXT_CHEST_L = None
ROUNDS = 0
USED_TILES = 0
TEXT_ROUNDS = None
TEXT_USED_TILES = None
TEXT_MEAN_PER_ROUND = None
canvas_tile_tiles = None
img = []
canvas_tile = None
canvas_tile_act = None
canvas_stats = None
canvas_stats2 = None
canvas_board = None
canvas_board_tiles = []

# # # # # # # # # # # # # # # # # # #
# START GAME
# # # # # # # # # # # # # # # # # # #
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

class jigsaw_piece:
	idx = None
	form = None
	color = None
	rows = None
	colsw = None
	# Constructor
	def __init__(self):
		global canvas_tile
		global canvas_tile_act
		global img
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
		if canvas_tile_act != None:
			canvas_tile.delete(canvas_tile_act)
		canvas_tile_act = canvas_tile.create_image(0*SCALE, 0*SCALE, anchor=NW, image=img[piece_number])
		self.idx = piece_number

def run_brute_force():
	global ROUNDS
	global CHEST_L
	global CHEST_M
	global CHEST_S
	global USED_TILES
	global canvas_board_tiles
	global canvas_board
	global img
	act_jigsaw_game = jigsaw_game()
	ROUNDS += 1
	local_tiles = 0
	while(act_jigsaw_game.solved() == False):
		time.sleep(TIMER)
		act_jigsaw_piece = jigsaw_piece()
		USED_TILES += 1
		local_tiles += 1
		actualizeChests()
		actualizeStats2()
		worked = False
		for y in range(0, act_jigsaw_game.rows):
			for x in range(0, act_jigsaw_game.cols):
				if(act_jigsaw_game.move(act_jigsaw_piece, x, y) == True):
					worked = True
					canvas_board_tiles.append(canvas_board.create_image(x*SCALE, y*SCALE, anchor=NW, image=img[act_jigsaw_piece.idx]))
					break
			if worked == True:
				break
	if local_tiles <= 10:
		CHEST_L += 1
	elif local_tiles <= 24:
		CHEST_M += 1
	else:
		CHEST_S += 1
	return
# # # # # # # # # # # # # # # # # # #
# END GAME
# # # # # # # # # # # # # # # # # # #

# Threads
#
def game():
	global TIMER
	global canvas_board
	global canvas_board_tiles
	while True:
		time.sleep(TIMER)
		run_brute_force()
		for tile in canvas_board_tiles:
			canvas_board.delete(tile)
		

def actualizeChests():
	global canvas_stats
	S = ("00000"+str(CHEST_S))
	S = S[len(S)-5:len(S)]
	M = ("00000"+str(CHEST_M))
	M = M[len(M)-5:len(M)]
	L = ("00000"+str(CHEST_L))
	L = L[len(L)-5:len(L)]
	canvas_stats.itemconfig(TEXT_CHEST_S, text="S:"+S)
	canvas_stats.itemconfig(TEXT_CHEST_M, text="M:"+M)
	canvas_stats.itemconfig(TEXT_CHEST_L, text="L:"+L)

def actualizeStats2():
	global canvas_stats2
	R = ("00000"+str(ROUNDS))
	R = R[len(R)-5:len(R)]
	T = ("00000"+str(USED_TILES))
	T = T[len(T)-5:len(T)]
	MEAN = ("00000"+str(int(USED_TILES/ROUNDS if ROUNDS else 0)))
	MEAN = MEAN[len(MEAN)-5:len(MEAN)]
	canvas_stats2.itemconfig(TEXT_ROUNDS, text="R:"+R)
	canvas_stats2.itemconfig(TEXT_USED_TILES, text="T:"+T)
	canvas_stats2.itemconfig(TEXT_MEAN_PER_ROUND, text="\u2300:"+MEAN)

# Create new window
#
root = Tk(className='Fishing Jigsaw - AI')
root.configure(background='#000000')
root.geometry(str(13*SCALE) + "x" + str(10*SCALE))

# Create Board canvas
#
canvas_board = Canvas(root, bg="#000000", height=HEIGHT*SCALE+1, width=WIDTH*SCALE+1) # +1 for border
canvas_board.place(x=1*SCALE, y=1*SCALE, anchor=NW)
canvas_board.config(highlightthickness=0)

# Print grid
#
for y in range(0, HEIGHT+1):
	canvas_board.create_line(0*SCALE,y*SCALE,WIDTH*SCALE,y*SCALE, fill="#FFFFFF", width=1)
for x in range(0, WIDTH+1):
	canvas_board.create_line(x*SCALE,0*SCALE,x*SCALE,HEIGHT*SCALE, fill="#FFFFFF", width=1)

# Create actual tile canvas
#
canvas_tile = Canvas(root, bg="#000000", height=HEIGHT*SCALE+1, width=HEIGHT*SCALE+1) # +1 for border
canvas_tile.place(x=8*SCALE, y=1*SCALE, anchor=NW)
canvas_tile.config(highlightthickness=0)

# Print grid
#
for y in range(0, HEIGHT+1):
	canvas_tile.create_line(0*SCALE,y*SCALE,WIDTH*SCALE,y*SCALE, fill="#FFFFFF", width=1)
for x in range(0, HEIGHT+1):
	canvas_tile.create_line(x*SCALE,0*SCALE,x*SCALE,HEIGHT*SCALE, fill="#FFFFFF", width=1)

# Create stats canvas
#
canvas_stats = Canvas(root, bg="#000000", height=1*SCALE+1, width=11*SCALE+1) # +1 for border
canvas_stats.place(x=1*SCALE, y=6*SCALE, anchor=NW)
canvas_stats.config(highlightthickness=0)
# horizontal
canvas_stats.create_line(0*SCALE,0*SCALE,11*SCALE,0*SCALE, fill="#FFFFFF", width=1)
canvas_stats.create_line(0*SCALE,1*SCALE,11*SCALE,1*SCALE, fill="#FFFFFF", width=1)
# vertical
canvas_stats.create_line(0*SCALE,0*SCALE,0*SCALE,1*SCALE, fill="#FFFFFF", width=1)
canvas_stats.create_line(11*SCALE,0*SCALE,11*SCALE,1*SCALE, fill="#FFFFFF", width=1)
TEXT_CHEST_S = canvas_stats.create_text(2+0*SCALE,1, text="L:00000", fill="#ffffff", font=('Helvetica','24'), anchor=NW)
TEXT_CHEST_M = canvas_stats.create_text(2+3.75*SCALE,1, text="M:00000", fill="#ffffff", font=('Helvetica','24'), anchor=NW)
TEXT_CHEST_L = canvas_stats.create_text(2+7.5*SCALE,1, text="S:00000", fill="#ffffff", font=('Helvetica','24'), anchor=NW)


# Create stats2 canvas
#
canvas_stats2 = Canvas(root, bg="#000000", height=1*SCALE+1, width=11*SCALE+1) # +1 for border
canvas_stats2.place(x=1*SCALE, y=8*SCALE, anchor=NW)
canvas_stats2.config(highlightthickness=0)
# horizontal
canvas_stats2.create_line(0*SCALE,0*SCALE,11*SCALE,0*SCALE, fill="#FFFFFF", width=1)
canvas_stats2.create_line(0*SCALE,1*SCALE,11*SCALE,1*SCALE, fill="#FFFFFF", width=1)
# vertical
canvas_stats2.create_line(0*SCALE,0*SCALE,0*SCALE,1*SCALE, fill="#FFFFFF", width=1)
canvas_stats2.create_line(11*SCALE,0*SCALE,11*SCALE,1*SCALE, fill="#FFFFFF", width=1)
TEXT_ROUNDS = canvas_stats2.create_text(2+0*SCALE,1, text="R:00000", fill="#ffffff", font=('Helvetica','24'), anchor=NW)
TEXT_USED_TILES = canvas_stats2.create_text(2+3.75*SCALE,1, text="T:00000", fill="#ffffff", font=('Helvetica','24'), anchor=NW)
TEXT_MEAN_PER_ROUND = canvas_stats2.create_text(2+7.5*SCALE,1, text="\u2300:00000", fill="#ffffff", font=('Helvetica','24'), anchor=NW)


# Load and SCALE Images
#
img.append(ImageTk.PhotoImage(Image.open('pieces/0.png').resize((2*SCALE, 2*SCALE), Image.ANTIALIAS)))
img.append(ImageTk.PhotoImage(Image.open('pieces/1.png').resize((2*SCALE, 2*SCALE), Image.ANTIALIAS)))
img.append(ImageTk.PhotoImage(Image.open('pieces/2.png').resize((2*SCALE, 2*SCALE), Image.ANTIALIAS)))
img.append(ImageTk.PhotoImage(Image.open('pieces/3.png').resize((1*SCALE, 3*SCALE), Image.ANTIALIAS)))
img.append(ImageTk.PhotoImage(Image.open('pieces/4.png').resize((3*SCALE, 2*SCALE), Image.ANTIALIAS)))
img.append(ImageTk.PhotoImage(Image.open('pieces/5.png').resize((1*SCALE, 1*SCALE), Image.ANTIALIAS)))

# Game
#canvas_board_tiles = []
#canvas_board_tiles.append(canvas_board.create_image(0*SCALE, 0*SCALE, anchor=NW, image=img0))
#canvas_board_tiles.append(canvas_board.create_image(1*SCALE, 0*SCALE, anchor=NW, image=img1))
#canvas_board_tiles.append(canvas_board.create_image(3*SCALE, 1*SCALE, anchor=NW, image=img3))

# start thread
#
t1 = threading.Thread(target=game, args=[])
t1.start()

# Window loop
#
root.mainloop()
