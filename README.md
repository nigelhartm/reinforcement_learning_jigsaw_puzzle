# reinforcement_learning_jigsaw_puzzle
In a computer game I liked to play as a kid there exist a minigame called Jigsaw. It is hard to do efficiently. The reward varies dependent on how much steps you need to solve, so it will be interesting to find a solver. It is a tetris like puzzle which can be solved easily using heuristics. For learning purpose I will use reinforcement learning

## Rules
The Field consists of 4 rows and 6 cols and is initialized empty
o o o o o o
o o o o o o
o o o o o o
o o o o o o

There are 6 possible puzzle pieces you can get
1)
x
x x

2)
x x
  x

3)
x x
x x

4)
x
x
x

5)
x x
  x x

6)
x

At every turn you get one of these randomly.
You can place it on a empty position on the field or throw it away and start a new turn.
The game is done if the field is completely filled with forms with nothing empty.


## Brute force
After setting up the basic rules for the game I created a brute force algorithm, which received following statistics on 100 runs:
Average Brute Force (Turn: 32.08 Used: 10.74 Not used: 21.34

## Rewards
To get an idea about if this is bad or good here some information about the rewards:
10 or fewer: Large treasure chest
11 to 24: medium treasure chest
25 or more: small tresure chest