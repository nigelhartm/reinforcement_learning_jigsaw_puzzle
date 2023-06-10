# reinforcement_learning_jigsaw_puzzle
In a computer game I liked to play as a kid there exist a minigame called Jigsaw. It is hard to do efficiently. The reward varies dependent on how much steps you need to solve, so it will be interesting to find a solver. It is a tetris like puzzle which can be solved easily using heuristics. For learning purpose I will use reinforcement learning

## Environment
conda create -n jigsaw numpy pytorch -c anaconda -c pytorch
conda activate jigsaw

## Rules
The Field consists of 4 rows and 6 cols and is initialized empty<br>
o o o o o o<br>
o o o o o o<br>
o o o o o o<br>
o o o o o o<br>

There are 6 possible puzzle pieces you can get<br>
1)<br>
x<br>
x x<br>

2)<br>
x x<br>
&ensp;&ensp;x<br>

3)<br>
x x<br>
x x<br>

4)<br>
x<br>
x<br>
x<br>

5)<br>
x x<br>
&ensp;&ensp;x x<br>

6)<br>
x<br>

At every turn you get one of these randomly.<br>
You can place it on a empty position on the field or throw it away and start a new turn.<br>
The game is done if the field is completely filled with forms with nothing empty.<br>


## Brute force
After setting up the basic rules for the game I created a brute force algorithm, which received following statistics on 100 runs:<br>
Average Brute Force (Turn: 32.08 Used: 10.74 Not used: 21.34

## Rewards
To get an idea about if this is bad or good here some information about the rewards:<br>
10 or fewer: Large treasure chest<br>
11 to 24: medium treasure chest<br>
25 or more: small tresure chest<br>

## Help
https://www.toptal.com/deep-learning/pytorch-reinforcement-learning-tutorial

https://pub.towardsai.net/understanding-tensor-dimensions-in-deep-learning-models-with-pytorch-4ee828693826