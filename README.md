# reinforcement_learning_jigsaw_puzzle
In a computer game I liked to play as a kid there exist a minigame called Jigsaw. It is hard to do efficiently. The reward varies dependent on how much steps you need to solve, so it will be interesting to find a solver. It is a tetris like puzzle which can be solved easily using heuristics. For learning purpose I will use reinforcement learning

## Environment
conda create -n jigsaw numpy pytorch -c anaconda -c pytorch
conda activate jigsaw

## Rules
The Field consists of 4 rows and 6 cols and is initialized empty<br>
![alt text](img/game_no_piece.png)

At each turn you can place a piece on the board or draw a random new one frome the card boxes.
![alt text](img/card_box.png)

There are 6 possible puzzle pieces you can get out of the box:<br>
id=0
![alt text](img/pieces/0.png)
id=1
![alt text](img/pieces/1.png)
id=2
![alt text](img/pieces/2.png)
id=3
![alt text](img/pieces/3.png)
id=4
![alt text](img/pieces/4.png)
id=5
![alt text](img/pieces/5.png)

The game is done if the field is completely filled with forms with nothing empty.<br>
![alt text](img/game_finished.png)


## Brute force
After setting up the basic rules for the game I created a brute force algorithm, which received following statistics on 100 runs:<br>
Average Brute Force (Turn: 32.08 Used: 10.74 Not used: 21.34

## Rewards
To get an idea about if this is bad or good here some information about the rewards:<br>
10 or fewer:<br>
Large treasure chest ![alt text](img/treasure_chest.png)
11 to 24:<br>
medium treasure chest ![alt text](img/treasure_chest.png)
25 or more:<br>
small tresure chest ![alt text](img/treasure_chest.png)

## Resources
https://www.toptal.com/deep-learning/pytorch-reinforcement-learning-tutorial
https://pub.towardsai.net/understanding-tensor-dimensions-in-deep-learning-models-with-pytorch-4ee828693826
https://en-wiki.metin2.gameforge.com/index.php/Fishing_Jigsaw