# TicTacToeAI

Adversarial search: TicTacToe AI Opponent

In this project, various algorithms for adversarial search are implemented. 
4 Search Algorithms are implemented: MinMax, MinMax with Cutoff depth (uses Evaluation function at set depth), AlphaBeta, AlphaBeta with depth cutoff, and MonteCarlo Tree search.

A few notes:
  1) The Evaluation function gives the highest value to the board configuration which has the most potential for a win by the player.
  2) There is a timer and a depth GUI for limiting search time and limit depth search. Timer is used for iterative deepening. So when you set timer, the depth setting is not used in the code. When timer or depth is set to -1 it means no time limit and no depth limit.
  3) You are also able to play 3 x 3, 4 x 4, and 5 x 5 TicTacToe. 


