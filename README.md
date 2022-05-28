## A fork of the Ray framework that aims to implements two-players AlphaZero algorithm

In ray, by default, the alpha-zero algorithm is for one-player game.
Now, you can specify if your game is made for one or two players.
You just have to set the `mcts` parameter *is_two_players* to *True*

Example:
```json
    # === MCTS ===
    "mcts_config": {
        "puct_coefficient": 1.0,
        "num_simulations": 30,
        "temperature": 1.5,
        "dirichlet_epsilon": 0.25,
        "dirichlet_noise": 0.03,
        "argmax_tree_policy": False,
        "add_dirichlet_noise": True,
        "is_two_players": True,
    }
```
## Example games
Some games along with their trained agent are implemented in the `examples` directory.
- `Connect2` (To win, connect two tokens of your color on a board made of 1 row and 4 columns)
- `Tic Tac Toe`
- `Connect4`
  
For `Connect2` and `TicTacToe`, their trained agent plays perfectly.
Regarding `Connect4`, the agent plays very well but struggles to find the perfect move at the very beginning of the game.

![](https://github.com/ray-project/ray/raw/master/doc/source/images/ray_header_logo.png)
