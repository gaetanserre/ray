import numpy as np
import gym
from gym.spaces import Discrete, Dict, Box
import matplotlib.pyplot as plt

class TicTacToe(gym.Env):
  ROWS = 3
  COLUMNS = 3
  nb_loc = COLUMNS * ROWS

  def __init__(self, config=None):
    self.action_space = Discrete(TicTacToe.nb_loc)
    self.observation_space = Dict(
      {
        "obs": Box(low=-1, high=1, shape=(TicTacToe.nb_loc+1,)),
        "action_mask": Box(low=0, high=1, shape=(self.action_space.n,)),
      }
    )
    self.running_reward = 0

    self.reset()
  
  def _is_win(self, state, player):
    # Check horizontal locations for win
    for c in range(TicTacToe.COLUMNS-2):
      for r in range(TicTacToe.ROWS):
        if state[r][c] == player and state[r][c+1] == player and state[r][c+2] == player:
          return True

    # Check vertical locations for win
    for c in range(TicTacToe.COLUMNS):
      for r in range(TicTacToe.ROWS-2):
        if state[r][c] == player and state[r+1][c] == player and state[r+2][c] == player:
          return True

    # Check positively sloped diagonals
    for c in range(TicTacToe.COLUMNS-2):
      for r in range(TicTacToe.ROWS-2):
        if state[r][c] == player and state[r+1][c+1] == player and state[r+2][c+2] == player:
          return True

    # Check negatively sloped diagonals
    for c in range(TicTacToe.COLUMNS-2):
      for r in range(2, TicTacToe.ROWS):
        if state[r][c] == player and state[r-1][c+1] == player and state[r-2][c+2] == player:
          return True
    return False

  def _get_reward(self, state):
    player = state[-1]
    state = state[:-1].reshape(TicTacToe.ROWS, TicTacToe.COLUMNS)
    if self._is_win(state, player):
      return 1
    elif self._is_win(state, -player):
      return -1
    elif not (state == 0).any():
      return 0
    else:
      return None

  def reset(self):
    self.obs = np.zeros(TicTacToe.nb_loc + 1)
    self.obs[-1] = 1
    return {
      "obs": self.obs,
      "action_mask": np.ones(TicTacToe.nb_loc, dtype=np.float32),
    }
  
  def step(self, action):
    player = self.obs[-1]
    self.obs[action] = player
    self.obs[-1] = -player

    rew = self._get_reward(self.obs)
    done = rew is not None

    score = rew if done else 0
    action_mask = (self.obs[:-1] == 0).astype(np.float32)

    return (
        {"obs": self.obs,
        "action_mask": action_mask},
        score,
        done,
        {},
    )

  def set_state(self, state):
    self.obs = state.copy()
    action_mask = (self.obs[:-1] == 0).astype(np.float32)
    return {"obs": self.obs, "action_mask": action_mask}

  def get_state(self):
    return self.obs.copy()
  
  def _colorize_state(self, state):
    state = state[:-1].reshape(TicTacToe.ROWS, TicTacToe.COLUMNS)
    color = np.zeros((TicTacToe.ROWS, TicTacToe.COLUMNS, 3))
    color[state == 1] = [1, 0, 0]
    color[state == -1] = [1, 1, 0]
    return color
  
  def render(self):
    plt.imshow(self._colorize_state(self.obs))
    plt.axis('off')
    plt.show()