import numpy as np
import gym
from gym.spaces import Discrete, Dict, Box
import matplotlib.pyplot as plt

class Connect4(gym.Env):
  ROWS = 6
  COLUMNS = 7
  nb_loc = COLUMNS
  shape = ROWS * COLUMNS + 1

  def __init__(self, config=None):
    self.action_space = Discrete(Connect4.nb_loc)
    self.observation_space = Dict(
      {
        "obs": Box(low=-1, high=1, shape=(Connect4.shape,)),
        "action_mask": Box(low=0, high=1, shape=(self.action_space.n,)),
      }
    )
    self.running_reward = 0

    self.reset()
  
  def _get_valid_acts(self):
    return (self.obs[0 ,:] == 0).astype(np.float32)

  def _is_win(self, state, player):
    # Check horizontal locations for win
    for c in range(Connect4.COLUMNS-3):
      for r in range(Connect4.ROWS):
        if state[r][c] == player and state[r][c+1] == player and state[r][c+2] == player and state[r][c+3] == player:
          return True

    # Check vertical locations for win
    for c in range(Connect4.COLUMNS):
      for r in range(Connect4.ROWS-3):
        if state[r][c] == player and state[r+1][c] == player and state[r+2][c] == player and state[r+3][c] == player:
          return True

    # Check positively sloped diagonals
    for c in range(Connect4.COLUMNS-3):
      for r in range(Connect4.ROWS-3):
        if state[r][c] == player and state[r+1][c+1] == player and state[r+2][c+2] == player and state[r+3][c+3] == player:
          return True

    # Check negatively sloped diagonals
    for c in range(Connect4.COLUMNS-3):
      for r in range(3, Connect4.ROWS):
        if state[r][c] == player and state[r-1][c+1] == player and state[r-2][c+2] == player and state[r-3][c+3] == player:
          return True
    return False

  def _get_reward(self, state):
    player = state[-1, 0]
    state = state[:-1, :]
    if self._is_win(state, player):
      return 1
    elif self._is_win(state, -player):
      return -1
    elif not (state[0, :] == 0).any():
      return 0
    else:
      return None

  def reset(self):
    # We encode the player on an additional line
    self.obs = np.zeros((Connect4.ROWS + 1, Connect4.COLUMNS))
    self.obs[-1, :] = 1
    return {
      "obs": self.obs.flatten()[:Connect4.shape],
      "action_mask": np.ones(Connect4.nb_loc, dtype=np.float32),
    }
  
  def step(self, action):
    player = self.obs[-1, 0]

    row = np.argmax(np.argwhere(self.obs[:, action] == 0))
    self.obs[row, action] = player
    self.obs[-1, :] = -player

    rew = self._get_reward(self.obs)
    done = rew is not None

    score = rew if done else 0
    action_mask = self._get_valid_acts()

    return (
        {"obs": self.obs.flatten()[:Connect4.shape],
        "action_mask": action_mask},
        score,
        done,
        {},
    )

  def set_state(self, state):
    self.obs = state.copy()
    action_mask = self._get_valid_acts()
    return {"obs": self.obs.flatten()[:Connect4.shape], "action_mask": action_mask}

  def get_state(self):
    return self.obs.copy()
  
  def _colorize_state(self, state):
    state = state[:-1, :]
    color = np.zeros((Connect4.ROWS, Connect4.COLUMNS, 3))
    color[state == 1] = [1, 0, 0]
    color[state == -1] = [1, 1, 0]
    return color
  
  def render(self):
    plt.imshow(self._colorize_state(self.obs))
    plt.axis('off')
    plt.show()