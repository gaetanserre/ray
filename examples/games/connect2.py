import numpy as np
import gym
from gym.spaces import Discrete, Dict, Box
import matplotlib.pyplot as plt

class Connect2(gym.Env):
  nb_loc = 4

  def __init__(self, config=None):
    self.action_space = Discrete(Connect2.nb_loc)
    self.observation_space = Dict(
      {
        "obs": Box(low=-1, high=1, shape=(Connect2.nb_loc+1,)),
        "action_mask": Box(low=0, high=1, shape=(self.action_space.n,)),
      }
    )
    self.running_reward = 0

    self.reset()

  def _get_valid_acts(self):
    return (self.obs[:-1] == 0).astype(np.float32)
  
  def _is_win(self, state, player):
    count = 0
    for index in range(state.shape[0]):
        if state[index] == player:
            count = count + 1
        else:
            count = 0
        if count == 2:
            return True
    return False

  def _get_reward(self, state):
    player = state[-1]
    state = state[:-1]
    if self._is_win(state, player):
      return 1
    elif self._is_win(state, -player):
      return -1
    elif not (state == 0).any():
      return 0
    else:
      return None

  def reset(self):
    self.obs = np.zeros(Connect2.nb_loc + 1)
    self.obs[-1] = 1
    return {
      "obs": self.obs,
      "action_mask": np.ones(Connect2.nb_loc, dtype=np.float32),
    }
  
  def step(self, action):
    player = self.obs[-1]
    self.obs[action] = player
    self.obs[-1] = -player

    rew = self._get_reward(self.obs)
    done = rew is not None

    score = rew if done else 0
    action_mask = self._get_valid_acts()

    return (
        {"obs": self.obs,
        "action_mask": action_mask},
        score,
        done,
        {},
    )

  def set_state(self, state):
    self.obs = state.copy()
    action_mask = self._get_valid_acts()
    return {"obs": self.obs, "action_mask": action_mask}

  def get_state(self):
    return self.obs.copy()
  
  def _colorize_state(self, state):
    state = state[:-1]
    color = np.zeros((1, Connect2.nb_loc, 3))
    color[:, state == 1] = [1, 0, 0]
    color[:, state == -1] = [1, 1, 0]
    return color
  
  def render(self):
    plt.imshow(self._colorize_state(self.obs))
    plt.axis('off')
    plt.show()