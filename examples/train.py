from ray import tune
import ray
from ray.rllib.models.catalog import ModelCatalog
from ray.tune.logger import JsonLoggerCallback, CSVLoggerCallback, TBXLoggerCallback
from ray.tune.callback import Callback
import numpy as np
import matplotlib.pyplot as plt
from parameters import parameters

class PlotCallback(Callback):
  def __init__(self):
    self.episode_mean_rewards = []
  
  def on_trial_result(self, iteration: int, trials, trial, result, **info):
    self.episode_mean_rewards.append(result["episode_reward_mean"])
  
  def save_plot(
              self,
              expected_reward: float,
              title: str,
              filename: str
              ):
    x = np.arange(0, len(self.episode_mean_rewards), 1)
    plt.figure(figsize=(8, 6))
    plt.plot(x, self.episode_mean_rewards, label="Episode mean reward")
    plt.axhline(y=expected_reward, color="black", linestyle="--", label="Expected reward")
    plt.title(title)
    plt.xlabel("Training iteration")
    plt.savefig(filename)

ray.init(num_cpus=parameters["num_cpus"], num_gpus=parameters["num_gpus"])
ModelCatalog.register_custom_model("model", parameters["model"])
plot_callback = PlotCallback()

analysis = tune.run(
    "contrib/AlphaZero",
    local_dir=parameters["local_dir"],
    name=parameters["name"],
    stop={"training_iteration": parameters["nb_iterations"]},
    max_failures=0,
    config=parameters["config"],
    checkpoint_freq=parameters["checkpoint_freq"],
    checkpoint_at_end = True,
    callbacks=[
            plot_callback,
            JsonLoggerCallback(),
            CSVLoggerCallback(),
            TBXLoggerCallback()
            ]
)

plot_callback.save_plot(
                      expected_reward=0,
                      title="Episode reward on Tic Tac Toe",
                      filename="imgs/TicTacToe.pdf"
                      )