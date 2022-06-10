from games import TicTacToe, Connect2, Connect4
from ray.rllib.contrib.alpha_zero.models.custom_torch_models import DenseModel

parameters = {}
parameters["num_cpus"] = 8
parameters["num_gpus"] = 1

parameters["env"] = Connect4
parameters["model"] = DenseModel
parameters["config"] = {
                        "env": parameters["env"],
                        "rollout_fragment_length": 128,
                        "train_batch_size": 1024,
                        "sgd_minibatch_size": 128,
                        "lr": 1e-4,
                        "num_sgd_iter": 30,
                        "mcts_config": {
                            "puct_coefficient": 1.5,
                            "num_simulations": 3000,
                            "temperature": 1.5,
                            "dirichlet_epsilon": 0.25,
                            "dirichlet_noise": 0.03,
                            "argmax_tree_policy": False,
                            "add_dirichlet_noise": True,
                            "is_two_players": True
                        },
                        "ranked_rewards": {
                            "enable": False,
                        },
                        "model": {
                            "custom_model": "model",
                        }
                      }
parameters["nb_iterations"] = 150
parameters["local_dir"] = "exps"
parameters["name"] = "AlphaZero_Connect4"
parameters["checkpoint_freq"] = 10