from ray import tune
import ray
from ray.rllib.models.catalog import ModelCatalog

from parameters import parameters

ray.init(num_cpus=parameters["num_cpus"], num_gpus=parameters["num_gpus"])
ModelCatalog.register_custom_model("model", parameters["model"])

analysis = tune.run(
    "contrib/AlphaZero",
    local_dir=parameters["local_dir"],
    name=parameters["name"],
    stop={"training_iteration": parameters["nb_iterations"]},
    max_failures=0,
    config=parameters["config"],
    checkpoint_freq=parameters["checkpoint_freq"],
    checkpoint_at_end = True
)