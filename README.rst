**A fork of the Ray framework that aims to implements two-players AlphaZero algorithm**

In ray, by default, the alpha-zero algorithm is for one-play game.
Now, you can specify if your game is made for one or two players.
You just have to set the mcts parameter *is_two_players* to *True*
Example::
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


.. image:: https://github.com/ray-project/ray/raw/master/doc/source/images/ray_header_logo.png

**Ray provides a simple, universal API for building distributed applications.**

Ray is packaged with the following libraries for accelerating machine learning workloads:

- `Tune`_: Scalable Hyperparameter Tuning
- `RLlib`_: Scalable Reinforcement Learning
- `Train`_: Distributed Deep Learning (beta)
- `Datasets`_: Distributed Data Loading and Compute

As well as libraries for taking ML and distributed apps to production:

- `Serve`_: Scalable and Programmable Serving
- `Workflows`_: Fast, Durable Application Flows (alpha)

There are also many `community integrations <https://docs.ray.io/en/master/ray-libraries.html>`_ with Ray, including `Dask`_, `MARS`_, `Modin`_, `Horovod`_, `Hugging Face`_, `Scikit-learn`_, and others. Check out the `full list of Ray distributed libraries here <https://docs.ray.io/en/master/ray-libraries.html>`_.
