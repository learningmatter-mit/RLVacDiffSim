import os
import random
import warnings

import numpy as np
import toml
from rgnn.common.registry import registry

from rlsim.drl.simulator import RLSimulator
from rlsim.drl.trainer import Trainer
from rlsim.environment import Environment
from rlsim.memory import Memory

warnings.filterwarnings("ignore", category=UserWarning)


def train_DQN(task, logger, config):
    logger.info(f"Training DQN model in: {os.path.realpath(task)}")
    toml.dump(config, open(f"{task}/config_copied.toml", "w"))
    model_config = config["model"]
    train_config = config["train"]

    if "traj" not in os.listdir(task):
        os.mkdir(task + "/traj")
    if "model" not in os.listdir(task):
        os.mkdir(task + "/model")

    with open(task + "/loss.txt", "w") as file:
        file.write("epoch\t loss\n")

    calc_params = train_config.pop("calc_info")
    calc_params.update({"relax_log": f"{task}/{calc_params['relax_log']}"})
    n_episodes = train_config.pop("n_episodes")
    horizon = train_config.pop("horizon")
    train_mode = train_config.pop("mode")

    # new_pool = []
    poscar_dir = train_config.pop("poscar_dir")
    n_poscars = train_config.pop("n_poscars")
    pool = []
    for directory, n_files in zip(poscar_dir, n_poscars):
        pool += [f"{directory}/POSCAR_" + str(i) for i in range(0, n_files)]

    gcnn = registry.get_reaction_model_class(model_config["reaction_model"]["@name"]).load(model_config["reaction_model"]["model_path"])

    model = registry.get_model_class(model_config["@name"])(gcnn, N_emb=model_config["n_emb"], N_feat=model_config["n_feat"], canonical=True)
    if train_mode == "dqn":
        offline_model = registry.get_model_class(model_config["@name"])(gcnn, N_emb=model_config["n_emb"], N_feat=model_config["n_feat"], canonical=True)
    else:
        offline_model = None
    q_params = model_config["params"]
    temperature = random.choice(train_config["temperature"])
    q_params.update({"temperature": temperature})
    trainer = Trainer(
        model,
        q_params=q_params,
        offline_model=offline_model,
        lr=train_config["lr"],
        train_all=train_config["train_all"],
    )

    replay_list = []
    for epoch in range(n_episodes):
        logger.info(f"Episode : {epoch}")
        atoms = pool[np.random.randint(len(pool))]
        env = Environment(atoms, calc_params=calc_params)
        env.relax()
        simulator = RLSimulator(environment=env,
                                model=model,
                                model_params=model_config["params"],
                                params=train_config)
        replay_list.append(
            Memory(q_params["alpha"], q_params["beta"], T=q_params["temperature"])
        )
        for tstep in range(horizon):
            info = simulator.step(q_params["temperature"])
            replay_list[-1].add(info)
            logger.info(f"  tstep : {tstep}")
        train_config["update_params"].update({"episode_size": int(1 + epoch ** (2 / 3))})
        loss = trainer.update(memory_l=replay_list, mode=train_mode, **train_config["update_params"])
        with open(task + "/loss.txt", "a") as file:
            file.write(str(epoch) + "\t" + str(loss) + "\n")
            logger.info(f"   Loss : {loss:.3f}")
        try:
            replay_list[epoch].save(task + "/traj/traj" + str(epoch))
        except:
            logger.info("saving failure")

        if epoch % 10 == 0:
            model.save(task + "/model/model" + str(epoch))
        model.save(task + "/model/model_trained")

