import argparse
import os
import warnings

import numpy as np
import toml
from rgnn.common.registry import registry

from rlsim.drl.simulator import RLSimulator
from rlsim.drl.trainer import Trainer
from rlsim.environment import Environment
from rlsim.memory import Memory
from rlsim.utils.logger import setup_logger

warnings.filterwarnings("ignore", category=UserWarning)


def train_DQN(settings):
    with open(settings, "r") as f:
        config = toml.load(f)
        task = config["task"]
        logger_config = config["logger"]
        model_config = config["model"]
        train_config = config["train"]

    if task not in os.listdir():
        os.makedirs(task, exist_ok=True)
    if "traj" not in os.listdir(task):
        os.mkdir(task + "/traj")
    if "model" not in os.listdir(task):
        os.mkdir(task + "/model")

    with open(task + "/loss.txt", "w") as file:
        file.write("epoch\t loss\n")

    log_filename = f"{task}/{logger_config['filename']}.log"
    logger = setup_logger(logger_config["name"], log_filename)

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
    q_params.update({"temperature": train_config["temperature"]})
    trainer = Trainer(
        model,
        q_params=q_params,
        offline_model=offline_model,
        lr=train_config["lr"],
        train_all=train_config["train_all"],
    )

    replay_list = []
    for epoch in range(n_episodes):

        file = pool[np.random.randint(len(pool))]
        logger.info("epoch = " + str(epoch) + ":  " + file)
        env = Environment(file, calc_params=calc_params)
        env.relax(accuracy=train_config["relax_accuracy"])
        simulator = RLSimulator(environment=env,
                                model=model,
                                model_params=model_config["params"],
                                params=train_config)
        info = simulator.step()
        replay_list.append(
            Memory(q_params["alpha"], q_params["beta"], T=q_params["temperature"])
        )
        for tstep in range(horizon):

            replay_list[-1].add(info)

            if tstep % 10 == 0 and tstep > 0:
                logger.info("    t = " + str(tstep))
        steps = (int(1 + epoch ** (2 / 3)), 5)
        train_config["update_params"].update({"steps": steps})
        loss = trainer.update(memory_l=replay_list, mode=train_mode, **train_config["update_params"])
        logger.info(f"Epoch: {epoch} | Loss: {loss}")
        with open(task + "/loss.txt", "a") as file:
            file.write(str(epoch) + "\t" + str(loss) + "\n")
        try:
            replay_list[epoch].save(task + "/traj/traj" + str(epoch))
        except:
            logger.info("saving failure")

        if epoch % 10 == 0:
            model.save(task + "/model/model" + str(epoch))
        model.save(task + "/model/model_trained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DRL model')
    parser.add_argument('--config', required=True, help='config file path')

    args = parser.parse_args()
    train_DQN(args.config)
