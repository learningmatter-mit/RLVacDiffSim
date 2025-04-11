import argparse
import os
import warnings

import click
import numpy as np
import toml

from rlsim.drl.simulator import RLSimulator
from rlsim.drl.trainer import Trainer
from rlsim.environment import Environment
from rlsim.memory import Memory
from rlsim.utils.logger import setup_logger

warnings.filterwarnings("ignore", category=UserWarning)


def gen_pretrain_data(settings):
    with open(settings, "r") as f:
        config = toml.load(f)
        task = config["task"]
        logger_config = config["logger"]
        simulation_config = config["simulation"]

    if task not in os.listdir():
        os.makedirs(task, exist_ok=True)
    if "traj" not in os.listdir(task):
        os.mkdir(task + "/traj")

    log_filename = f"{task}/{logger_config['filename']}.log"
    logger = setup_logger(logger_config["name"], log_filename)

    calc_params = simulation_config.pop("calc_info")
    calc_params.update({"relax_log": f"{task}/{calc_params['relax_log']}"})
    n_episodes = simulation_config.pop("n_episodes")
    horizon = simulation_config.pop("horizon")
    poscar_dir = simulation_config.pop("poscar_dir")
    n_poscars = simulation_config.pop("n_poscars")
    pool = []
    for directory, n_files in zip(poscar_dir, n_poscars):
        pool += [f"{directory}/POSCAR_" + str(i) for i in range(0, n_files)]
    replay_list = []

    for epoch in range(n_episodes):

        file = pool[np.random.randint(len(pool))]
        logger.info("epoch = " + str(epoch) + ":  " + file)
        env = Environment(file, calc_params=calc_params)
        env.relax()
        simulator = RLSimulator(environment=env)
        replay_list.append(
            Memory(1.0, 0.0) # 1.0 and 0.0 is random
        )
        for tstep in range(horizon):
            info = simulator.step(random=True)
            replay_list[-1].add(info)
            logger.info("    t = " + str(tstep))
        try:
            replay_list[epoch].save(task + "/traj/traj" + str(epoch))
        except:
            logger.info("saving failure")
            pass


@click.command()
@click.option("-c", "--config", required=True, help="config file path")
def main(config):
    """
    Generate pretraining data for the time estimator model
    """
    gen_pretrain_data(config)
