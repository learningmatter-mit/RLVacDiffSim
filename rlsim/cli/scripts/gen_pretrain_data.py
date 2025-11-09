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

    train_labels = simulation_config["train_labels"]
    if "barrier" in train_labels:
        q_params = {"alpha": 1.0, "beta": 1.0}
    else:
        q_params = {"alpha": 0.0, "beta": 0.0}
    calc_params = simulation_config.pop("calc_info")
    calc_params.update({"relax_log": f"{task}/{calc_params['relax_log']}"})
    n_episodes = simulation_config.pop("n_episodes")
    horizon = simulation_config.pop("horizon")
    if simulation_config.get("atoms_list", None) is not None:
        atoms_list = simulation_config.pop("atoms_list")
        pool = atoms_list
    else:
        poscar_dir = simulation_config.pop("poscar_dir")
        n_poscars = simulation_config.pop("n_poscars")
        pool = []
        for directory, n_files in zip(poscar_dir, n_poscars):
            pool += [f"{directory}/POSCAR_" + str(i) for i in range(0, n_files)]
    replay_list = []

    for epoch in range(n_episodes):

        atoms_file = pool[np.random.randint(len(pool))]
        logger.info("epoch = " + str(epoch) + ":  " + atoms_file)
        env = Environment(atoms_file, calc_params=calc_params)
        env.relax()
        simulator = RLSimulator(environment=env, q_params=q_params)
        replay_list.append(
            Memory(q_params["alpha"], q_params["beta"]) # 1.0 and 0.0 is random
        )
        for tstep in range(horizon):
            info = simulator.step(random=True)
            replay_list[-1].add(info)
            logger.info("Step = " + str(tstep) + " | " + f"E_s: {info['E_s']:.3f}, E_min: {info['E_min']:.3f}, E_next: {info['E_next']:.3f}, freq: {info['log_freq']:.3f}, fail: {bool(info['fail'])}")
        try:
            replay_list[epoch].save(task + "/traj/traj" + str(epoch))
        except:
            logger.info("saving failure")
            pass
    logger.info(f"Generated {len(replay_list)} trajectories")


@click.command()
@click.option("-c", "--config", required=True, help="config file path")
def main(config):
    """
    Generate pretraining data for the reaction model training
    Example:
    rlsim-gen_pretrain_data -c '/path/to/config'
    It will generate reaction labels for the reaction model training
    """
    gen_pretrain_data(config)
