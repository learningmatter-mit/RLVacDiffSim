import os
from itertools import product

import click
import toml
from ase import io

from rlsim.drl.deploy import deploy_RL
from rlsim.utils.logger import setup_logger


def get_atoms_list(folder, pool_ind):
    atoms_list = []
    traj = pool_ind[0]
    frame = pool_ind[1]
    traj = io.read(f"{folder}/XDATCAR{traj}", index=f":{frame}")
    for atoms in traj:
        atoms_list.append(atoms)
    return atoms_list


@click.command()
@click.option('-f', '--folder', required=True, help='Trajectory path')
@click.option('-c', '--config_name', required=True, help='Config file path')
@click.option('-s', '--step', type=int, default=None, help='Number of steps to extract from XDATCAR')
@click.option('-n', '--n-traj', type=int, default=None, help='Number of XDATCAR files to process')
def main(folder, config_name, step, n_traj):
    """
    CLI for deploying DRL models using DQN trajectory data.
    """
    if step and n_traj:
        print("Collecting trajectory...")
        atoms_traj = []
        # pool = [(j, k) for j, k in product(range(n_traj), step)]
        for traj_index in range(n_traj):
            atoms_traj.extend(get_atoms_list(folder, [traj_index, step]))
    else:
        atoms_traj = io.read(folder, index=':')
    with open(config_name, "r") as f:
        config = toml.load(f)
        task = config.pop("task")
        logger_config = config.pop("logger")
    if task not in os.listdir():
        os.makedirs(task, exist_ok=True)
    log_filename = f"{task}/{logger_config['filename']}.log"
    logger = setup_logger(logger_config["name"], log_filename)

    deploy_RL(task, logger, config, atoms_traj=atoms_traj)
