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
    frame0 = pool_ind[1]
    frame1 = pool_ind[2]
    traj = io.read(f"{folder}/XDATCAR{traj}", index=f"{frame0}:{frame1}")
    for atoms in traj:
        atoms_list.append(atoms)
    return atoms_list


@click.command()
@click.option('-f', '--folder', required=True, help='Trajectory path')
@click.option('-c', '--config_name', required=True, help='Config file path')
@click.option('-s', '--steps', type=int, multiple=True, default=None, help='Range of number of steps to extract from XDATCAR')
@click.option('-n', '--n-traj', type=int, default=None, help='Number of XDATCAR files to process')
def main(folder, config_name, steps, n_traj):
    """
    CLI for deploying DRL models using DQN trajectory data.
    """
    if len(steps) > 2:
        raise click.BadParameter("You can specify at most two values for --steps.")
    if len(steps) == 1:
        step0 = 0
        step1 = steps
        click.echo(f"Single step: {step1}")
    else:
        step0, step1 = sorted(steps)
        click.echo(f"Range steps: {step0} to {step1}")

    if steps and n_traj:
        print("Collecting trajectory...")
        atoms_traj = []
        step0, step1 = sorted(steps)[:2]
        for traj_index in range(n_traj):
            atoms_traj.extend(get_atoms_list(folder, [traj_index, step0, step1]))
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
