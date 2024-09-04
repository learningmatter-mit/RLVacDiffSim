import argparse
from itertools import product

import click
from ase import io

from rlsim.drl.deploy import deploy_RL


def get_atoms(folder, pool_ind):
    traj = pool_ind[0]
    frame = pool_ind[1]
    atoms = io.read(f"{folder}/XDATCAR{traj}", index=frame)
    return atoms


@click.command()
@click.option('-f', '--folder', required=True, help='Trajectory path')
@click.option('-c', '--config', required=True, help='Config file path')
@click.option('-s', '--step', type=int, default=None, help='Number of steps to extract from XDATCAR')
@click.option('-n', '--n-traj', type=int, default=None, help='Number of XDATCAR files to process')
def main(dqn_path, config, step, n_traj):
    """
    CLI for deploying DRL models using DQN trajectory data.
    """
    if step and n_traj:
        atoms_traj = []
        pool = [(j, k) for j, k in product(range(n_traj), range(step))]
        for traj_index, image_index in pool:
            atoms_traj.append(get_atoms(dqn_path, [traj_index, image_index]))
    else:
        atoms_traj = io.read(dqn_path, index=':')

    deploy_RL(config, atoms_traj=atoms_traj)