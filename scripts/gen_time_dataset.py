import argparse
from itertools import product

from ase import io

from rlsim.drl.deploy import deploy_RL


def get_atoms(folder, pool_ind):
    traj = pool_ind[0]
    frame = pool_ind[1]
    atoms = io.read(f"{folder}/XDATCAR{traj}", index=frame)
    return atoms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DRL model')
    parser.add_argument('-d', '--dqn_path', required=True, help='DQN trajectory path')
    parser.add_argument('-c', '--config', required=True, help='config file path')
    parser.add_argument('-s', '--step', help='step', type=int, default=None)
    parser.add_argument('-n', '--n_traj', help='number of xdatcar', type=int, default=None)
    args = parser.parse_args()
    if args.step and args.n_traj:
        atoms_traj = []
        pool = [(j, k) for j, k in product(range(args.n_traj), range(args.step))]
        for traj_index, image_index in pool:
            atoms_traj.append(get_atoms(args.dqn_path, [traj_index, image_index]))
    else:
        atoms_traj = io.read(args.dqn_path, index=':')

    deploy_RL(args.config, atoms_traj=atoms_traj)
