import json
import multiprocessing as mp
import os

import click
import numpy as np
from ase import io
from tqdm import tqdm


def get_dx2(atoms_list):
    nframe = len(atoms_list)

    vacancy = np.zeros((nframe, 3))

    for i in range(nframe):
        atoms = atoms_list[i]
        dist_all = atoms.get_all_distances(mic=True)
        dist_list = dist_all.flatten()
        dist_list = dist_list[dist_list > 0.1]
        NN_dist = np.min(dist_list)
        r_cut = (1 + np.sqrt(2)) / 2 * NN_dist

        n_NN = np.sum((dist_all > 0.1) * (dist_all < r_cut), axis=1)
        vacancy_NN = np.argwhere(n_NN < 12).T[0]
        pos_0 = atoms.get_positions()[vacancy_NN[0]]
        dist_vec = atoms.get_distances(vacancy_NN[0], vacancy_NN, mic=True, vector=True)
        vacancy[i] = pos_0 + np.mean(dist_vec, axis=0)

    a_vec = np.diag(atoms.cell)
    for i in range(1, len(vacancy)):
        r0 = vacancy[i - 1]
        r1 = vacancy[i]
        displacement = (r1 - r0 + a_vec / 2) % a_vec - a_vec / 2
        vacancy[i] = r0 + displacement

    r0 = vacancy[0, :]
    dx2_list = np.linalg.norm(vacancy - vacancy[0], axis=1) ** 2

    return dx2_list


def process_trajectory(args):
    traj, index = args
    dx2 = get_dx2(traj)
    return index, dx2


@click.command()
@click.option("-i", "--input_dir", required=True, type=click.Path(exists=True), help="Input dir containing diffussion info")
@click.option("-f", "--file_list", required=True, multiple=True, help="Directory where thermodynamic trajectories are")
@click.option("-n", "--nproc", type=int, default=10, help="Number of process")
@click.option("-s", "--save_dir", required=True, default="./", help="Directory to save the results")
def main(input_dir, file_list, nproc, save_dir):
    if save_dir not in os.listdir():
        os.makedirs(save_dir, exist_ok=True)
    input_file = os.path.join(input_dir, "diffuse.json")
    with open(input_file, "r") as file:
        data_d = json.load(file)
    traj_l = [io.read(os.path.join(input_dir, file), index=':') for file in file_list]

    # Prepare arguments for multiprocessing
    args = [(traj, i) for i, traj in enumerate(traj_l)]

    # Use multiprocessing to process trajectories in parallel
    with mp.Pool(nproc) as pool:
        # Use tqdm to show progress with pool.imap_unordered for real-time updates
        dx2_results = list(tqdm(pool.imap(process_trajectory, args), total=len(args), desc="Processing trajectories"))

    # Sort results by index to maintain the order
    dx2_results.sort(key=lambda x: x[0])
    dx2_list = [result[1] for result in dx2_results]

    # Load the diffusion time data
    dt_list = []
    for i in range(len(dx2_list)):
        dt_list.append(data_d[0][i][: len(dx2_list[i])])

    # Flatten the results
    dx2_list = np.hstack(dx2_list)
    dt_list = np.hstack(dt_list)
    tfit = dt_list.tolist() + (-dt_list).tolist()
    xfit = dx2_list.tolist() + (-dx2_list).tolist()
    D = np.polyfit(tfit, xfit, 1)[0] / 6
    with open(f"{save_dir}/diffusivity.json", "w") as file:
        json.dump(
            {
                "x2": dx2_list.tolist(),
                "t": dt_list.tolist(),
                "D": [D, r"$\times$ 10$^{-14}$ m$^2$/s"], # ang**2/(microsecond)
            },
            file,
        )

