import json
import multiprocessing as mp
import os
from itertools import product
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ase import io


def dx2(file_path):
    atoms_list = io.read(file_path, ":")
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


def plot(x2, t, D: float | None = None):
    if isinstance(x2, List):
        x2 = np.array(x2)
    if isinstance(t, List):
        t = np.array(t)
    plt.figure(figsize=(16, 10))

    plt.rcParams["font.size"] = 32
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Arial"]

    plt.scatter(t, x2, c="gray")

    tfit = t.tolist() + (-t).tolist()
    xfit = x2.tolist() + (-x2).tolist()
    slope = np.polyfit(tfit, xfit, 1)[0]
    if D is None:
        D = slope / 6
    plt.plot([0, np.max(t)], [0, np.max(t) * slope], c="blue", linewidth=3)
    plt.axis([0, np.max(t), 0, np.max(x2)])

    plt.annotate(
        f"D = {D:.2e} m$^2$/s",
        xy=(0.75, 0.75),
        color="black",
        fontsize=20,
    )
    plt.xlabel(r"time ($\mu$s)")
    plt.ylabel(r"$\Delta$x$^2$ (A$^2$)")
    plt.savefig("plot_diff.png", dpi=500)

    return D


if __name__ == "__main__":
    ##### setting analysis input ##############
    # please first set the number of mpi processes by nproc = ...
    # then provide a list of path to trajectory .json files for analysis.
    # For example, if the file is /path/to/traj/traj13.json,
    # put string '/path/to/traj/traj13' in the list to specify this path.
    # The output file 'diffuse.json' from the model deployment
    # needs to be put in the same folder when running this script.
    # Make sure the order of data in 'diffuse.json' is the same as in path_list

    nproc = 10
    path_list = ["XDATCAR" + str(i) for i in range(10)]
    results = []
    with mp.Pool(nproc) as pool:
        results = pool.map(dx2, path_list)

    dx2_list = []
    dt_list = []

    with open("diffuse.json", "r") as file:
        data_d = json.load(file)

    for i in range(len(results)):
        if results[i] is not None:
            dx2_list.append(results[i])
            dt_list.append(data_d[0][i][: len(results[i])])

    dx2_list = np.hstack(dx2_list)
    dt_list = np.hstack(dt_list)
    tfit = dt_list.tolist() + (-dt_list).tolist()
    xfit = dx2_list.tolist() + (-dx2_list).tolist()
    D = np.polyfit(tfit, xfit, 1)[0] / 6
    with open("diffusivity.json", "w") as file:
        json.dump(
            {
                "x2": dx2_list.tolist(),
                "t": dt_list.tolist(),
                # "D": [D, r"$\times$ 10$^{-14}$ m$^2$/s"],
            },
            file,
        )

    plot(dx2_list, dt_list)
