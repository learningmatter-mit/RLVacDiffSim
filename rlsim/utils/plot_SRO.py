# import os
import json
import math
import multiprocessing as mp
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# from itertools import product
from ase import io

from .sro import get_sro


def plot(filename, results, info):
    if isinstance(results, List):
        results = np.array(results)
    
    font = {"size": 28}
    matplotlib.rc("font", **font)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Arial"]
    specie_dict = {24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn"}
    fig, ax = plt.subplots(figsize=(16, 10))

    for u in range(len(info["species"])):
        for v in range(u + 1):
            name = (
                specie_dict[info["species"][u]] + "-" + specie_dict[info["species"][v]]
            )
            ax.plot(results[:, u, v], label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Warren-Cowley parameter")
    ax.set_xlim(0, results.shape[0])
    ax.set_ylim(-0.50,0.50)
    ax.legend()
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=math.factorial(len(info["species"]))/math.factorial(2)/math.factorial(len(info["species"])-2))
    ax.set_title(f"Total number of atoms: {info['n_atoms']}")
    fig.tight_layout()
    fig.savefig(filename, dpi=500)


if __name__ == "__main__":
    ##### setting analysis input ##############
    # please first set the number of mpi processes by nproc = ...
    # then provide a list of path to trajectory .json files for analysis
    # for example, if the file is /path/to/traj/traj13.json,
    # put string '/path/to/traj/traj13' in the list to specify this path

    nproc = 10
    path_list = ["XDATCAR" + str(i) for i in range(10)]

    ##### running the analysis ################
    # path_list = [path for path in path_list if os.path.exists(path + ".json")]
    results = []
    with mp.Pool(nproc) as pool:
        results = pool.map(get_sro, path_list)
    SRO_results = []
    for i in range(len(results)):
        if results[i][0] is not None:
            SRO_results.append(results[i][0])
    # SRO_results = np.stack(SRO_results)
    # SRO_results = np.mean(SRO_results, axis=0)
    for i, sro_result in enumerate(SRO_results):
        plot(f"{i}_sro.png", sro_result, results[0][1])
        with open(f"{i}_SRO_results.json", "w") as file:
            json.dump({"SRO": sro_result.tolist(), "info": results[i][1]}, file)

