from heapq import nsmallest
from itertools import product

import ase
import numpy as np
from numpy.linalg import norm
from scipy.spatial import ConvexHull


def get_action_space(config, lattice_parameter: float = 3.528):
    a = lattice_parameter
    actions = []
    dist_mat = config.atoms.get_all_distances(mic=True)

    crit = np.sum(dist_mat < a/np.sqrt(2)*1.2, axis=1)
    vacancy_l = np.argwhere(crit != 13).T[0]

    def test(i, vec):
        test = config.atoms.copy()
        pos_test = test.get_positions()
        pos_test[i] += vec
        test.set_positions(pos_test)

        return np.sum(test.get_distances(i, range(len(test)), mic=True) < 0.8) == 1

    acts = np.array([[1, 1, 0],
                     [1, -1, 0],
                     [-1, 1, 0],
                     [-1, -1, 0],
                     [1, 0, 1],
                     [1, 0, -1],
                     [-1, 0, 1],
                     [-1, 0, -1],
                     [0, 1, 1],
                     [0, 1, -1],
                     [0, -1, 1],
                     [0, -1, -1]])*a/2*0.8

    for i in vacancy_l:
        for vec in acts:
            vacant = test(i, vec)
            if vacant:
                actions.append([i]+vec.tolist())

    return actions


def get_action_space_mcmc(config, lattice_parameter: float = 3.528, action_mode="vacancy_only"):
    """First choosing focal atom and generate actions for that site

    Args:
        config (Environment): Environment
        lattice_parameter (float): Lattice paramter for a unit celll. Defaults to 3.528.
        action_mode (str): Types of actions. Defaults to "vacancy_only".

    Returns:
        action_space: List of [site, vector]
    """
    a = lattice_parameter
    actions = []

    dist_mat = config.atoms.get_all_distances(mic=True)

    crit = np.sum(dist_mat < a/np.sqrt(2)*1.2, axis=1)
    vacancy_l = np.argwhere(crit != 13).T[0]
    filled_l = np.argwhere(crit == 13).T[0]

    def test(i, vec):
        test = config.atoms.copy()
        pos_test = test.get_positions()
        pos_test[i] += vec
        test.set_positions(pos_test)

        return np.sum(test.get_distances(i, range(len(test)), mic=True) < 0.8) == 1

    def test_filled(i, vec):
        test = config.atoms.copy()
        pos_test = test.get_positions()
        symbols_test = test.get_chemical_symbols()
        pos_test[i] += vec
        test.set_positions(pos_test)
        distnaces = test.get_distances(i, range(len(test)), mic=True)
        swap_sites = []
        for j, dist in enumerate(distnaces):
            if dist < 0.8 and symbols_test[j] != symbols_test[i]:
                swap_sites.append(j)

        return swap_sites

    acts = np.array([[1, 1, 0],
                     [1, -1, 0],
                     [-1, 1, 0],
                     [-1, -1, 0],
                     [1, 0, 1],
                     [1, 0, -1],
                     [-1, 0, 1],
                     [-1, 0, -1],
                     [0, 1, 1],
                     [0, 1, -1],
                     [0, -1, 1],
                     [0, -1, -1]])*a/2*0.8

    actions = []
    if action_mode == "vacancy_only":
        index = np.random.choice(vacancy_l)
        for vec in acts:
            vacant = test(index, vec)
            if vacant:
                actions.append([index]+vec.tolist())
        return actions
    elif action_mode == "fix_vacancy":
        while not actions:
            index = np.random.choice(filled_l)
            for vec in acts:
                swap_sites = test_filled(index, vec)
                for site in swap_sites:
                    actions.append([index]+[site])
        return actions
    elif action_mode == "all":
        while not actions:
            index = np.random.choice(np.concatenate([vacancy_l, filled_l]))
            if index in vacancy_l:
                for vec in acts:
                    vacant = test(index, vec)
                    if vacant:
                        actions.append([index]+vec.tolist())
                    else:
                        swap_sites = test_filled(index, vec)
                        for site in swap_sites:
                            actions.append([index]+[site])
            elif index in filled_l:
                for vec in acts:
                    swap_sites = test_filled(index, vec)
                    for site in swap_sites:
                        actions.append([index]+[site])
            return actions
        else:
            raise ValueError(f"Invalid action mode: {action_mode}")
