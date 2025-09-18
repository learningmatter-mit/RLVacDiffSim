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
        new_site = pos_test[i] + vec
        swap_sites = []
        for j, pos in enumerate(pos_test):
            if np.linalg.norm(pos - new_site) < 0.8 and symbols_test[j] != symbols_test[i]:
                swap_sites.append(j)
        return swap_sites

    def test_filled_all(i):
        test = config.atoms.copy()
        symbols_test = test.get_chemical_symbols()
        swap_sites = []
        for j, symbol in enumerate(symbols_test):
            if symbol != symbols_test[i]:
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
        rng = np.random.default_rng()
        index = rng.choice(vacancy_l)
        for vec in acts:
            vacant = test(index, vec)
            if vacant:
                actions.append([index]+vec.tolist())
        return actions
    elif action_mode == "fix_vacancy":
        while not actions:
            rng = np.random.default_rng()
            index = rng.choice(filled_l)
            for vec in acts:
                swap_sites = test_filled(index, vec)
                for site in swap_sites:
                    actions.append([index]+[site])
        return actions
    
    elif action_mode == "fix_vacancy_all":
        while not actions:
            rng = np.random.default_rng()
            index = rng.choice(filled_l)
            swap_sites = test_filled_all(index)
            for site in swap_sites:
                actions.append([index]+[site])
        return actions
    else:
        while not actions:
            rng = np.random.default_rng()
            index = rng.choice(np.concatenate([vacancy_l, filled_l]))
            if index in vacancy_l:
                for vec in acts:
                    vacant = test(index, vec)
                    if vacant:
                        actions.append([index]+vec.tolist())
            elif index in filled_l:
                for vec in acts:
                    swap_sites = test_filled(index, vec)
                    for site in swap_sites:
                        actions.append([index]+[site])
        return actions
