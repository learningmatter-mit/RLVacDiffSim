from heapq import nsmallest
from itertools import product

import ase
import numpy as np
from numpy.linalg import norm
from scipy.spatial import ConvexHull
from ase.build import bulk
from ase.geometry import get_distances


def get_action_space(config, lattice_parameter=3.615, perfect_lattice_element="Cu"):
    """
    For CrCoNi: perfect_lattice_element: Ni, lattice_parameters: 3.528
    For Cu3Au: perfect_lattice_element: Cu, lattice_parameters: 3.615
    """
    cell = np.array(config.atoms.get_cell())
    pbc = config.atoms.get_pbc()

    Nrepeat = np.round(cell.diagonal() / lattice_parameter).astype(int)
    # Step 1: Build perfect Nrepeat x Nrepeat x Nrepeat FCC lattice sites
    perfect = bulk(perfect_lattice_element, 'fcc', a=lattice_parameter, cubic=True).repeat(Nrepeat)
    lattice_sites = perfect.get_positions()

    # Step 2: Assign each atom to its nearest lattice site via MIC distance
    atom_pos = config.atoms.get_positions()
    _, dists = get_distances(atom_pos, lattice_sites, cell=cell, pbc=pbc)
    # dists shape: (n_atoms, n_sites)
    atom_to_site = np.argmin(dists, axis=1)

    # Step 3: Identify the vacant lattice site
    occupied = set(atom_to_site.tolist())
    vacant = list(set(range(len(lattice_sites))) - occupied)
    assert len(vacant) == 1, f"Expected 1 vacancy, found {len(vacant)}"
    vac_site_idx = vacant[0]
    vac_pos = lattice_sites[vac_site_idx]
    # print(f"Vacant lattice site index: {vac_site_idx}, position: {vac_pos}")

    # Step 4: Find NN lattice sites of the vacancy (FCC NN distance = lattice_parameter/sqrt(2), expect 12)
    nn_dist = lattice_parameter / np.sqrt(2)
    _, vac_to_sites = get_distances([vac_pos], lattice_sites, cell=cell, pbc=pbc)
    nn_site_mask = (vac_to_sites[0] > 0.1) & (vac_to_sites[0] < nn_dist * 1.1)
    nn_sites = np.where(nn_site_mask)[0]
    # print(f"Number of NN sites to vacancy: {len(nn_sites)}")

    # Step 5: Build actions — each NN atom moves to the vacant site
    site_to_atom = {int(site): int(atom) for atom, site in enumerate(atom_to_site)}

    actions = []
    for site in nn_sites:
        atom_idx = site_to_atom[site]
        # MIC displacement from atom's current position to the vacant lattice site
        diff = vac_pos - atom_pos[atom_idx]
        frac = np.linalg.solve(cell.T, diff)
        frac -= np.round(frac)
        disp = (cell.T @ frac)

        actions.append([atom_idx] + disp.tolist())

    return actions

def get_action_space_mcmc(config, lattice_parameter=3.615, perfect_lattice_element="Cu", action_mode="vacancy_only"):
    """
    First choosing focal atom and generate actions for that site
    Uses perfect lattice mapping to efficiently find valid actions without trial displacements.
    
    Args:
        config (Environment): Environment
        lattice_parameter (float): Lattice parameter for a unit cell. Defaults to 3.615.
        perfect_lattice_element (str): Element to build perfect FCC. Defaults to "Cu".
        action_mode (str): Types of actions. Defaults to "vacancy_only".

    Returns:
        action_space: List of [site, vector] for vacancy jumps, or [site, site] for swaps.
    """
    cell = np.array(config.atoms.get_cell())
    pbc = config.atoms.get_pbc()

    Nrepeat = np.round(cell.diagonal() / lattice_parameter).astype(int)
    # Step 1: Build perfect Nrepeat x Nrepeat x Nrepeat FCC lattice sites
    perfect = bulk(perfect_lattice_element, 'fcc', a=lattice_parameter, cubic=True).repeat(Nrepeat)
    lattice_sites = perfect.get_positions()

    # Step 2: Assign each atom to its nearest lattice site via MIC distance
    atom_pos = config.atoms.get_positions()
    _, dists = get_distances(atom_pos, lattice_sites, cell=cell, pbc=pbc)
    atom_to_site = np.argmin(dists, axis=1)

    # Step 3: Identify the vacant lattice site
    occupied = set(atom_to_site.tolist())
    vacant = list(set(range(len(lattice_sites))) - occupied)
    assert len(vacant) == 1, f"Expected 1 vacancy, found {len(vacant)}"
    vac_site_idx = vacant[0]
    vac_pos = lattice_sites[vac_site_idx]

    # Step 4: Map sites to atoms
    site_to_atom = {int(site): int(atom) for atom, site in enumerate(atom_to_site)}

    # Find NN lattice sites of the vacancy to determine vacancy_l
    nn_dist = lattice_parameter / np.sqrt(2)
    _, vac_to_sites = get_distances([vac_pos], lattice_sites, cell=cell, pbc=pbc)
    nn_site_mask = (vac_to_sites[0] > 0.1) & (vac_to_sites[0] < nn_dist * 1.1)
    nn_sites_of_vac = np.where(nn_site_mask)[0]
    
    vacancy_l = [site_to_atom[s] for s in nn_sites_of_vac if s in site_to_atom]
    filled_l = list(set(site_to_atom.values()) - set(vacancy_l))

    actions = []
    
    if action_mode == "vacancy_only":
        index = np.random.choice(vacancy_l)
        diff = vac_pos - atom_pos[index]
        frac = np.linalg.solve(cell.T, diff)
        frac -= np.round(frac)
        disp = (cell.T @ frac)
        actions.append([int(index)] + disp.tolist())
        return actions

    symbols = config.atoms.get_chemical_symbols()

    if action_mode == "fix_vacancy":
        while not actions:
            index = np.random.choice(filled_l)
            site = atom_to_site[index]
            site_pos = lattice_sites[site]
            
            _, site_dists = get_distances([site_pos], lattice_sites, cell=cell, pbc=pbc)
            nn_sites = np.where((site_dists[0] > 0.1) & (site_dists[0] < nn_dist * 1.1))[0]
            
            for nn_site in nn_sites:
                if nn_site in site_to_atom:
                    nn_atom = site_to_atom[nn_site]
                    if symbols[nn_atom] != symbols[index]:
                        actions.append([int(index), int(nn_atom)])
        return actions

    if action_mode == "all":
        while not actions:
            index = np.random.choice(list(site_to_atom.values()))
            site = atom_to_site[index]
            site_pos = lattice_sites[site]
            
            _, site_dists = get_distances([site_pos], lattice_sites, cell=cell, pbc=pbc)
            nn_sites = np.where((site_dists[0] > 0.1) & (site_dists[0] < nn_dist * 1.1))[0]
            
            for nn_site in nn_sites:
                if nn_site == vac_site_idx:
                    diff = vac_pos - atom_pos[index]
                    frac = np.linalg.solve(cell.T, diff)
                    frac -= np.round(frac)
                    disp = (cell.T @ frac)
                    actions.append([int(index)] + disp.tolist())
                elif nn_site in site_to_atom:
                    nn_atom = site_to_atom[nn_site]
                    if symbols[nn_atom] != symbols[index]:
                        actions.append([int(index), int(nn_atom)])
        return actions

    raise ValueError(f"Invalid action mode: {action_mode}")

### Previous version that worked perfectly for CrCoNi
# def get_action_space(config, lattice_parameter: float = 3.528):
#     a = lattice_parameter
#     actions = []
#     dist_mat = config.atoms.get_all_distances(mic=True)

#     crit = np.sum(dist_mat < a/np.sqrt(2)*1.2, axis=1)
#     vacancy_l = np.argwhere(crit != 13).T[0]

#     def test(i, vec):
#         test = config.atoms.copy()
#         pos_test = test.get_positions()
#         pos_test[i] += vec
#         test.set_positions(pos_test)

#         return np.sum(test.get_distances(i, range(len(test)), mic=True) < 0.8) == 1

#     acts = np.array([[1, 1, 0],
#                      [1, -1, 0],
#                      [-1, 1, 0],
#                      [-1, -1, 0],
#                      [1, 0, 1],
#                      [1, 0, -1],
#                      [-1, 0, 1],
#                      [-1, 0, -1],
#                      [0, 1, 1],
#                      [0, 1, -1],
#                      [0, -1, 1],
#                      [0, -1, -1]])*a/2*0.8

#     for i in vacancy_l:
#         for vec in acts:
#             vacant = test(i, vec)
#             if vacant:
#                 actions.append([i]+vec.tolist())

#     return actions

# #TODO: implement this for Cu3Au
# def get_action_space_mcmc(config, lattice_parameter: float = 3.528, action_mode="vacancy_only"):
#     """First choosing focal atom and generate actions for that site
# 
#     Args:
#         config (Environment): Environment
#         lattice_parameter (float): Lattice paramter for a unit celll. Defaults to 3.528.
#         action_mode (str): Types of actions. Defaults to "vacancy_only".
# 
#     Returns:
#         action_space: List of [site, vector]
#     """
#     a = lattice_parameter
#     actions = []
# 
#     dist_mat = config.atoms.get_all_distances(mic=True)
# 
#     crit = np.sum(dist_mat < a/np.sqrt(2)*1.2, axis=1)
#     vacancy_l = np.argwhere(crit != 13).T[0]
#     filled_l = np.argwhere(crit == 13).T[0]
# 
#     def test(i, vec):
#         test = config.atoms.copy()
#         pos_test = test.get_positions()
#         pos_test[i] += vec
#         test.set_positions(pos_test)
# 
#         return np.sum(test.get_distances(i, range(len(test)), mic=True) < 0.8) == 1
# 
#     def test_filled(i, vec):
#         test = config.atoms.copy()
#         pos_test = test.get_positions()
#         symbols_test = test.get_chemical_symbols()
#         pos_test[i] += vec
#         test.set_positions(pos_test)
#         distnaces = test.get_distances(i, range(len(test)), mic=True)
#         swap_sites = []
#         for j, dist in enumerate(distnaces):
#             if dist < 0.8 and symbols_test[j] != symbols_test[i]:
#                 swap_sites.append(j)
# 
#         return swap_sites
# 
#     acts = np.array([[1, 1, 0],
#                      [1, -1, 0],
#                      [-1, 1, 0],
#                      [-1, -1, 0],
#                      [1, 0, 1],
#                      [1, 0, -1],
#                      [-1, 0, 1],
#                      [-1, 0, -1],
#                      [0, 1, 1],
#                      [0, 1, -1],
#                      [0, -1, 1],
#                      [0, -1, -1]])*a/2*0.8
# 
#     actions = []
#     if action_mode == "vacancy_only":
#         index = np.random.choice(vacancy_l)
#         for vec in acts:
#             vacant = test(index, vec)
#             if vacant:
#                 actions.append([index]+vec.tolist())
#         return actions
#     elif action_mode == "fix_vacancy":
#         while not actions:
#             index = np.random.choice(filled_l)
#             for vec in acts:
#                 swap_sites = test_filled(index, vec)
#                 for site in swap_sites:
#                     actions.append([index]+[site])
#         return actions
#     elif action_mode == "all":
#         while not actions:
#             index = np.random.choice(np.concatenate([vacancy_l, filled_l]))
#             if index in vacancy_l:
#                 for vec in acts:
#                     vacant = test(index, vec)
#                     if vacant:
#                         actions.append([index]+vec.tolist())
#                     else:
#                         swap_sites = test_filled(index, vec)
#                         for site in swap_sites:
#                             actions.append([index]+[site])
#             elif index in filled_l:
#                 for vec in acts:
#                     swap_sites = test_filled(index, vec)
#                     for site in swap_sites:
#                         actions.append([index]+[site])
#             return actions
#         else:
#             raise ValueError(f"Invalid action mode: {action_mode}")
