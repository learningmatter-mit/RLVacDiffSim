#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 11:47:40 2024

@author: ubuntu
"""
import json
import os

import click
import numpy as np
from ase import io
from ase.data import chemical_symbols

from rlsim.utils.sro import get_sro_from_atoms

dataset = []


def get_WC0(elements, goal):
    # Initialize an empty matrix
    matrix_size = len(elements)
    matrix = np.zeros((matrix_size, matrix_size))
    # Fill the matrix using the dictionary
    for i, item1 in enumerate(elements):
        element1 = chemical_symbols[item1]
        for j, item2 in enumerate(elements):
            element2 = chemical_symbols[item2]
            key1 = f"{element1}-{element2}"
            key2 = f"{element2}-{element1}"
            if key1 in goal:
                matrix[i, j] = goal[key1]
            elif key2 in goal:
                matrix[i, j] = goal[key2]
    return np.triu(matrix)


def atoms_to_dict(atoms):
    state = {'positions': atoms.get_positions().tolist(),
             'atomic_numbers': atoms.get_atomic_numbers().tolist(),
             'cell': atoms.cell.tolist()}
    return state


@click.command()
@click.option("-t", "--temperature", required=True, type=float, help="Temperature of the simulation")
@click.option("-g", "--goal_state_file", required=True, type=str, help="Path to the goal state data")
@click.option("--species", required=True, type=int, multiple=True, help="Species to consider")
@click.option("--traj_list", required=True, type=str, multiple=True, help="List of directories containing the trajectories")
@click.option("--save_dir", required=True, default="./", help="Directory to save the results")
def main(temperature, species, traj_list, goal_state_file, save_dir):

    numerator, denominator = 0, 0
    next_frame_numerator = 0.0
    goal_state_data = json.load(open(goal_state_file, "r"))
    cutoff = goal_state_data["cutoff"][str(int(temperature))]
    species = sorted(species)
    WC0 = get_WC0(species, goal_state_data["goal_state"][str(int(temperature))])
    print(f"Goal state: \n{WC0}")
    print(f"Goal state cutoff: {cutoff:.3f}")
    for name in traj_list:
        print(name)
        with open(os.path.join(name, "diffuse.json"), 'r') as file:
            time = json.load(file)[0]

        Ntraj = len(time)
        for j in range(Ntraj):
            atomsj = io.read(os.path.join(name, f'XDATCAR{j}'), index=':')
            for frame in range(len(atomsj)-1):
                dt = time[j][frame+1] - time[j][frame]
                state = atoms_to_dict(atomsj[frame])
                next_state = atoms_to_dict(atomsj[frame+1])
                SRO = np.triu(get_sro_from_atoms(atomsj[frame]))
                SRO_next = np.triu(get_sro_from_atoms(atomsj[frame+1]))
                terminate = (np.linalg.norm(SRO-WC0) < cutoff)
                terminate_next = (np.linalg.norm(SRO_next-WC0) < cutoff)
                if terminate:
                    numerator += 1
                if not terminate and terminate_next:
                    next_frame_numerator += 1
                denominator += 1
                dataset.append({'state': state,
                                'next': next_state,
                                'dt': dt,
                                'SRO': SRO.tolist(),
                                'terminate': int(terminate),
                                'terminate_next': int(terminate_next)})
            if j % 10 == 0:
                print(f"{j} | {numerator/denominator*100:.3f} % | {next_frame_numerator / denominator*100:.3f} %")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f'dataset_{int(temperature)}.json')
    with open(filename, 'w') as file:
        json.dump(dataset, file)

