#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 2026

@author: Ubuntu/Antigravity
"""
import json
import os

import click
import numpy as np
from ase import io
from ase.data import chemical_symbols

from rlsim.utils.sro import get_binary_sro_from_atoms



def atoms_to_dict(atoms):
    state = {'positions': atoms.get_positions().tolist(),
             'atomic_numbers': atoms.get_atomic_numbers().tolist(),
             'cell': atoms.cell.tolist()}
    return state


@click.command()
@click.option("-t", "--temperature", required=True, type=float, help="Temperature of the simulation")
@click.option("-g", "--goal_state_file", required=True, type=str, help="Path to the goal state data")
@click.option("--traj_list", required=True, type=str, multiple=True, help="List of directories containing the trajectories")
@click.option("--save_dir", required=True, default="./", help="Directory to save the results")
@click.option("--multiplier", default=1.0, type=float, help="Multiplier for the threshold (std)")
def main(temperature, traj_list, goal_state_file, save_dir, multiplier):
    dataset = []
    numerator, denominator = 0, 0
    next_frame_numerator = 0.0
    goal_state_data = json.load(open(goal_state_file, "r"))
    species = goal_state_data["species"]
    
    WC0, threshold = goal_state_data["goal_state"][str(int(temperature))]
    
    print(f"Goal state: {WC0}")
    print(f"Goal state threshold (std): {threshold}")
    print(f"Multiplier: {multiplier}")
    effective_threshold = threshold * multiplier
    print(f"Effective range: [{WC0 - effective_threshold:.4f}, {WC0 + effective_threshold:.4f}]")
    
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
                
                SRO = get_binary_sro_from_atoms(atomsj[frame])
                SRO_next = get_binary_sro_from_atoms(atomsj[frame+1])
                
                terminate = abs(SRO - WC0) <= effective_threshold
                terminate_next = abs(SRO_next - WC0) <= effective_threshold
                
                if terminate:
                    numerator += 1
                if not terminate and terminate_next:
                    next_frame_numerator += 1
                denominator += 1
                dataset.append({'state': state,
                                'next': next_state,
                                'dt': dt,
                                'SRO': SRO,
                                'terminate': int(terminate),
                                'terminate_next': int(terminate_next)})
            if j % 10 == 0:
                print(f"{j} | {numerator/denominator*100:.3f} % | {next_frame_numerator / denominator*100:.3f} %")
                
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f'dataset_{int(temperature)}.json')
    with open(filename, 'w') as file:
        json.dump(dataset, file)
    print(f"Saved dataset to {filename}")


if __name__ == "__main__":
    main()
