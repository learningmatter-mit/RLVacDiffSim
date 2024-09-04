#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:18:49 2024

@author: ubuntu
"""

import argparse
import json

import click
import numpy as np
import torch
from ase import io
from rgnn.common.registry import registry
from rgnn.graph.atoms import AtomsGraph
from rgnn.graph.utils import batch_to

from rlsim.utils.sro import get_sro


def timer_converter(t, tau=30, threshold=0.9999):
    if t < threshold * tau:
        return -tau * torch.log(1 - t / tau)
    else:
        # Linear approximation for large t values
        return tau * (t / tau - 1 + torch.log(torch.tensor(threshold, device=t.device, dtype=t.dtype)))


def estimate_time(model, temperature, concentration, atoms_l, n_traj, time_file, sro_file, device):
    out = []
    sro_out = []

    for j in range(int(len(atoms_l[:][0])/10)):
        i = 10*j
        out.append([])
        sro_out.append([])
        for k in range(n_traj):
            data = AtomsGraph.from_ase(atoms_l[k][i], model.cutoff, read_properties=False, neighborlist_backend="ase", add_batch=True)
            batch = batch_to(data, device)
            pred_time = model(batch, temperature=temperature, defect=concentration)
            time_real = timer_converter(pred_time, model.tau)
            out[-1] += [float(time_real.detach())]
            SRO = get_sro(atoms_l[k][i])
            sro_norm = np.linalg.norm(SRO)
            sro_out[-1] += [sro_norm]
        if i % 20 == 0:
            print('complete '+str(i//20+1)+'%')

    with open(time_file) as file:
        json.dump(out, file)
    with open(sro_file) as file:
        json.dump(sro_out, file)


@click.command()
@click.option("-m", "--model", required=True, help="Time estimator model")
@click.option("-v", "--vacancy-info", nargs=2, required=True, type=int, help="Vacancy concentration and ideal number of atoms")
@click.option("-t", "--temperature", required=True, type=int, help="Temperature of the simulation")
@click.option("--trajectory", required=True, help="Directory where thermodynamic trajectories are")
@click.option("-n", "--n_traj", default=10, type=int, help="Number of thermodynamic trajectories")
@click.option("-d", "--device", default="cuda:0", help="Device to run the model (default: 'cuda:0')")
def main(model, vacancy_info, temperature, trajectory, n_traj, device):
    """
    Main command-line interface for time estimation simulation.

    Parameters:
    - model (str): Path to the time estimator model file.
    - vacancy_info (tuple): A tuple of (vacancy, ideal_n_atoms) for defect calculation.
    - temperature (int): Temperature of the simulation.
    - trajectory (str): Directory where thermodynamic trajectories are stored.
    - n_traj (int): Number of thermodynamic trajectories.
    - device (str): Device to run the model on (e.g., "cuda:0").
    """
    vacancy, ideal_n_atoms = vacancy_info
    defect = vacancy / ideal_n_atoms

    # Load the model
    model = registry.get_model_class("t_net").load(model)
    model.eval()
    model.to(device)

    # Read atoms
    atoms_l = [io.read(f"{trajectory}/{k}", index=':') for k in range(n_traj)]

    # Prepare file names
    time_filename = f"Time_{temperature}K_{defect:.3f}.json"
    sro_filename = f"SRO_{temperature}K_{defect:.3f}.json"

    # Output filenames
    print(f"Time: {time_filename},  SRO: {sro_filename}")
    # Run time estimation
    estimate_time(model, temperature, defect, atoms_l, n_traj, time_filename, sro_filename, device)
