#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:18:49 2024

@author: ubuntu
"""

import argparse
import json

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimate Time')
    parser.add_argument('-m', '--model', required=True, help='Time estimator model')
    parser.add_argument('-v', '--vacancy_info',nargs=2, required=True, help='vacancy concentration', default=[1, 256])
    parser.add_argument('-t', '--temperature', required=True, help='temperature of the simulation')
    parser.add_argument('-t', '--trajectory', required=True, help='directory where thermodynamic trajectories are')
    parser.add_argument('-n', '--n_traj', help='number of thermodynamic trajectory', type=int, default=10)
    parser.add_argument('-d', '--device', help='device to run the model', default='cuda:0')
    args = parser.parse_args()

    vacancy, ideal_n_atoms = args.vacancy_info
    defect = vacancy/ideal_n_atoms
    model = registry.get_model_class("t_net").load(args.model)
    model.eval()
    model.to(args.device)
    atoms_l = [io.read(f"{args.trajectory}/{k}"+str(k), index=':') for k in range(10)]
    time_filename = f"Time_{args.temperature}K_{defect:.3f}.json"
    sro_filename = f"SRO_{args.temperature}K_{defect:.3f}.json"
    print(f"Time: {time_filename},  SRO: {sro_filename}")
    estimate_time(model, args.temperature, defect, atoms_l, args.n_traj, time_filename, sro_filename, device)