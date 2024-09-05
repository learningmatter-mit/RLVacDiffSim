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


def estimate_time(model, temperature, concentration, atoms_l, time_file, sro_file, device):
    out = []
    sro_out = []
    for i, traj in enumerate(atoms_l):
        n_frames = len(traj)
        out.append([])
        sro_out.append([])
        for j in range(int(n_frames/10)):
            k = 10*j
            data = AtomsGraph.from_ase(traj[k], model.cutoff, read_properties=False, neighborlist_backend="ase", add_batch=True)
            batch = batch_to(data, device)
            pred_time = model(batch, temperature=temperature, defect=concentration)
            time_real = timer_converter(pred_time, model.tau)
            out[-1].extend([float(time_real.detach())])
            SRO = get_sro([traj[k]])[0]
            sro_norm = np.linalg.norm(SRO)
            sro_out[-1].extend([sro_norm])
            if k % 50 == 0:
                print(f"Complete {i+1}/{len(atoms_l)} | {(k/n_frames*100):.2f} %")
    out = np.transpose(np.array(out)).tolist()
    sro_out = np.transpose(np.array(sro_out)).tolist()
    with open(time_file, "w") as file:
        json.dump(out, file)
    with open(sro_file, "w") as file:
        json.dump(sro_out, file)


@click.command()
@click.option("-m", "--model", required=True, help="Time estimator model")
@click.option("-v", "--vacancy-info", nargs=2, required=True, type=int, help="Vacancy concentration and ideal number of atoms")
@click.option("-t", "--temperature", required=True, type=int, help="Temperature of the simulation")
@click.option("-f", "--file_list", required=True, type=click.Path(exists=True), multiple=True, help="Directory where thermodynamic trajectories are")
@click.option("-s", "--save_dir", required=True, default="./", type=click.Path(exists=True), help="Directory to save the results")
@click.option("-d", "--device", default="cuda", help="Device to run the model (default: 'cuda')")
def main(model, vacancy_info, temperature, file_list, save_dir, device):
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
    atoms_l = [io.read(f"{file}", index=':') for file in file_list]

    # Prepare file names
    time_filename = f"{save_dir}/Time_{temperature}K_{defect:.3f}.json"
    sro_filename = f"{save_dir}/SRO_{temperature}K_{defect:.3f}.json"

    # Output filenames
    print(f"Time: {time_filename},  SRO: {sro_filename}")
    # Run time estimation
    estimate_time(model, temperature, defect, atoms_l, time_filename, sro_filename, device)
