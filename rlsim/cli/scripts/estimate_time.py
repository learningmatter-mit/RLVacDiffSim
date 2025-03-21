import json
import os

import click
import torch
from ase import io
from rgnn.common.registry import registry
from rgnn.graph.atoms import AtomsGraph
from rgnn.graph.dataset.atoms import AtomsDataset
from rgnn.graph.utils import batch_to
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def timer_converter(t, tau, threshold=0.9999):
    """
    Converts time values in a batch-wise manner.

    Args:
        t (torch.Tensor): Input tensor of time values.
        tau (float or torch.Tensor): Time constant (can be scalar or tensor).
        threshold (float, optional): Threshold value for switching to linear approximation.

    Returns:
        torch.Tensor: Converted time values with batch support.
    """
    # Create a mask for values below threshold * tau
    mask = t < threshold * tau

    # Compute values for both cases
    log_values = -tau * torch.log(1 - t / tau)
    linear_values = tau * (t / tau - 1 + torch.log(torch.tensor(threshold, device=t.device, dtype=t.dtype)))

    # Apply mask to select appropriate values
    return torch.where(mask, log_values, linear_values)


def estimate_time(model, temperature, defect, traj_l, time_file, batch_size, device):
    out = []
    for i, traj in enumerate(traj_l):
        # n_frames = len(traj)
        time_list = []
        datas_list = []
        print(f"Traj {i+1}/{len(traj_l)}.")
        for atoms in traj:
            data = AtomsGraph.from_ase(atoms, model.cutoff, read_properties=False, neighborlist_backend="torch")
            data.T = torch.as_tensor(temperature,
                            dtype=torch.get_default_dtype(),
                            device=data["elems"].device)
            data.defect = torch.as_tensor(defect,
                            dtype=torch.get_default_dtype(),
                            device=data["elems"].device)
            datas_list.append(data)
        dataset = AtomsDataset(datas_list)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch in loader:
            batch = batch_to(batch, device)
            pred_time = model(batch, inference=True)["time"]
            time_real = timer_converter(pred_time, model.tau)
            time_list.append(time_real.detach().cpu())
            del batch
        del loader, dataset
        out.append(torch.cat(time_list, dim=0).tolist())

    with open(time_file, "w") as file:
        json.dump(out, file)


@click.command()
@click.option("-m", "--model_info", required=True, nargs=2, help="Time estimator model (model name and path)")
@click.option("-v", "--vacancy-info", nargs=2, required=True, type=int, help="Vacancy concentration and ideal number of atoms")
@click.option("-t", "--temperature", required=True, type=int, help="Temperature of the simulation")
@click.option("-i", "--input_dir", required=True, type=click.Path(exists=True), help="Input dir containing trajectory files (XDATCAR) to read from.")
@click.option("-n", "--num_files", required=True, type=int, help="numer of XDATCAR files")
@click.option("-s", "--save_dir", required=True, default="./", type=click.Path(), help="Directory to save the results")
@click.option("--skip", default=1, type=int, help="Skip every n frames")
@click.option("-b", "--batch_size", default=8, type=int, help="Batch size for the DataLoader")
@click.option("-d", "--device", default="cuda", help="Device to run the model (default: 'cuda')")
def main(model_info, vacancy_info, temperature, input_dir, num_files, save_dir, skip, batch_size, device):
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
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    vacancy, ideal_n_atoms = vacancy_info
    defect = vacancy / ideal_n_atoms
    model_name, model_path = model_info
    # Load the model
    model = registry.get_model_class(model_name).load(model_path)
    model.eval()
    model.to(device)

    # Read atoms
    traj_l = []
    for i in tqdm(range(num_files), desc=f"Reading trajectories in {input_dir}"):
        filename = os.path.join(input_dir, f"XDATCAR{i}")
        traj = io.read(filename, index=f'::{skip}')
        traj_l.append(traj)

    # Prepare file names
    time_filename = f"{save_dir}/Time_{temperature}K_{defect:.3f}.json"

    # Run time estimation
    print(f"Estimating time | T: {temperature} K, V: {defect*100:.2f} %, save to {time_filename}.")
    estimate_time(model, temperature, defect, traj_l, time_filename, batch_size, device)
