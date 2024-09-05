import json
import multiprocessing as mp
import os

import click
from ase import io
from tqdm import tqdm

from rlsim.utils.sro import get_sro


def get_info(atoms):
    species = list(set(atoms.get_atomic_numbers().tolist()))
    info = {"species": species, "n_atoms": len(atoms)}
    return info


def process_trajectory(args):
    traj, save_dir, index = args
    info = get_info(traj[0])  # Assuming you want the info from the first frame
    sro_results = get_sro(traj)  # Process the full trajectory for SRO results

    # Save results as a JSON file
    save_path = os.path.join(save_dir, f"{index}_SRO_results.json")
    with open(save_path, "w") as file:
        json.dump({"SRO": sro_results.tolist(), "info": info}, file)
    
    return f"Processed trajectory {index}"


@click.command()
@click.option("-i", "--input_dir", required=True, type=click.Path(exists=True), help="Input dir containing diffussion info")
@click.option("-f", "--file_list", required=True, multiple=True, help="Directory where thermodynamic trajectories are")
@click.option("-s", "--save_dir", required=True, default="./", help="Directory to save the results")
@click.option("-n", "--nproc", type=int, default=4, help="Number of process")
def main(input_dir, file_list, nproc, save_dir):
    if save_dir not in os.listdir():
        os.makedirs(save_dir, exist_ok=True)
    print(file_list)
    traj_l = [io.read(os.path.join(input_dir, file), index=':') for file in file_list]

    # Prepare arguments for multiprocessing
    args = [(traj, save_dir, i) for i, traj in enumerate(traj_l)]

    # Use multiprocessing pool
    with mp.Pool(nproc) as pool:
        results = list(tqdm(pool.imap(process_trajectory, args), total=len(args), desc="Processing trajectories"))
    # Optionally print the results of each process
    for result in results:
        print(result)
