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
        

@click.command()
@click.option("-i", "--input_dir", required=True, type=click.Path(exists=True), help="Input dir containing diffussion info")
@click.option("-n", "--num_files", required=True, type=int, help="numer of XDATCAR files")
@click.option("-s", "--save_dir", required=True, default="./", help="Directory to save the results")
def main(input_dir, num_files, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    traj_l = []
    for i in tqdm(range(num_files), desc="Reading trajectories"):
        filename = os.path.join(input_dir, f"XDATCAR{i}")
        traj = io.read(filename, index=':')
        # process_trajectory((traj, save_dir, i))
        traj_l.append(traj)

    # Prepare arguments for multiprocessing
    args = [(traj, save_dir, i) for i, traj in enumerate(traj_l)]
    # Use multiprocessing pool
    with mp.Pool(num_files) as pool:
        list(tqdm(pool.imap(process_trajectory, args), total=len(traj_l), desc="Processing trajectories"))

