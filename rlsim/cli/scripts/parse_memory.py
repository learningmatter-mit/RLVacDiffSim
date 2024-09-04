import os

import click
import numpy as np
import torch
from ase import Atoms
from numpy.linalg import norm
from rgnn.graph.dataset.reaction import make_dataset

from rlsim.memory import Memory


def load_data(dataset='training', folder_list=["../data/Vrandom_mace"], traj_n=100):
    traj = Memory(1, 0)
    for foldername in folder_list:
        for i in range(traj_n):
            if dataset == 'training':
                criterion = (i % 3 <= 1)
            elif dataset == 'testing':
                criterion = (i % 3 > 1)
            else:
                raise 'Dataset must be either training or testing'
            filename = f'{foldername}/traj{i}.json'
            print(filename)
            if not os.path.exists(filename):
                raise FileNotFoundError(f'File {filename} not found')
            if criterion:
                traj_list = Memory(1, 0)
                traj_list.load(filename.split(".")[0])
                print(traj_list.E_min, traj_list.rewards)
                traj.rewards += traj_list.rewards[:-1]
                traj.states += traj_list.states[:-1]
                traj.next_states += traj_list.next_states[:-1]
                traj.act_space += traj_list.act_space[:-1]
                traj.actions += traj_list.actions[:-1]
                traj.freq += traj_list.freq[:-1]
                traj.E_min += traj_list.E_min[:-1]
                traj.E_next += traj_list.E_next[:-1]
            else:
                pass

    nframes = len(traj.rewards)
    for u in range(nframes):
        v = nframes - u - 1
        if traj.freq[v] <= 0 or traj.rewards[v] <- 5 or np.abs(traj.E_next[v] - traj.E_min[v]) > 3 or np.isnan(traj.freq[v]) or np.isnan(traj.rewards[v]):
            del traj.freq[v]
            del traj.rewards[v]
            del traj.states[v]
            del traj.next_states[v]
            del traj.act_space[v]
            del traj.actions[v]
            del traj.E_min[v]
            del traj.E_next[v]
    return traj


def set_atoms(atoms, pos: list, convention='frac', slab=False):    
    # set atomic positions in the configuration by fraction coordinate list pos = [\vec x_1, ..., \vec x_N]
    element = atoms.get_atomic_numbers()
    cell = atoms.cell
    pbc = atoms.pbc
    if slab:
        cell[2, 2] *= 1 + 10/norm(cell[2,2])
    if convention == 'frac':
        atoms = Atoms(element, cell=cell, pbc=pbc, scaled_positions=pos);
    else:
        atoms = Atoms(element, cell=cell, pbc=pbc, positions=pos);
    return atoms


def get_atoms(atoms, action):
    act_atom = action[0]
    pos = atoms.get_positions().tolist()
    n_atom = len(pos)
    act_displace = np.array(
        [[0, 0, 0]] * act_atom + [action[1:]] + [[0, 0, 0]] * (n_atom - 1 - act_atom)
    )
    pos = (atoms.get_positions() + act_displace).tolist()
    new_atoms = set_atoms(atoms, pos, convention="cartesion")
    return new_atoms


@click.command()
@click.option('--dataset', type=click.Choice(['training', 'testing']), default='training', help="Specify dataset type: 'training' or 'testing'")
@click.option('--folder_list', type=click.Path(exists=True), multiple=True, help="List of folder paths containing trajectory")
@click.option('--traj-n', type=int, default=100, help="Number of trajectories per run")
@click.option('--save-filename', type=click.Path(), default="traj.json", help="Filename to save trajectory data")
def main(dataset, folder_list, traj_n, save_filename):
    """
    CLI for loading trajectory data.
    """
    traj = load_data(dataset=dataset, folder_list=folder_list, traj_n=traj_n)
    print(f"Loaded trajectory data with {len(traj.rewards)} frames.")
    traj.save(save_filename)
    print(f"Saved trajectory data in {save_filename}.")
    E_r = []
    E_p = []
    Ea = []
    freq = []
    traj_reaction = []
    traj_product = []
    diffuse_atom = []
    for i in range(len(traj.states)):
        traj_reaction.append(traj.states[i])
        next_atoms = get_atoms(traj.states[i], traj.act_space[i][traj.actions[i]])
        traj_product.append(next_atoms)
        E_r.append(traj.E_min[i])
        E_p.append(traj.E_next[i])
        try:
            Ea.append(traj.barrier[i])
        except:
            Ea.append(traj.rewards[i]*-1) # E_s-E_r
        freq.append(traj.freq[i])
        diffuse_atom_idx = traj.act_space[i][traj.actions[i]][0]
        diffuse_atom_number = traj.states[i][diffuse_atom_idx].number
        diffuse_atom.append(diffuse_atom_number)
        if traj.freq[i] == "nan":
            print(f"NaN error: {i}")
        if traj.E_next[i] > 0 or traj.E_min[i] > 0:
            print(f"Weird Energy: {i}")
        if np.abs(traj.E_next[i] - traj.E_min[i]) > 10:
            print(f"Weird Energy: {i}")
    dataset = make_dataset(traj_reaction, traj_product, E_r, E_p, Ea, freq, **{"diffuse_atom": (diffuse_atom, torch.int)})
    torch.save(dataset, f"{save_filename}.pth.tar")