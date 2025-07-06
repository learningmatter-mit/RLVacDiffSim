import json
import os
import pickle as pkl
import sys
from copy import deepcopy
from itertools import islice
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from ase import Atoms
from rgnn.common.registry import registry
from rgnn.graph.atoms import AtomsGraph
from rgnn.graph.dataset.atoms import AtomsDataset
from rgnn.graph.utils import batch_to
from torch import nn
from torch.nn import MSELoss
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts, MultiStepLR,
                                      ReduceLROnPlateau, StepLR)
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.sampler import Sampler
from torch_geometric.loader import DataLoader


def preprocess_time(data, temperature_list, num_dataset: Optional[int] = None):
    """Preprocessing the timescape dataset

    Args:
        data : parsed dataset
        num_dataset : number of dataset
        temperature_list (List): list of temperatures
    terminate: 1 if the simulation is terminated, 0 otherwise
    """
    atoms_list_all = []
    next_list_all = []
    time_list_all = []
    next_time_list_all = []
    temperature_list_all = []
    atoms_list = [Atoms(positions=state['state']['positions'], cell=state['state']['cell'], numbers=state['state']['atomic_numbers'], pbc=[1, 1, 1]) for state in data]

    next_list = [Atoms(positions=state['next']['positions'], cell=state['next']['cell'], numbers=state['next']['atomic_numbers'], pbc=[1, 1, 1]) for state in data]

    time_list = torch.tensor([state['dt']*(1-state['terminate']) for state in data])

    temperature_list = torch.tensor([temp for temp in temperature_list])
    next_time_list = torch.tensor([(1-state['terminate_next']) for state in data])

    atoms_list_all += atoms_list
    next_list_all += next_list
    next_time_list_all.append(next_time_list)
    time_list_all.append(time_list)
    temperature_list_all.append(temperature_list)
    time_list_all = torch.cat(time_list_all, dim=0)
    next_time_list_all = torch.cat(next_time_list_all, dim=0)
    temperature_list_all = torch.cat(temperature_list_all, dim=0)

    indices = np.arange(len(atoms_list_all))
    np.random.shuffle(indices)

    shuffled_time_total = []
    shuffled_atoms_total = []
    shuffled_next_atoms_total = []
    shuffled_temp_total = []
    shuffled_next_time_total = []
    for idx in indices:
        shuffled_next_time_total.append(next_time_list_all[idx].item())
        shuffled_time_total.append(time_list_all[idx].item())
        shuffled_temp_total.append(temperature_list_all[idx].item())
        shuffled_atoms_total.append(atoms_list_all[idx])
        shuffled_next_atoms_total.append(next_list_all[idx])
    if num_dataset is None:
        num_dataset = len(shuffled_atoms_total)
    filtered_atoms_final = shuffled_atoms_total[:num_dataset]
    filtered_next_atoms_final = shuffled_next_atoms_total[:num_dataset]
    filtered_time_final = shuffled_time_total[:num_dataset]
    filtered_next_time_final = shuffled_next_time_total[:num_dataset]
    filtered_temp_final = shuffled_temp_total[:num_dataset]

    return filtered_atoms_final, filtered_next_atoms_final, filtered_time_final, filtered_next_time_final, filtered_temp_final


# TODO: Should be generalized
def get_total_n_atoms(n_atoms):
    if n_atoms < 200:
        total_n_atoms = 108
    elif n_atoms >= 200 and n_atoms < 300:
        total_n_atoms = 256
    elif n_atoms >= 300 and n_atoms < 600:
        total_n_atoms = 500
    elif n_atoms >= 600 and n_atoms < 900:
        total_n_atoms = 864
    return total_n_atoms


def make_t_dataset(atoms_list, atoms_list_next, target_time_list, target_next_time_list,
                   temperature_list, cutoff=5.0) -> AtomsDataset:
    dataset_list = []
    dataset_list_next = []
    # target_p_tensor = pad_tensor_list(target_p_list)
    for i, atoms in enumerate(atoms_list):
        total_n_atoms = get_total_n_atoms(len(atoms))
        data = AtomsGraph.from_ase(atoms,
                                cutoff,
                                read_properties=False,
                                neighborlist_backend="ase",
                                add_batch=True)
        # if len(target_time_list) > 0 and len(temperature_list) > 0:
        data.id = torch.as_tensor(i, dtype=torch.get_default_dtype(), device=data["elems"].device)
        data.T = torch.as_tensor(temperature_list[i],
                        dtype=torch.get_default_dtype(),
                        device=data["elems"].device)
        data.time = torch.as_tensor(target_time_list[i],
                                    dtype=torch.get_default_dtype(),
                                    device=data["elems"].device)
        # (target_time_list[i].clone().detach()).to(dtype=torch.get_default_dtype(), device=data['elems'].device).unsqueeze(0)
        defect = torch.as_tensor((total_n_atoms - len(atoms))/total_n_atoms,
                        dtype=torch.get_default_dtype(),
                        device=data["elems"].device)
        data.defect = defect
        data.next_time = torch.as_tensor(target_next_time_list[i],
                                    dtype=torch.get_default_dtype(),
                                    device=data["elems"].device)
        # data.next_time = (target_next_time_list[i].clone().detach()).to(dtype=torch.get_default_dtype(), device=data['elems'].device).unsqueeze(0)
        # if len(atoms_list_next) > 0 and len(temperature_list) > 0:
        data_next = AtomsGraph.from_ase(atoms_list_next[i],
                                cutoff,
                                read_properties=False,
                                neighborlist_backend="ase",
                                add_batch=True)
        data_next.id = torch.as_tensor(i, dtype=torch.get_default_dtype(), device=data["elems"].device)
        data_next.T = torch.as_tensor(temperature_list[i],
                        dtype=torch.get_default_dtype(),
                        device=data["elems"].device)
        data_next.defect = defect
        dataset_list_next.append(data_next)
        dataset_list.append(data)
        
    dataset = AtomsDataset(dataset_list)
    # if len(atoms_list_next) > 0:
    dataset_next = AtomsDataset(dataset_list_next)
    # else:
    #     dataset_next = None

    return dataset, dataset_next


def combined_loss(prediction, time_labels, goal_labels, tau0, omega_tau, omega_g=1.0, omega_t=1.0):
    # time_predictions = torch.log1p(prediction["time"])
    # time_labels = torch.log1p(time_labels)
    time_predictions = prediction["time"]
    is_not_goal_state = (goal_labels == 0)
    time_loss = torch.mean((time_predictions[is_not_goal_state] - time_labels[is_not_goal_state])**2)
    if isinstance(tau0, (int, float)):
        # convert scalar to tensor and broadcast
        tau0_tensor = torch.full_like(time_predictions, fill_value=tau0)
    else:
        # assume it's already a tensor with correct shape
        tau0_tensor = tau0

    time_loss_tau = torch.mean((tau0_tensor[is_not_goal_state] - time_predictions[is_not_goal_state]) ** 2)

    goal_loss = torch.mean(time_predictions[~is_not_goal_state]**2)
    n_goal = len(time_predictions[~is_not_goal_state])
    n_non_goal = len(time_predictions[is_not_goal_state])

    if n_goal > 0 and n_non_goal > 0:
        # Case 1: goal + non-goal
        total_loss = (omega_g * goal_loss + omega_t * time_loss) + omega_tau * time_loss_tau
    elif n_goal > 0 and n_non_goal == 0:
        # Case 2: only goal
        total_loss = omega_g * goal_loss
    elif n_goal == 0 and n_non_goal > 0:
        # Case 3: only non-goal
        total_loss = omega_t * time_loss + omega_tau * time_loss_tau
    else:
        # Case 4: both empty – potentially invalid state
        raise ValueError("Empty goal and non-goal states. Cannot compute total_loss.")
    return total_loss


def combined_loss_binary(prediction, time_labels, goal_labels, tau0, omega_tau, omega_g=1.0, omega_t=1.0, omega_cls=1.0):
    # time_predictions = torch.log1p(prediction["time"])
    # time_labels = torch.log1p(time_labels)
    time_predictions = prediction["time"]
    is_not_goal_state = (goal_labels == 0)

    time_loss = torch.mean((time_predictions[is_not_goal_state]- time_labels[is_not_goal_state])**2)
    if isinstance(tau0, (int, float)):
        # convert scalar to tensor and broadcast
        tau0_tensor = torch.full_like(time_predictions, fill_value=tau0)
    else:
        # assume it's already a tensor with correct shape
        tau0_tensor = tau0

    time_loss_tau = torch.mean((tau0_tensor[is_not_goal_state] - time_predictions[is_not_goal_state]) ** 2)

    goal_loss = torch.mean(time_predictions[~is_not_goal_state]**2)
    goal_loss_binary = F.binary_cross_entropy_with_logits(prediction["goal"], goal_labels)
    total_loss = omega_g*goal_loss + omega_t*time_loss + omega_cls*goal_loss_binary
    n_goal = len(time_predictions[~is_not_goal_state])
    n_non_goal = len(time_predictions[is_not_goal_state])

    if n_goal > 0 and n_non_goal > 0:
        # Case 1: both present
        total_loss = omega_g * goal_loss + omega_t * time_loss + omega_cls * goal_loss_binary + omega_tau * time_loss_tau
    elif n_goal > 0 and n_non_goal == 0:
        # Case 2: only goal
        total_loss = omega_g * goal_loss + omega_cls * goal_loss_binary
    elif n_goal == 0 and n_non_goal > 0:
        # Case 3: only non-goal
        total_loss = omega_t * time_loss + omega_cls * goal_loss_binary + omega_tau * time_loss_tau
    else:
        # Case 4: both empty – potentially invalid state
        raise ValueError("Empty goal and non-goal states. Cannot compute total_loss.")
    return total_loss

class CustomSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    
