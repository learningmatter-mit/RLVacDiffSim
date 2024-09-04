#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:22:58 2024

@author: ubuntu
"""

import json
import os
import pickle as pkl
import sys
from itertools import islice

import numpy as np
import toml
import torch
from rgnn.common.registry import registry
from rgnn.graph.utils import batch_to
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts, MultiStepLR,
                                      ReduceLROnPlateau, StepLR)
from torch_geometric.loader import DataLoader

sys.path.append("/home2/hojechun/github_repo/RL_H_diffusion")

from rlsim.time.misc import (combined_loss, get_time_imbalance_sampler,
                             make_t_dataset, preprcess_time)


def train_Time(task, logger, config):
    logger.info(f"Training Time model in: {os.path.realpath(task)}")
    toml.dump(config, open(f"{task}/config_copied.toml", "w"))
    train_config = config["train"]
    dataset_list = config["train"].pop("dataset_list")
    dqn_model_path = config["train"].pop("dqn_model_path")
    t_model_config = config["model"]
    dqn_model = registry.get_model_class("dqn_v2").load(f"{dqn_model_path}")
    reaction_model = dqn_model.reaction_model
    t_model = registry.get_model_class("t_net").load_representation(reaction_model, **t_model_config)
    t_model_offline = registry.get_model_class("t_net").load_representation(reaction_model, **t_model_config)
    optimizer = Adam(t_model.parameters(), lr=train_config["lr"])

    with open(f"{task}/{train_config['loss_filename']}", 'w') as file:
        file.write('Epoch\t Loss\n')

    current_state_file = train_config['dataset_path']['state']
    next_state_file = train_config['dataset_path']['next_state']
    if os.path.isfile(current_state_file) and os.path.isfile(next_state_file):
        logger.info("Load Dataset")
        dataset = torch.load(current_state_file)
        dataset_next = torch.load(next_state_file)
    else:
        logger.info("Make Dataset")
        data_read = []
        temperature_list = []
        for i, dset_path in enumerate(dataset_list):
            with open(dset_path) as file:
                data = json.load(file)
                data_read += data
                for _ in range(len(data)):
                    temperature_list.append(train_config["temperature"][i])
        filtered_atoms_final, filtered_next_atoms_final, filtered_time_final, filtered_next_time_final, filtered_temp_final = preprcess_time(data_read, train_config, temperature_list)
        logger.info("Make Dataset")
        dataset, dataset_next = make_t_dataset(filtered_atoms_final, 
                                               filtered_next_atoms_final, 
                                               filtered_time_final, 
                                               filtered_next_time_final, 
                                               filtered_temp_final, 
                                               cutoff=t_model.cutoff)
        current_state_file = f"{task}/{train_config['dataset_path']['state']}"
        next_state_file = f"{task}/{train_config['dataset_path']['next_state']}"
        torch.save(dataset, current_state_file)
        torch.save(dataset_next, next_state_file)

    if train_config.get("sampler", None) == "imbalance":
        logger.info("Using imbalance sampler")
        sampler = get_time_imbalance_sampler(dataset)
        dataloader = DataLoader(dataset,
                                batch_size=train_config["batch_size"],
                                sampler=sampler)
        next_data_loader = DataLoader(dataset_next,
                                batch_size=train_config["batch_size"],
                                sampler=sampler)
    else:
        logger.info("Using random sampler")
        dataloader = DataLoader(dataset,
                                batch_size=train_config["batch_size"],
                                shuffle=False)
        next_data_loader = DataLoader(dataset_next,
                                batch_size=train_config["batch_size"],
                                shuffle=False)
    t_model.train()
    t_model_offline.eval()
    t_model.to(train_config["device"])
    t_model_offline.to(train_config["device"])
    logger.info("Start Training")
    for epoch in range(train_config["epoch"]):
        record = 0.
        Nstep = 0
        pred_list = []
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            batch = batch_to(batch, train_config["device"])
            next_batch = batch_to(next(islice(next_data_loader, i, None)), train_config["device"])
            pred = t_model(batch)
            gamma = torch.exp(-batch["time"]/t_model.tau)
            term1 = t_model.tau*(1-gamma)
            success = (batch["time"]==0)
            sucess_next = (batch["next_time"]==0)
            goal_states = torch.tensor(1, dtype=term1.dtype, device=term1.device)*success
            next_out = t_model_offline(next_batch)
            next_time = next_out * (~sucess_next)
            label0 = gamma*next_time + term1
            label = label0 * (~success)
            loss = combined_loss(pred, label.detach(), goal_states.detach(), t_scaler=t_model.scaler, d_scaler=t_model.defect_scaler, alpha=train_config["alpha"])
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(t_model.parameters(), 1.2)
            record += loss
            Nstep += 1
            pred_list.append(pred.detach())
            del label
            del loss
            del pred
            torch.cuda.empty_cache()
        values = torch.sort(torch.cat(pred_list, dim=0)).values
        write_list = [float(values[int(u1)]) for u1 in np.linspace(0, len(values)-1, 6)]
        logger.info(f"Epoch {epoch} | {write_list}")
        t_model.save(f"{task}/{train_config['save_model_name']}")

        if epoch % train_config["offline_update"]==0:
            t_model_offline.load_state_dict(t_model.state_dict())

        with open(train_config["loss_filename"], 'a') as file:
            file.write(str(epoch)+'\t'+str(float(record/Nstep))+'\n')
