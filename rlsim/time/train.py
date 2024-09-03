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
from rlsim.utils.logger import setup_logger
import toml


def train_Time(dataset_path, dqn_model_path, train_config, t_model_config, logger):
    with open(settings, "r") as f:
        config = toml.load(f)
        task = config["task"]
        logger_config = config["logger"]
        train_config = config["train"]
        dataset_list = config["train"].pop("dataset_list")
        dqn_model_path = config["train"].pop("dqn_model_path")
        t_model_config = config["model"]

    if task not in os.listdir():
        os.makedirs(task, exist_ok=True)
    log_filename = f"{task}/{logger_config['filename']}.log"
    logger = setup_logger(logger_config["name"], log_filename)

    dqn_model = registry.get_model_class("dqn_v2").load(f"{dqn_model_path}/model/model_trained")
    reaction_model = dqn_model.reaction_model
    t_model = registry.get_model_class("t_net").load_representation(reaction_model, **t_model_config)
    t_model_offline = registry.get_model_class("t_net").load_representation(reaction_model, **t_model_config)
    optimizer = Adam(t_model.parameters(), lr = train_config["lr"]);

    with open(f"{task}/{train_config['logfilename']}", 'w') as file:
        file.write('Epoch\t Loss\n');

    current_state_file = f"{task}/{train_config['dataset_path']['state']}"
    next_state_file = f"{task}/{train_config['dataset_path']['next_state']}"
    if os.path.isfile(current_state_file) and os.path.isfile(next_state_file):
        dataset = torch.load(current_state_file)
        dataset_next = torch.load(next_state_)
    else:
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
        torch.save(dataset, current_state_file)
        torch.save(dataset_next, next_state_)

    if train_config.get("sampler", None) == "imbalance":
        logger.info("We are using imbalance sampler")
        sampler = get_time_imbalance_sampler(dataset)
        dataloader = DataLoader(dataset,
                                batch_size=train_config["batch_size"],
                                sampler=sampler)
        next_data_loader = DataLoader(dataset_next,
                                batch_size=train_config["batch_size"],
                                sampler=sampler)
    else:
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
            # logger.info(label[:10])
            loss = combined_loss(pred, label.detach(), goal_states.detach(), t_scaler=t_model.scaler, d_scaler=t_model.defect_scaler, alpha=train_config["alpha"])
            loss.backward()
            optimizer.step()
            # loss = torch.mean((1+1*success)*(pred - label.detach())**2)
            record += loss
            Nstep += 1
            pred_list.append(pred.detach())
            del label
            del loss
            del pred
            torch.cuda.empty_cache()
            # torch.nn.utils.clip_grad_norm_(t_model.parameters(), 1.2)
            # scheduler.step(record/Nstep)
        values = torch.sort(torch.cat(pred_list, dim=0)).values
        write_list = [float(values[int(u1)]) for u1 in np.linspace(0, len(values)-1, 6)];
        logger.info(epoch, write_list);
        t_model.save(f"{task}/{train_config['save_model_name']}")

        if epoch % train_config["offline_update"]==0:
            t_model_offline.load_state_dict(t_model.state_dict())

        with open(train_config["logfilename"], 'a') as file:
            file.write(str(epoch)+'\t'+str(float(record/Nstep))+'\n');
        if epoch == train_config["epoch"]-1:
            pred_list = []
            label_list = []
            term1_list = []
            real_time_list = []
            for i, batch in enumerate(dataloader):
                batch = batch_to(batch, train_config["device"])
                pred = t_model(batch)
                next_batch = batch_to(next(islice(next_data_loader, i, None)), train_config["device"])
                gamma = torch.exp(-batch["time"]/t_model.tau);
                term1 = t_model.tau*(1-gamma);
                success = (batch["time"]==0)
                sucess_next = (batch["next_time"]==0)
                # logger.info(success)
                goal_states = torch.tensor(1, dtype=term1.dtype, device=term1.device)*success
                next_out = t_model(next_batch) 
                next_time = next_out * (~sucess_next)
                label0 = gamma*next_time + term1;
                label = label0 * (~success);
                pred_list.append(pred.detach() / t_model.scaler / t_model.defect_scaler)
                label_list.append(label.detach()/ t_model.scaler / t_model.defect_scaler)
                term1_list.append((term1*(~success)).detach()/ t_model.scaler / t_model.defect_scaler)
                real_time_list.append(batch["time"].detach()/ t_model.scaler / t_model.defect_scaler)
                del label
                del pred
                del batch

            pkl.dump((torch.cat(pred_list, dim=0).cpu().numpy(),
                      torch.cat(label_list, dim=0).cpu().numpy(), 
                      torch.cat(term1_list, dim=0).cpu().numpy(),
                      torch.cat(real_time_list, dim=0).cpu().numpy()), 
                      open(f"{task}/{train_config['label_filename']}", "wb"))


if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    temperature = [300, 500,700, 900]
    t_model_config = {"dropout_rate" : 0.10,
                      "n_feat":64, 
                      "tau": 30,
                      "T_scaler_m":7.918951990135469,
                      "D_scaler": 0.00390625}
    train_config = {"batch_size": 32, 
                    "device": "cuda", 
                    "epoch": 101, 
                    "temperature": temperature, 
                    "save_model_name": f"t_model_combined_0723_{t_model_config['tau']}.pth.tar", 
                    "alpha": 1.0, 
                    # "sampler": "imbalance",
                    "lr": 1e-5,
                    "num_dataset": 6000,
                    "offline_update": 10,
                    "dataset_path": {"state": "t_dataset_0714_combined_total.pth.tar", "next_state": "t_next_dataset_0714_combined_total.pth.tar"},
                    "logfilename": f"loss_combined_tau_{t_model_config['tau']}_total_0714.txt",
                    "label_filename": f"label_pred_combined_{t_model_config['tau']}_total_0714.pkl",}
    # dataset_path = ["../data/dataset_500.json"]
    dataset_path = [f"/home2/hojechun/00-research/14-time_estimation/dataset_{temperature[0]}_new_0.1_0714.json", 
                    f"/home2/hojechun/00-research/14-time_estimation/dataset_{temperature[1]}_new_0.1_0714.json", 
                    f"/home2/hojechun/00-research/14-time_estimation/dataset_{temperature[2]}_new_0.1_0714.json",
                    f"/home2/hojechun/00-research/14-time_estimation/dataset_{temperature[3]}_new_0.1_0714.json",
                    ]
    dqn_path = "/home2/hojechun/github_repo/RL_H_diffusion/dev/Vrandom_DQN_new_sum"
    logger = setup_logger("TIME Train", "logger.log")

    train_Time(dataset_path, dqn_path, train_config, t_model_config, logger)
