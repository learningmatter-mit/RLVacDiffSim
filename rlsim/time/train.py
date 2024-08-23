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
from copy import deepcopy
from itertools import islice

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from ase import Atoms
from rgnn.common.registry import registry
from rgnn.graph.utils import batch_to
from torch import nn
from torch.nn import MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts, MultiStepLR,
                                      ReduceLROnPlateau, StepLR)
from torch_geometric.loader import DataLoader

sys.path.append("/home2/hojechun/github_repo/RL_H_diffusion")

from rlsim.time.misc import (get_time_imbalance_sampler, make_t_dataset,
                             preprcess_time)


def train_T(dataset_path, dqn_model_path, train_config, t_model_config):
    # q_params = {"temperature": 500};

    data_read = []
    temperature_list = []
    for i, dset_path in enumerate(dataset_path):
        with open(dset_path) as file:
            data = json.load(file)
            data_read += data
            for _ in range(len(data)):
                temperature_list.append(train_config["temperature"][i])
    filtered_atoms_final, filtered_next_atoms_final, filtered_time_final, filtered_next_time_final, filtered_temp_final = preprcess_time(data_read, train_config, temperature_list)

    dqn_model = registry.get_model_class("dqn_v2").load(f"{dqn_model_path}/model/model_trained")
    reaction_model = dqn_model.reaction_model
    t_model = registry.get_model_class("t_net").load_representation(reaction_model, **t_model_config)
    t_model_offline = registry.get_model_class("t_net").load_representation(reaction_model, **t_model_config)
    # t_model_offline = registry.get_model_class("t_net").load_representation(reaction_model, **t_model_config)
    # t_model = registry.get_model_class("t_net").load("t_model_500.pth.tar")
    # for name, params in t_model.named_parameters():
    #     if "representation" in name:
    #         params.requires_grad = False
    #     else:
    #         params.requires_grad = True
    optimizer = Adam(t_model.parameters(), lr = train_config["lr"]);


    with open(train_config["logfilename"], 'w') as file:
        
        file.write('Epoch\t Loss\n');

    if os.path.isfile(train_config["dataset_path"]["state"]) and os.path.isfile(train_config["dataset_path"]["next_state"]):
        dataset = torch.load(train_config["dataset_path"]["state"])
        dataset_next = torch.load(train_config["dataset_path"]["next_state"])
    else:
        print("Make Dataset")
        dataset, dataset_next = make_t_dataset(filtered_atoms_final, filtered_next_atoms_final, filtered_time_final, filtered_next_time_final, filtered_temp_final, cutoff=t_model.cutoff)
        torch.save(dataset, train_config["dataset_path"]["state"])
        torch.save(dataset_next,train_config["dataset_path"]["next_state"])
    print(len(dataset))
    if train_config.get("sampler", None) == "imbalance":
        print("We are using imbalance sampler")
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
            
            gamma = torch.exp(-batch["time"]/t_model.tau);
            term1 = t_model.tau*(1-gamma);
            success = (batch["time"]==0)
            sucess_next = (batch["next_time"]==0)
            # print(success)
            goal_states = torch.tensor(1, dtype=term1.dtype, device=term1.device)*success
            next_out = t_model_offline(next_batch)
            next_time = next_out * (~sucess_next)
            label0 = gamma*next_time + term1;
            label = label0 * (~success);
            # print(label[:10])
            loss = combined_loss(pred, label.detach(), goal_states.detach(), t_scaler=t_model.scaler, d_scaler=t_model.defect_scaler, alpha=train_config["alpha"])
            loss.backward();
            optimizer.step();
            # loss = torch.mean((1+1*success)*(pred - label.detach())**2);
            record += loss;
            Nstep+=1
            pred_list.append(pred.detach())
            del label
            del loss
            del pred
            torch.cuda.empty_cache()
            # torch.nn.utils.clip_grad_norm_(t_model.parameters(), 1.2)
            # scheduler.step(record/Nstep)
        values = torch.sort(torch.cat(pred_list, dim=0)).values;
        write_list = [float(values[int(u1)]) for u1 in np.linspace(0, len(values)-1, 6)];
        print(epoch, write_list);
        t_model.save(train_config["save_model_name"])

        if epoch%train_config["offline_update"]==0:
            t_model_offline.load_state_dict(t_model.state_dict())


        with open(train_config["logfilename"], 'a') as file:
            file.write(str(epoch)+'\t'+str(float(record/Nstep))+'\n');
        if epoch==train_config["epoch"]-1:
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
                # print(success)
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
                      open(f"label_pred_combined_{t_model_config['tau']}_total_0723.pkl", "wb"))


def combined_loss(time_predictions, time_labels,  goal_labels, t_scaler, d_scaler, alpha=1.0):
    is_not_goal_state = (goal_labels == 0)

    total_scaler = t_scaler * d_scaler
    scaled_preds = time_predictions / total_scaler
    scaled_labels = time_labels / total_scaler
    time_loss = torch.mean((scaled_preds[is_not_goal_state]- scaled_labels[is_not_goal_state])**2)
    if len(scaled_preds[~is_not_goal_state]) !=0:
        goal_loss = torch.mean((scaled_preds[~is_not_goal_state]- scaled_labels[~is_not_goal_state])**2)
        # Combine the losses with a weight parameter alpha
        total_loss = (alpha*goal_loss + time_loss) 
    else:
        total_loss = time_loss
    return total_loss



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
                    "logfilename": f"loss_combined_tau_{t_model_config['tau']}_total_0714.txt"}
    # dataset_path = ["../data/dataset_500.json"]
    dataset_path = [f"/home2/hojechun/00-research/14-time_estimation/dataset_{temperature[0]}_new_0.1_0714.json", 
                    f"/home2/hojechun/00-research/14-time_estimation/dataset_{temperature[1]}_new_0.1_0714.json", 
                    f"/home2/hojechun/00-research/14-time_estimation/dataset_{temperature[2]}_new_0.1_0714.json",
                    f"/home2/hojechun/00-research/14-time_estimation/dataset_{temperature[3]}_new_0.1_0714.json",
                    ]
    dqn_path = "/home2/hojechun/github_repo/RL_H_diffusion/dev/Vrandom_DQN_new_sum"
    train_T(dataset_path, dqn_path, train_config, t_model_config)
