#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:22:58 2024

@author: ubuntu
"""

import json
import os
from itertools import islice

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

from rlsim.time.misc import (combined_loss, combined_loss_binary,
                             get_time_imbalance_sampler, make_t_dataset,
                             preprcess_time)

BEST_LOSS = 1e10


class TimeTrainer:
    def __init__(self, task, logger, config):
        self.task = task
        self.logger = logger
        self.config = config

        self.train_config = self.config["train"]
        dataset_list = self.config["train"].pop("dataset_list")
        dqn_model_path = self.config["train"].pop("dqn_model_path")
        t_model_config = self.config["model"]
        dqn_model = registry.get_model_class("dqn_v2").load(f"{dqn_model_path}")
        reaction_model = dqn_model.reaction_model
        self.t_model_name = t_model_config.pop("@name")
        self.t_model = registry.get_model_class(self.t_model_name).load_representation(reaction_model, **t_model_config)
        self.t_model_offline = registry.get_model_class(self.t_model_name).load_representation(reaction_model, **t_model_config)
        self.optimizer = Adam(self.t_model.parameters(), lr=self.train_config["lr"])
        self.scheduler = ReduceLROnPlateau(optimizer=self.optimizer, mode="min", factor=0.5, threshold=1e-2, threshold_mode="abs", patience=5)
        # Dataset
        dataset_path = self.train_config.pop('dataset_path')
        current_state_file = dataset_path['state']
        next_state_file = dataset_path['next_state']
        if os.path.isfile(current_state_file) and os.path.isfile(next_state_file):
            self.logger.info("Load Dataset")
            dataset = torch.load(current_state_file)
            dataset_next = torch.load(next_state_file)
        else:
            self.logger.info("Make Dataset")
            data_read = []
            temperature_list = []
            for i, dset_path in enumerate(dataset_list):
                with open(dset_path) as file:
                    data = json.load(file)
                    data_read += data
                    for _ in range(len(data)):
                        temperature_list.append(self.train_config["temperature"][i])
            filtered_atoms_final, filtered_next_atoms_final, filtered_time_final, filtered_next_time_final, filtered_temp_final = preprcess_time(data_read, self.train_config, temperature_list)
            dataset, dataset_next = make_t_dataset(filtered_atoms_final, filtered_next_atoms_final, filtered_time_final, filtered_next_time_final, filtered_temp_final, cutoff=self.t_model.cutoff)
            torch.save(dataset, current_state_file)
            torch.save(dataset_next, next_state_file)
        goal_state_count = 0
        for data in dataset:
            if data["time"] == 0:
                goal_state_count += 1
        self.logger.info(f"Goal state count: {goal_state_count} / {len(dataset)}")
        toml.dump(self.config, open(f"{self.task}/config_copied.toml", "w"))

        with open(f"{self.task}/loss.txt", 'w') as file:
            file.write('Epoch\t Loss\n')
            
        with open(f"{self.task}/val_loss.txt", 'w') as file:
            file.write('Epoch\t Loss\n')

        (train_dataset, val_dataset), (train_idx, val_idx) = dataset.train_val_split(train_size=self.train_config["train_size"], return_idx=True)
        train_dataset_next = dataset_next[train_idx]
        val_dataset_next = dataset_next[val_idx]

        if self.train_config.get("sampler", None) == "imbalance":
            self.logger.info("Using imbalance sampler")
            sampler = get_time_imbalance_sampler(train_dataset)
            val_sampler = get_time_imbalance_sampler(val_dataset)
        else:
            self.logger.info("Using random sampler")
            sampler = None
            val_sampler = None
            self.dataloader = DataLoader(train_dataset,
                                         batch_size=self.train_config["batch_size"],
                                         sampler=sampler)
            self.dataloader_next = DataLoader(train_dataset_next,
                                              batch_size=self.train_config["batch_size"],
                                              sampler=sampler)
            self.val_dataloader = DataLoader(val_dataset,
                                             batch_size=self.train_config["batch_size"],
                                             sampler=val_sampler)
            self.val_dataloader_next = DataLoader(val_dataset_next,
                                                  batch_size=self.train_config["batch_size"],
                                                  sampler=val_sampler)

    def train(self, best_loss=BEST_LOSS):
        self.logger.info(f"Training Time model in: {os.path.realpath(self.task)}")

        self.t_model.train()
        self.t_model_offline.eval()
        self.t_model.to(self.train_config["device"])
        self.t_model_offline.to(self.train_config["device"])
        self.logger.info("Start Training")
        for epoch in range(self.train_config["epoch"]):
            record = 0.
            Nstep = 0
            for i, batch in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                batch = batch_to(batch, self.train_config["device"])
                next_batch = batch_to(next(islice(self.dataloader_next, i, None)), self.train_config["device"])
                pred = self.t_model(batch)
                gamma = torch.exp(-batch["time"]/self.t_model.tau)
                term1 = self.t_model.tau*(1-gamma)
                success = (batch["time"] == 0)
                sucess_next = (batch["next_time"] == 0)
                goal_states = torch.tensor(1, dtype=term1.dtype, device=term1.device)*success
                next_out = self.t_model_offline(next_batch, inference=True)["time"]
                next_time = next_out * (~sucess_next)
                label0 = gamma*next_time + term1
                label = label0 * (~success)
                if self.t_model_name == "t_net":
                    loss = combined_loss(pred, label.detach(), goal_states.detach(), t_scaler=self.t_model.scaler, d_scaler=self.t_model.defect_scaler, alpha=self.train_config["alpha"])
                elif self.t_model_name == "t_net_binary":
                    loss = combined_loss_binary(pred, label.detach(), goal_states.detach(), t_scaler=self.t_model.scaler, d_scaler=self.t_model.defect_scaler, alpha=self.train_config["alpha"])

                loss.backward()
                self.optimizer.step()
                torch.nn.utils.clip_grad_norm_(self.t_model.parameters(), 1.2)
                record += loss.detach().cpu()
                Nstep += 1
            torch.cuda.empty_cache()
            val_loss = self.validate(epoch)
            is_best = val_loss <= best_loss
            best_loss = min(val_loss, best_loss)
            self.scheduler.step(val_loss)
            self.save_model(is_best=is_best)

            if epoch % self.train_config["offline_update"] == 0:
                self.t_model_offline.load_state_dict(self.t_model.state_dict())

            with open(f"{self.task}/{self.train_config['loss_filename']}", 'a') as file:
                file.write(str(epoch)+'\t'+str(float(record/Nstep))+'\n')
        self.logger.info("Done...")

    def validate(self, epoch):
        """Validate the current state of the model using the validation set"""
        self.t_model.eval()
        record = 0.
        Nstep = 0
        with torch.no_grad():
            for i, batch in enumerate(self.val_dataloader):
                batch = batch_to(batch, self.train_config["device"])
                next_batch = batch_to(next(islice(self.val_dataloader_next, i, None)), self.train_config["device"])
                pred = self.t_model(batch)
                gamma = torch.exp(-batch["time"]/self.t_model.tau)
                term1 = self.t_model.tau*(1-gamma)
                success = (batch["time"] == 0)
                sucess_next = (batch["next_time"] == 0)
                goal_states = torch.tensor(1, dtype=term1.dtype, device=term1.device)*success
                next_out = self.t_model_offline(next_batch, inference=True)["time"]
                next_time = next_out * (~sucess_next)
                label0 = gamma*next_time + term1
                label = label0 * (~success)
                if self.t_model_name == "t_net":
                    loss = combined_loss(pred, label.detach(), goal_states.detach(), t_scaler=self.t_model.scaler, d_scaler=self.t_model.defect_scaler, alpha=self.train_config["alpha"])
                elif self.t_model_name == "t_net_binary":
                    loss = combined_loss_binary(pred, label.detach(), goal_states.detach(), t_scaler=self.t_model.scaler, d_scaler=self.t_model.defect_scaler, alpha=self.train_config["alpha"])
                record += loss.detach().cpu()
                Nstep += 1
        with open(f"{self.task}/val_loss.txt", 'a') as file:
            file.write(str(epoch)+'\t'+str(float(record/Nstep))+'\n')
        return record/Nstep

    def save_model(self, is_best):
        """Save the model to the specified path"""
        self.t_model.save(f"{self.task}/checkpoint.pth.tar")
        if is_best:
            self.t_model.save(f"{self.task}/{self.train_config['save_model_name']}")
