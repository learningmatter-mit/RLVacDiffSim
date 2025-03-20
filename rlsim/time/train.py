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
import numpy as np
from rgnn.common.registry import registry
from rgnn.graph.utils import batch_to
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts, MultiStepLR,
                                      ReduceLROnPlateau, StepLR)
from torch_geometric.loader import DataLoader

from rlsim.time.misc import (combined_loss, combined_loss_binary, CustomSampler)

BEST_LOSS = 1e18


class TimeTrainer:
    def __init__(self, task, logger, config):
        self.task = task
        self.logger = logger
        self.config = config

        self.train_config = self.config["train"]
        # Convert to list of dictionaries
        dqn_model_path = self.config["train"].pop("dqn_model_path")
        t_model_config = self.config["model"]
        dqn_model = registry.get_model_class("dqn_v2").load(f"{dqn_model_path}")
        reaction_model = dqn_model.reaction_model
        self.t_model_name = t_model_config.pop("@name")
        self.t_model = registry.get_model_class(self.t_model_name).load_representation(reaction_model, **t_model_config)
        self.logger.info(f"Training timescape estimator with {self.t_model_name}")
        self.t_model_offline = registry.get_model_class(self.t_model_name).load_representation(reaction_model, **t_model_config)
        self.optimizer = Adam(self.t_model.parameters(), lr=self.train_config["lr"])
        self.scheduler = ReduceLROnPlateau(optimizer=self.optimizer, mode="min", factor=0.5, threshold=1e-2, threshold_mode="abs", patience=5)
        # Dataset
        dataset_path = self.train_config.pop('save_dataset_path')
        current_state_file = dataset_path['state']
        next_state_file = dataset_path['next_state']

        self.logger.info("Load Dataset...")
        self.train_dataset = torch.load(current_state_file["train"])
        self.train_dataset_next = torch.load(next_state_file["train"])
        self.val_dataset = torch.load(current_state_file["val"])
        self.val_dataset_next = torch.load(next_state_file["val"])
        assert len(self.train_dataset) == len(self.train_dataset_next), "Train dataset and next dataset must have the same length"
        self.logger.info(f"Train dataset size: {len(self.train_dataset)}")
        assert len(self.val_dataset) == len(self.val_dataset_next), "Train dataset and next dataset must have the same length"
        self.logger.info(f"Val dataset size: {len(self.val_dataset)}")
        goal_state_count = 0
        for data in self.train_dataset:
            if data["time"] == 0:
                goal_state_count += 1
        self.logger.info(f"Goal state count: {goal_state_count} / {len(self.train_dataset)}")
        toml.dump(self.config, open(f"{self.task}/config_copied.toml", "w"))

        with open(f"{self.task}/loss.txt", 'w') as file:
            file.write('Epoch\t Loss\n')
            
        with open(f"{self.task}/val_loss.txt", 'w') as file:
            file.write('Epoch\t Loss\n')

        self.val_dataloader = DataLoader(self.val_dataset,
                                         batch_size=self.train_config["batch_size"],
                                         shuffle=False)
        self.val_dataloader_next = DataLoader(self.val_dataset_next,
                                              batch_size=self.train_config["batch_size"],
                                              shuffle=False)

    def train(self, best_loss=BEST_LOSS):

        self.logger.info(f"Working directory: {os.path.realpath(self.task)}")

        self.t_model.train()
        self.t_model_offline.eval()
        self.t_model.to(self.train_config["device"])
        self.t_model_offline.to(self.train_config["device"])
        self.logger.info("Start Training...")
        for epoch in range(self.train_config["epoch"]):
            indices_train = np.arange(len(self.train_dataset))
            np.random.shuffle(indices_train)
            sampler = CustomSampler(indices_train)
            loader_dset = DataLoader(self.train_dataset, batch_size=self.train_config["batch_size"], sampler=sampler, shuffle=False)
            loader_dset_next = DataLoader(self.train_dataset_next, batch_size=self.train_config["batch_size"], sampler=sampler, shuffle=False)
            record = 0.
            Nstep = 0
            for i, (batch, next_batch) in enumerate(zip(loader_dset, loader_dset_next)):
                self.optimizer.zero_grad()
                batch = batch_to(batch, self.train_config["device"])
                next_batch = batch_to(next_batch, self.train_config["device"])
                pred = self.t_model(batch)
                gamma = torch.exp(-batch["time"] / self.t_model.tau)
                term1 = self.t_model.tau0 * (1-gamma)
                success = (batch["time"] == 0) # Goal state 1 is goal state 0 is not goal state
                sucess_next = (batch["next_time"] == 0) # Goal state 1 is goal state 0 is not goal state
                goal_states = torch.tensor(1, dtype=term1.dtype, device=term1.device) * success
                next_out = self.t_model_offline(next_batch)["time"]
                next_time = next_out * (~sucess_next)
                label0 = gamma * next_time + term1
                label = label0 * (~success)
                if self.t_model_name == "t_net":
                    loss = combined_loss(pred, 
                                         label.detach(), 
                                         goal_states.detach(), 
                                         omega_g=self.train_config["omega_g"],
                                         omega_t=self.train_config.get("omega_t", 1.0),
                                         )
                elif self.t_model_name == "t_net_binary":
                    loss = combined_loss_binary(pred, 
                                                label.detach(), 
                                                goal_states.detach(), 
                                                omega_g=self.train_config["omega_g"],
                                                omega_t=self.train_config.get("omega_t", 1.0),
                                                omega_cls=self.train_config.get("omega_cls", 1.0), 
                                                )
                record += loss.detach().cpu()
                Nstep += 1
                self.logger.info(f"Batch: {i} / {len(loader_dset)-1}, Loss: {record/Nstep:.4f}")
                loss.backward()
                self.optimizer.step()
                del batch
                del next_batch
                torch.nn.utils.clip_grad_norm_(self.t_model.parameters(), 1.2)
            del loader_dset
            del loader_dset_next
            torch.cuda.empty_cache()
            self.logger.info(f"Epoch: {epoch} / {self.train_config['epoch']}, Loss: {float(record/Nstep):.4f}")
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
            for batch, next_batch in zip(self.val_dataloader, self.val_dataloader_next):
                batch = batch_to(batch, self.train_config["device"])
                next_batch = batch_to(next_batch, self.train_config["device"])
                pred = self.t_model(batch)
                gamma = torch.exp(-batch["time"] / self.t_model.tau)
                term1 = self.t_model.tau0 * (1-gamma)
                success = (batch["time"] == 0) # Goal state
                sucess_next = (batch["next_time"] == 0) # Goal state
                goal_states = torch.tensor(1, dtype=term1.dtype, device=term1.device) * success
                next_out = self.t_model_offline(next_batch)["time"]
                next_time = next_out * (~sucess_next)
                label0 = gamma * next_time + term1
                label = label0 * (~success)
                if self.t_model_name == "t_net":
                    loss = combined_loss(pred, 
                                         label.detach(), 
                                         goal_states.detach(), 
                                         omega_g=self.train_config["omega_g"],
                                         omega_t=self.train_config.get("omega_t", 1.0),
                                         )
                elif self.t_model_name == "t_net_binary":
                    loss = combined_loss_binary(pred, 
                                                label.detach(), 
                                                goal_states.detach(), 
                                                omega_g=self.train_config["omega_g"],
                                                omega_t=self.train_config.get("omega_t", 1.0),
                                                omega_cls=self.train_config.get("omega_cls", 1.0), 
                                                )
                record += loss.detach().cpu()
                Nstep += 1
        self.logger.info(f"Epoch: {epoch} / {self.train_config['epoch']}, VAL Loss: {float(record/Nstep):.4f}")
        with open(f"{self.task}/val_loss.txt", 'a') as file:
            file.write(str(epoch)+'\t'+str(float(record/Nstep))+'\n')
        self.t_model.train()
        return record/Nstep

    def save_model(self, is_best):
        """Save the model to the specified path"""
        self.t_model.save(f"{self.task}/checkpoint.pth.tar")
        if is_best:
            self.t_model.save(f"{self.task}/{self.train_config['save_model_name']}")
