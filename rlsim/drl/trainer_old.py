# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:40:00 2022

@author: 17000
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ase import Atoms
from rgnn.graph.dataset.reaction import ReactionDataset
from rgnn.graph.reaction import ReactionGraph
from rgnn.graph.utils import batch_to
from rgnn.models.dqn import ReactionDQN
from rgnn.train.loss import WeightedSumLoss
from torch_geometric.loader import DataLoader


class ContextBandit:
    def __init__(
        self,
        model: ReactionDQN,
        logger,
        lr=10**-3,
        eps_clip=0.01,
        temperature=300,
    ):
        self.policy_value_net = model
        self.logger = logger
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), lr=lr)
        self.eps_clip = eps_clip
        self.kT = temperature * 8.617 * 10**-5

    def select_action(self, state, action_space, device="cuda"):
        self.policy_value_net.to(device)
        self.policy_value_net.eval()
        dataset_list = convert(state, action_space)
        dataset = ReactionDataset(dataset_list)
        total_q_list = []
        dataloader = DataLoader(dataset, batch_size=16)
        with torch.no_grad():
            for batch in dataloader:
                batch = batch_to(batch, device)
                rl_q = self.policy_value_net.get_q(
                    batch, kT=self.kT, alpha=1.0, beta=0.0, dqn=False
                )["rl_q"]
                total_q_list.append(rl_q.detach())
        Q = torch.concat(total_q_list, dim=-1)
        action_probs = nn.Softmax(dim=0)(Q)
        action = np.random.choice(
            len(action_probs.detach().cpu().numpy()),
            p=action_probs.detach().cpu().numpy(),
        )

        return action, action_probs, Q

    def update(self, memory_l, steps, batch_size=8, device="cuda"):
        self.policy_value_net.to(device)
        self.policy_value_net.train()
        ave_loss = 0
        loss_fn = WeightedSumLoss(
            keys=("q0", "q1"),
            weights=(1.0, 1.0),
            loss_fns=("mse_loss", "mse_loss"),
        )
        # loss_fn = WeightedSumLoss(keys=("q0", "q1"),weights=(1.0, self.kT**2), loss_fns=("mse_loss", "mse_loss"))

        for m in range(steps[0]):
            prob = [0.99 ** (len(memory_l) - i) for i in range(len(memory_l))]
            randint = np.random.choice(range(len(memory_l)), p=prob / np.sum(prob))
            memory = memory_l[randint]
            states = memory.states
            aspace = memory.act_space
            actions = memory.actions
            taken_actions = [[aspace[i][actions[i]]] for i in range(len(aspace))]
            rewards = torch.tensor(memory.rewards, dtype=torch.float)
            freq = torch.tensor(memory.freq, dtype=torch.float)
            delta_e = torch.tensor(
                [val - memory.E_min[i] for i, val in enumerate(memory.E_next)],
                dtype=torch.float,
            )

            total_dataset_list = []
            for i, state in enumerate(states):
                dataset_list = convert(state, taken_actions[i])
                total_dataset_list += dataset_list

            total_dataset_added_list = []
            for i, data in enumerate(total_dataset_list):
                data.q1 = torch.tensor([freq[i]])
                data.q0 = torch.tensor([rewards[i]])
                data.delta_e = torch.tensor([delta_e[i]])
                data.freq = torch.tensor([freq[i]])
                data.barrier = torch.tensor([-rewards[i]])
                data.q0 = torch.tensor([rewards[i]])
                data.q1 = torch.tensor([freq[i]])
                data.rl_q = torch.tensor([rewards[i] + freq[i] * self.kT])
                total_dataset_added_list.append(data)

            total_dataset = ReactionDataset(total_dataset_added_list)

            q_dataloader = DataLoader(total_dataset, batch_size=batch_size)
            for epoch in range(steps[1]):
                for batch in q_dataloader:
                    batch = batch_to(batch, device)
                    output = self.policy_value_net.get_q(
                        batch, kT=self.kT, alpha=1.0, beta=0.0, dqn=False
                    )
                    q_loss = loss_fn(output, batch)
                    self.optimizer.zero_grad()
                    q_loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(
                        self.policy_value_net.parameters(), 1.0
                    )
                    self.optimizer.step()
                ave_loss += q_loss.detach().item()
                self.logger.info(f"* Epoch {epoch},Q loss: {q_loss.detach().item()}")

        return ave_loss / steps[0] / steps[1]


class Q_trainer:
    def __init__(
        self,
        model: ReactionDQN,
        logger,
        q_params: Dict[str, Dict[str, float] | bool | float],
        target_model: ReactionDQN | None = None,
        lr=10**-3,
        train_all=False,
    ):
        self.policy_value_net = model
        self.target_net = target_model
        self.logger = logger
        if not train_all:
            for name, param in self.policy_value_net.named_parameters():
                if "reaction_model" in name:
                    param.requires_grad = False

        trainable_params = filter(
            lambda p: p.requires_grad, self.policy_value_net.parameters()
        )
        # self.reward_normalizer = DynamicNormalizer()
        self.optimizer = optim.Adam(trainable_params, lr=lr)
        self.q_params = q_params
        self.kT = q_params["temperature"] * 8.617 * 10**-5

    def update_T(self, temperature):
        self.kT = temperature * 8.617 * 10**-5
        if self.alpha != 0.0:
            self.q_kT = temperature * 8.617 * 10**-5

    def select_action(self, state, action_space, device="cuda"):
        self.policy_value_net.to(device)
        self.policy_value_net.eval()
        dataset_list = convert(state, action_space)
        dataset = ReactionDataset(dataset_list)
        total_q_list = []
        dataloader = DataLoader(dataset, batch_size=16)
        with torch.no_grad():
            for batch in dataloader:
                batch = batch_to(batch, device)
                rl_q = self.policy_value_net(batch, q_params=self.q_params)["rl_q"]
                total_q_list.append(rl_q.detach())
        Q = torch.concat(total_q_list, dim=-1)
        action_probs = nn.Softmax(dim=0)(Q/(self.q_params['temperature']*8.617*10**-5))
        action = np.random.choice(
            len(action_probs.detach().cpu().numpy()),
            p=action_probs.detach().cpu().numpy(),
        )

        return action, action_probs, Q

    def get_max_Q(self, model, state, action_space, device="cuda"):
        dataset_list = convert(state, action_space)
        dataset = ReactionDataset(dataset_list)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        q_values_list = []
        with torch.no_grad():
            model.eval()
            model.to(device)
            for batch in dataloader:
                batch = batch_to(batch, device)
                q = model(batch, q_params=self.q_params)["rl_q"]
                q_values_list.append(q.detach())
        del dataset
        next_q = torch.cat(q_values_list, dim=0)
        max_q = torch.max(next_q)
        return max_q

    def update(self, memory_l, gamma, steps, size=10, batch_size=8, device="cuda"):
        self.target_net.load_state_dict(self.policy_value_net.state_dict())
        ave_loss = 0
        for m in range(steps[0]):
            randint = np.random.randint(len(memory_l), size=size)
            states, next_states, taken_actions, rewards, next_aspace = (
                [],
                [],
                [],
                [],
                [],
            )
            for u in randint:
                memory = memory_l[u]
                states += memory.states[:-1]
                next_states += memory.next_states[:-1]
                rewards += memory.rewards[:-1]
                aspace = memory.act_space
                actions = memory.actions
                taken_actions += [
                    [aspace[i][actions[i]]] for i in range(len(aspace) - 1)
                ]
                next_aspace += [aspace[i] for i in range(1, len(aspace))]
            rewards = torch.tensor(rewards, dtype=torch.float)
            next_Q = torch.zeros(len(next_aspace))

            for i, state in enumerate(next_states):
                max_q = self.get_max_Q(
                    self.target_net, state, next_aspace[i], device=device
                )
                next_Q[i] = max_q
            dataset_list = []
            for i, state in enumerate(states):
                dataset_list += convert(state, taken_actions[i])
            dataset = ReactionDataset(dataset_list)
            dataset_added_list = []
            for i, data in enumerate(dataset):
                data.rl_q = next_Q[i] * gamma + rewards[i]
                # data.q0 = memory_q0[i]
                dataset_added_list.append(data)
            dataset_added = ReactionDataset(dataset_added_list)
            q_dataloader = DataLoader(
                dataset_added, batch_size=batch_size, shuffle=False
            )
            self.policy_value_net.train()
            for epoch in range(steps[1]):
                for batch in q_dataloader:
                    batch = batch_to(batch, device)
                    q_pred = self.policy_value_net(batch, q_params=self.q_params)[
                        "rl_q"
                    ]
                    self.logger.info(
                        batch["rl_q"].detach(), q_pred.detach()
                    )  # TODO: Debug
                    loss = torch.mean((batch["rl_q"] - q_pred) ** 2)
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(
                        self.policy_value_net.parameters(), max_norm=0.8
                    )
                    self.optimizer.step()
                ave_loss += loss.detach().item()
            if device == "cuda":
                torch.cuda.empty_cache()

        return ave_loss / steps[0] / steps[1]


def convert(atoms: Atoms, actions: List[List[float]]) -> List[ReactionGraph]:
    traj_reactant = []
    traj_product = []
    for act in actions:
        traj_reactant.append(atoms)
        final_atoms = atoms.copy()
        final_positions = []
        for i, pos in enumerate(final_atoms.get_positions()):
            if i == act[0]:
                new_pos = pos + act[1:]
                final_positions.append(new_pos)
            else:
                final_positions.append(pos)
        final_atoms.set_positions(final_positions)
        traj_product.append(final_atoms)
    dataset_list = []
    for i in range(len(traj_reactant)):
        data = ReactionGraph.from_ase(traj_reactant[i], traj_product[i])
        dataset_list.append(data)
    # dataset = ReactionDataset(dataset_list)

    return dataset_list


def select_action(model, state, action_space, q_params, device="cuda"):
    model.to(device)
    model.eval()
    dataset_list = convert(state, action_space)
    dataset = ReactionDataset(dataset_list)
    total_q_list = []
    dataloader = DataLoader(dataset, batch_size=16)
    with torch.no_grad():
        for batch in dataloader:
            batch = batch_to(batch, device)
            rl_q = model(batch, q_params=q_params)["rl_q"]
            total_q_list.append(rl_q.detach())
    Q = torch.concat(total_q_list, dim=-1)
    action_probs = nn.Softmax(dim=0)(Q/(q_params['temperature']*8.617*10**-5))
    action = np.random.choice(
        len(action_probs.detach().cpu().numpy()),
        p=action_probs.detach().cpu().numpy(),
    )

    return action, action_probs, Q
