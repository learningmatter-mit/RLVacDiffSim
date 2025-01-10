from contextlib import redirect_stdout
from io import StringIO
from typing import Dict, List

import ase
import numpy as np
import torch
import torch.nn as nn
from ase import Atoms, io
from ase.neb import NEB
from ase.optimize import BFGS, FIRE, MDMin
from numpy.linalg import norm
from rgnn.graph.dataset.reaction import ReactionDataset
from rgnn.graph.reaction import ReactionGraph
from rgnn.graph.utils import batch_to
from torch_geometric.loader import DataLoader

from rlsim.actions.action import get_action_space
from rlsim.environment import Environment


class RLSimulator:
    def __init__(self,
                 environment: Environment,
                 model=None,
                 model_params: Dict[str, float | bool] | None = {"alpha": 0.0,
                                                          "beta": 1.0,
                                                          "dqn": True},
                 params: Dict[str, float | bool] = {"cutoff": 4.0,
                                                    "temperature": 900,
                                                    }):
        self.env = environment
        self.model = model
        self.calculator = self.env.get_calculator(**self.env.calc_params)
        model_params.update({"temperature": params.get("temperature", None)})
        self.q_params = model_params
        self.device = self.env.calc_params["device"]

    def select_action(self, action_space, temperature):
        self.update_q_params(**{"temperature": temperature})
        self.model.to(self.device)
        self.model.eval()
        dataset_list = convert_to_graph_list(self.env.atoms, action_space)
        dataset = ReactionDataset(dataset_list)
        total_q_list = []
        dataloader = DataLoader(dataset, batch_size=16)
        with torch.no_grad():
            for batch in dataloader:
                batch = batch_to(batch, self.device)
                rl_q = self.model(batch, q_params=self.q_params)["rl_q"]
                total_q_list.append(rl_q.detach())
        Q = torch.concat(total_q_list, dim=-1)
        action_probs = nn.Softmax(dim=0)(Q/(temperature*8.617*10**-5))
        action = np.random.choice(
            len(action_probs.detach().cpu().numpy()),
            p=action_probs.detach().cpu().numpy(),
        )

        return action, action_probs, Q

    def step(self, temperature=None, random=False):
        action_space = get_action_space(self.env)
        if random:
            act_id = np.random.choice(len(action_space))
            act_probs = np.array([])
        elif not random and temperature is not None:
            act_id, act_probs, _ = self.select_action(action_space, temperature)
        else:
            raise ValueError("Temperature must be provided if random is False")
        action = action_space[act_id]
        info = {
            "act": act_id,
            "act_probs": act_probs.tolist(),
            "act_space": action_space,
            "state": self.env.atoms.copy(),
            "E_min": self.env.potential(),
        }

        E_next, fail = self.env.step(action)
        self.env.normalize_positions()  # TODO: why do we need is this?
        if not fail and self.q_params["alpha"] != 0.0:
            initial_atoms = self.env.initial.copy()
            next_atoms = self.env.atoms.copy()
            E_s, freq, fail = self.env.saddle(
                initial_atoms=initial_atoms, next_atoms=next_atoms, moved_atom=action[0], n_points=8
            )
            info["E_s"], info["log_freq"] = E_s, freq
        else:
            info["E_s"] = 0
            info["log_freq"] = 0

        info["next"], info["fail"], info["E_next"] = self.env.atoms.copy(), fail, E_next
        return info

    def update_q_params(self, **new_q_params):
        self.q_params.update(**new_q_params)

    def run(self,
            horizon,
            logger,
            atoms_traj: str,
            mode: str = 'lss',
            **simulation_params):
        io.write(atoms_traj, self.env.atoms, format="vasp-xdatcar")

        if mode == "lss":
            outputs = self.run_LSS(horizon, atoms_traj, logger, **simulation_params)
        elif mode == "tks":
            outputs = self.run_TKS(horizon, atoms_traj, logger, **simulation_params)
        return outputs

    def run_LSS(self, horizon, atoms_traj, logger, **simulation_params):
        if simulation_params.get("annealing_time", None) is not None:
            T_scheduler = ThermalAnnealing(total_horizon=horizon,
                                           annealing_time=simulation_params["annealing_time"],
                                           T_start=simulation_params["T_start"],
                                           T_end=simulation_params["T_end"])
        Elist = [self.env.atoms.get_positions()[-1].tolist()]
        for tstep in range(horizon):
            if simulation_params.get("annealing_time", None) is not None:
                new_T = T_scheduler.get_temperature(tstep=tstep)
            else:
                new_T = simulation_params["temeperature"]
            self.update_q_params(**{"temperature": new_T})
            action_space = get_action_space(self.env)
            act_id, _, _ = self.select_action(action_space, new_T)
            action = action_space[act_id]
            _, _ = self.env.step(action)
            io.write(atoms_traj, self.env.atoms, format="vasp-xdatcar", append=True)
            energy = self.env.potential()
            Elist.append(energy)
            if tstep % 10 == 0:
                logger.info(
                    f"Step: {tstep}, T: {new_T:.2f}, E: {energy:.3f}"
                )
        return (Elist)

    def run_TKS(self, horizon, atoms_traj, logger, **simulation_params):
        tlist = [0]
        clist = [self.env.atoms.get_positions()[-1].tolist()]
        temperature = simulation_params["temperature"]
        kT = temperature * 8.617 * 10**-5
        for tstep in range(horizon):
            action_space = get_action_space(self.env)
            act_id, _, Q = self.select_action(action_space, temperature)
            action = action_space[act_id]
            _, _ = self.env.step(action)
            io.write(atoms_traj, self.env.atoms, format="vasp-xdatcar", append=True)
            Gamma = float(torch.sum(torch.exp(Q/kT)));
            dt = 1 / Gamma * 10**-6 # 1/(microsecond)
            tlist.append(tlist[-1] + dt)
            clist.append(self.env.atoms.get_positions()[-1].tolist())
            if tstep % 100 == 0:
                logger.info(
                    f"Step: {tstep}, T: {temperature:.2f}, E: {self.env.potential():.3f}"
                )
        return (tlist, clist)


class ThermalAnnealing:
    #TODO: set speed
    def __init__(self, total_horizon, T_start, T_end, annealing_time=None, T_speed=None):
        if T_speed is not None and annealing_time is None:
            self.annealing_time = int((T_start-T_end) / T_speed)
        elif annealing_time is not None and T_speed is None:
            self.annealing_time = annealing_time
        else:
            raise ValueError("Annealing time is not given")
        assert self.annealing_time <= total_horizon
        self.T_start = T_start
        self.T_end = T_end

    def get_temperature(self, tstep):
        if tstep <= self.annealing_time:
            T = self.T_start - (self.T_start - self.T_end) * tstep / self.annealing_time
        else:
            T = self.T_end
        return T


def convert_to_graph_list(atoms: Atoms, actions: List[List[float]]) -> List[ReactionGraph]:
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

