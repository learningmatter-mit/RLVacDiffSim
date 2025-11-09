import random
from contextlib import redirect_stdout
from io import StringIO
from typing import Dict, List, Tuple

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

from rlsim.actions.action import get_action_space, get_action_space_mcmc
from rlsim.environment import Environment
from rlsim.utils.sro import get_sro_from_atoms

ENERGY_DIFF_LIMIT = 1.5  # in eV


class RLSimulator:
    def __init__(self,
                 environment: Environment,
                 model=None,
                 q_params: Dict[str, float | bool] | None = {"alpha": 0.0, "beta": 0.5, "dqn": True},
                 sro_pixel: Tuple[float, float, float, float] | Tuple[float, float] | None = None):
        self.env = environment
        self.calculator = self.env.get_calculator(**self.env.calc_params)
        if model is not None:
            self.model = model
            self.q_params = q_params
        self.device = self.env.calc_params["device"]
        self.kb = 8.617*10**-5
        self.sro_pixel = sro_pixel

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

        if self.sro_pixel is not None:
            valid_actions = []
            atoms = self.env.atoms.copy()
            pos = atoms.get_positions()
            sro_list = []
            for i, action in enumerate(action_space):
                pos[action[0]] += np.array(action[1:])*1/0.8
                atoms.set_positions(pos)
                sro = get_sro_from_atoms(atoms)
                sro_list.append(sro)
                pos[action[0]] -= np.array(action[1:])*1/0.8
                diagonal_sro = np.diag(sro)
                if len(self.sro_pixel) == 4:
                    x0, y0, z0, L = self.sro_pixel
                    Cr_condition = diagonal_sro[0] > x0-L/2 and diagonal_sro[0] < x0+L/2
                    Co_condition = diagonal_sro[1] > y0-L/2 and diagonal_sro[1] < y0+L/2
                    Ni_condition = diagonal_sro[2] > z0-L/2 and diagonal_sro[2] < z0+L/2
                    if Cr_condition and Co_condition and Ni_condition:
                        valid_actions.append(i)
                elif len(self.sro_pixel) == 2:
                    x0, L = self.sro_pixel
                    lower_bound = x0-L/2
                    upper_bound = x0+L/2
                    scalar_sro = np.sqrt(np.sum(sro[0, :]**2) + np.sum(sro[1, 1:]**2) + sro[2, 2]**2)
                    if scalar_sro > lower_bound and scalar_sro < upper_bound:
                        valid_actions.append(i)
      
            valid_actions = torch.tensor(valid_actions, device=self.device)
            Q = Q[valid_actions]
            sro_list = torch.tensor(sro_list, device=self.device)
            sro_list = sro_list[valid_actions]
        else:
            sro_list = None
        
        action_probs = nn.Softmax(dim=0)(Q/(temperature*self.kb))
        action = np.random.choice(
            len(action_probs.detach().cpu().numpy()),
            p=action_probs.detach().cpu().numpy(),
        )

        if self.sro_pixel is not None:
            action = valid_actions[action]

        return action, action_probs, Q, sro_list

    def step(self, temperature=None, random=False):
        action_space = get_action_space(self.env)
        if random:
            act_id = np.random.choice(len(action_space))
            act_probs = np.array([])
        elif not random and temperature is not None:
            act_id, act_probs, _, _ = self.select_action(action_space, temperature)
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
        elif mode == "mcmc":
            outputs = self.run_MCMC(horizon, atoms_traj, logger, **simulation_params)
        elif mode == "tks":
            assert not self.q_params["dqn"], "TKS is only available for dqn==False."
            outputs = self.run_TKS(horizon, atoms_traj, logger, **simulation_params)
        logger.info("Simulation finished.")
        return outputs

    def run_LSS(self, horizon, atoms_traj, logger, **simulation_params):
        if simulation_params.get("annealing_time", None) is not None:
            T_scheduler = ThermalAnnealing(total_horizon=horizon,
                                           annealing_time=simulation_params["annealing_time"],
                                           T_start=simulation_params["T_start"],
                                           T_end=simulation_params["T_end"])
        Elist = [self.env.potential()]
        Qlist = [[]]
        SROlist = [[]]
        for tstep in range(horizon):
            if simulation_params.get("annealing_time", None) is not None:
                new_T = T_scheduler.get_temperature(tstep=tstep)
            else:
                new_T = simulation_params["temperature"]
            action_space = get_action_space(self.env)
            act_id, _, Q, sro_list = self.select_action(action_space, new_T)
            action = action_space[act_id]
            _, _ = self.env.step(action)
            io.write(atoms_traj, self.env.atoms, format="vasp-xdatcar", append=True)
            energy = self.env.potential()
            Elist.append(energy)
            Qlist.append(Q.tolist())
            SROlist.append(sro_list.tolist())
            if tstep % 10 == 0 or tstep == horizon - 1:
                logger.info(
                    f"Step: {tstep}, T: {new_T:.0f}, E: {energy:.3f}"
                )
        last_atoms_filename = atoms_traj.replace("XDATCAR", "last_atoms")
        io.write(last_atoms_filename, self.env.atoms, format="vasp")
        return (Elist, Qlist, SROlist)

    def run_TKS(self, horizon, atoms_traj, logger, **simulation_params):
        tlist = [0]
        clist = [self.env.atoms.get_positions()[-1].tolist()]
        temperature = simulation_params["temperature"]
        kT = temperature * 8.617 * 10**-5
        Elist = [self.env.potential()]
        Qlist = [[]]
        SROlist = [[]]
        for tstep in range(horizon):
            action_space = get_action_space(self.env)
            act_id, _, Q, sro_list = self.select_action(action_space, temperature)
            action = action_space[act_id]
            _, _ = self.env.step(action)
            io.write(atoms_traj, self.env.atoms, format="vasp-xdatcar", append=True)
            Gamma = float(torch.sum(torch.exp(Q/kT)));
            dt = 1 / Gamma * 10**-6  # microsecond unit
            energy = self.env.potential()
            Elist.append(energy)
            tlist.append(tlist[-1] + dt)
            Qlist.append(Q.tolist())
            SROlist.append(sro_list.tolist())
            clist.append(self.env.atoms.get_positions()[-1].tolist())
            if tstep % 10 == 0 or tstep == horizon - 1:
                logger.info(
                    f"Step: {tstep}, T: {temperature:.0f}, E: {self.env.potential():.3f}"
                )
        last_atoms_filename = atoms_traj.replace("XDATCAR", "last_atoms")
        io.write(last_atoms_filename, self.env.atoms, format="vasp")
        return (Elist, Qlist, SROlist, tlist, clist)
    
    def run_MCMC(self, horizon, atoms_traj, logger, **simulation_params):
        logger.info(f"Action mode: {simulation_params.get('action_mode', 'vacancy_only')}")
        if simulation_params.get("annealing_time", None) is not None:
            T_scheduler = ThermalAnnealing(total_horizon=horizon,
                                           annealing_time=simulation_params["annealing_time"],
                                           T_start=simulation_params["T_start"],
                                           T_end=simulation_params["T_end"])
        Elist = [self.env.potential()]
        self.total_mcmc_step = 0
        for tstep in range(horizon):
            if simulation_params.get("annealing_time", None) is not None:
                new_T = T_scheduler.get_temperature(tstep=tstep)
            else:
                new_T = simulation_params["temperature"]
            n_sweeps = simulation_params.get("n_sweeps", 1)
            energy, _, count = self.mcmc_sweep(n_sweeps=n_sweeps, temperature=new_T, action_mode=simulation_params.get("action_mode", "vacancy_only"))
            io.write(atoms_traj, self.env.atoms, format="vasp-xdatcar", append=True)

            Elist.append(energy)
            if tstep % 10 == 0 or tstep == horizon - 1:
                logger.info(
                    f"Step: {tstep} | Sweep: {count}/{n_sweeps}| T: {new_T:.0f} K | E: {energy:.3f} eV"
                )
        last_atoms_filename = atoms_traj.replace("XDATCAR", "last_atoms")
        io.write(last_atoms_filename, self.env.atoms, format="vasp")
        return (Elist,)

    def mcmc_sweep(self, n_sweeps, temperature, action_mode="vacancy_only"):
        accept = False
        count = 0
        while not accept and count < n_sweeps:
            energy, accept = self.mcmc_step(temperature, action_mode=action_mode)
            count += 1
            self.total_mcmc_step += 1
        return energy, accept, count
    
    def mcmc_step(self, temperature, action_mode):
        accept = False
        base_prob = 0.0
        kT = temperature * 8.617 * 10**-5
        action_space = get_action_space_mcmc(self.env, action_mode=action_mode)
        if self.sro_pixel is not None:
            valid_actions = []
            atoms = self.env.atoms.copy()
            pos = atoms.get_positions()
            for i, action in enumerate(action_space):
                pos[action[0]] += np.array(action[1:])*1/0.8
                atoms.set_positions(pos)
                sro = get_sro_from_atoms(atoms)
                pos[action[0]] -= np.array(action[1:])*1/0.8
                diagonal_sro = np.diag(sro)
                if len(self.sro_pixel) == 4:
                    x0, y0, z0, L = self.sro_pixel
                    Cr_condition = diagonal_sro[0] > x0-L/2 and diagonal_sro[0] < x0+L/2
                    Co_condition = diagonal_sro[1] > y0-L/2 and diagonal_sro[1] < y0+L/2
                    Ni_condition = diagonal_sro[2] > z0-L/2 and diagonal_sro[2] < z0+L/2
                    if Cr_condition and Co_condition and Ni_condition:
                        valid_actions.append(i)
                elif len(self.sro_pixel) == 2:
                    x0, L = self.sro_pixel
                    lower_bound = x0-L/2
                    upper_bound = x0+L/2
                    scalar_sro = np.sqrt(np.sum(sro[0, :]**2) + np.sum(sro[1, 1:]**2) + sro[2, 2]**2)
                    if scalar_sro > lower_bound and scalar_sro < upper_bound:
                        valid_actions.append(i)
      
            valid_actions = torch.tensor(valid_actions, device=self.device)
            action_space = [action_space[i] for i in valid_actions]

        action = random.choice(action_space)
        E_prev = self.env.potential()
        E_next, fail = self.env.step(action)

        if not fail:
            energy_diff = (E_next - E_prev) # / self.env.n_atom
            if energy_diff < 0.0:
                base_prob = 1.0  # Automatically accept for sufficiently negative energy diff
            elif np.abs(energy_diff) <= ENERGY_DIFF_LIMIT:
                try:
                    base_prob = np.exp(-energy_diff / kT)
                except OverflowError:
                    pass
        if np.random.rand() < base_prob:
            accept = True
        else:
            self.env.revert()
        energy = self.env.potential()
        return energy, accept


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

