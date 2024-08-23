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
                 model,
                 params: Dict[str, float | bool] = {"cutoff": 4.0,
                                                    "temperature": 900,
                                                    "alpha": 0.0,
                                                    "beta": 1.0,
                                                    "dqn": True
                                                    }):
        self.env = environment
        self.model = model
        self.calculator = self.env.get_calculator(**self.env.calc_params)
        # self.max_iter = params["max_iter"]
        # self.pos = self.env.positions("cartesion").tolist()
        # self.cell = self.env.atoms.cell.tolist()
        # self.n_atom = len(self.pos)
        # self.cutoff = params["cutoff"]
        # self.output = StringIO()
        # self.relax_log = params["relax_log"]
        self.relax_accuracy = params["relax_accuracy"]
        self.q_params = {"temperature": params["temperature"],
                         "alpha": params["alpha"],
                         "beta": params["beta"],
                         "dqn": params["dqn"]}
        self.device = self.env.calc_params["device"]

    # def relax(self, accuracy=0.05):
    #     self.env.atoms.set_constraint(
    #         ase.constraints.FixAtoms(mask=[False] * self.n_atom)
    #     )
    #     dyn = MDMin(self.env.atoms, logfile=self.relax_log)
    #     with redirect_stdout(self.output):
    #         converge = dyn.run(fmax=accuracy, steps=self.max_iter)
    #     self.pos = self.env.atoms.get_positions().tolist()

    #     return converge

    # def step(self, action: list = [], accuracy=0.05):
    #     self.action = action
    #     self.pos_last = self.env.positions(f="cartesion")
    #     self.initial = self.env.atoms.copy()

    #     self.act_atom = self.action[0]
    #     self.act_displace = np.array(
    #         [[0, 0, 0]] * self.act_atom
    #         + [self.action[1:]]
    #         + [[0, 0, 0]] * (self.n_atom - 1 - self.act_atom)
    #     )
    #     self.pos = (self.env.positions(f="cartesion") + self.act_displace).tolist()
    #     self.env.set_atoms(self.pos, convention="cartesion")

    #     converge = self.relax(accuracy=accuracy)
    #     fail = int(norm(self.pos_last - self.env.positions(f="cartesion")) < 0.2) + (
    #         not converge
    #     )
    #     E_next = self.env.potential()

    #     return E_next, fail

    # def revert(self):
    #     self.env.set_atoms(self.pos_last, convention="cartesion")

    # def mask(self):
    #     dist = self.env.atoms.get_distances(range(self.n_atom), self.act_atom)
    #     mask = [d > self.cutoff for d in dist]

    #     return mask

    # def normalize_positions(self):
    #     pos_frac = self.env.atoms.get_scaled_positions() % 1
    #     self.env.atoms.set_scaled_positions(pos_frac)
    #     return 0

    # def saddle(
    #     self, moved_atom=-1, accuracy=0.1, n_points=10, r_cut=4
    # ):
    #     self.env.atoms.set_constraint(
    #         ase.constraints.FixAtoms(mask=[False] * self.n_atom)
    #     )
    #     self.initial.set_constraint(
    #         ase.constraints.FixAtoms(mask=[False] * self.n_atom)
    #     )
    #     images = [self.initial]
    #     images += [self.initial.copy() for i in range(n_points - 2)]
    #     images += [self.env.atoms]

    #     neb = NEB(images)
    #     neb.interpolate()

    #     for image in range(n_points):
    #         images[image].calc = self.env.get_calculator(**self.env.calc_params)
    #         images[image].set_constraint(ase.constraints.FixAtoms(mask=self.mask()))
    #     with redirect_stdout(self.output):
    #         optimizer = MDMin(neb, logfile=self.relax_log)

    #     res = optimizer.run(fmax=accuracy, steps=self.max_iter)
    #     res = True
    #     if res:
    #         Elist = [image.get_potential_energy() for image in images]
    #         E = np.max(Elist)

    #         def log_niu_prod(input_atoms, NN, saddle=False):
    #             delta = 0.05

    #             mass = input_atoms.get_masses()[NN]
    #             mass = np.array([mass[i // 3] for i in range(3 * len(NN))])
    #             mass_mat = np.sqrt(mass[:, None] * mass[None, :])
    #             Hessian = np.zeros([len(NN) * 3, len(NN) * 3])
    #             f0 = input_atoms.get_forces()
    #             pos_init = input_atoms.get_positions()
    #             for u in range(len(NN)):
    #                 for j in range(3):
    #                     pos1 = pos_init.copy()
    #                     pos1[NN[u]][j] += delta
    #                     input_atoms.set_positions(pos1)
    #                     Hessian[3 * u + j] = (
    #                         -(input_atoms.get_forces() - f0)[NN].reshape(-1) / delta
    #                     )

    #             freq_mat = (Hessian + Hessian.T) / 2 / mass_mat
    #             if saddle:
    #                 prod = np.prod(np.linalg.eigvals(freq_mat)[1:])
    #             else:
    #                 prod = np.linalg.det(freq_mat)
    #             output = np.log(np.abs(prod)) / 2

    #             return output

    #         max_ind = np.argmax(Elist)
    #         r_cut = 3
    #         NN = self.initial.get_distances(
    #             moved_atom, range(len(self.initial)), mic=True
    #         )
    #         NN = np.argwhere(NN < r_cut).T[0]
    #         self.initial.calc = self.env.get_calculator(**self.env.calc_params)
    #         self.initial.set_constraint(
    #             ase.constraints.FixAtoms(mask=[False] * self.n_atom)
    #         )
    #         images[max_ind].calc = self.env.get_calculator(**self.env.calc_params)
    #         images[max_ind].set_constraint(
    #             ase.constraints.FixAtoms(mask=[False] * self.n_atom)
    #         )
    #         log_niu_min = log_niu_prod(self.initial, NN)
    #         log_niu_s = log_niu_prod(images[max_ind], NN, saddle=True)

    #         log_attempt_freq = log_niu_min - log_niu_s + np.log(1.55716 * 10)

    #     else:
    #         E = 0
    #         log_attempt_freq = 0
    #     self.env.atoms.set_constraint(
    #         ase.constraints.FixAtoms(mask=[False] * self.n_atom)
    #     )

    #     pos_frac = self.env.atoms.get_scaled_positions() % 1
    #     self.env.atoms.set_scaled_positions(pos_frac)

    #     return E, log_attempt_freq, int(not res)

    def select_action(self, action_space):
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
        action_probs = nn.Softmax(dim=0)(Q/(self.q_params['temperature']*8.617*10**-5))
        action = np.random.choice(
            len(action_probs.detach().cpu().numpy()),
            p=action_probs.detach().cpu().numpy(),
        )

        return action, action_probs, Q

    def update_q_params(self, **new_q_params):
        self.q_params.update(**new_q_params)

    def run(self,
            horizon,
            logger,
            atoms_traj: str,
            mode: str = 'lss',
            **input_params):
        io.write(atoms_traj, self.env.atoms, format="vasp-xdatcar")

        if mode == "lss":
            outputs = self.run_LSS(horizon, atoms_traj, logger, **input_params)
        elif mode == "tks":
            outputs = self.run_TKS(horizon, atoms_traj, logger, **input_params)
        return outputs

    def run_LSS(self, horizon, atoms_traj, logger, **input_params):
        T_scheduler = ThermalAnnealing(total_horizon=horizon,
                                       annealing_time=input_params["annealing_time"], 
                                       T_start=input_params["T_start"], 
                                       T_end=input_params["T_end"])
        Elist = [self.env.atoms.get_positions()[-1].tolist()]
        for tstep in range(horizon):
            new_T = T_scheduler.get_temperature(tstep=tstep)
            self.update_q_params(**{"temperature": new_T})
            action_space = get_action_space(self.env)
            act_id, _, _ = self.select_action(action_space)
            action = action_space[act_id]
            _, _ = self.env.step(action, accuracy=self.relax_accuracy)
            io.write(atoms_traj, self.env.atoms, format="vasp-xdatcar", append=True)
            energy = self.env.potential()
            Elist.append(energy)
            if tstep % 10 == 0:
                logger.info(
                    f"Step: {tstep}, T: {self.q_params['temperature']:.2f}, E: {energy:.3f}"
                )
        return (Elist)

    def run_TKS(self, horizon, atoms_traj, logger, **input_params):
        tlist = [0]
        clist = [self.env.atoms.get_positions()[-1].tolist()]
        kT = self.q_params["temperature"] * 8.617 * 10**-5
        for tstep in range(horizon):
            action_space = get_action_space(self.env)
            act_id, _, Q = self.select_action(action_space)
            action = action_space[act_id]
            _, _ = self.env.step(action, accuracy=self.relax_accuracy)
            io.write(atoms_traj, self.env.atoms, format="vasp-xdatcar", append=True)
            Gamma = float(torch.sum(torch.exp(Q/kT)));
            dt = 1 / Gamma * 10**-6
            tlist.append(tlist[-1] + dt)
            clist.append(self.env.atoms.get_positions()[-1].tolist())
            if tstep % 100 == 0:
                logger.info(
                    f"Step: {tstep}, T: {self.q_params['temperature']:.2f}, E: {self.env.potential():.3f}"
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

