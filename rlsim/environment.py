# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 17:32:08 2022

@author: 17000
"""
import os
from contextlib import redirect_stdout
from io import StringIO
from typing import Dict, List, Tuple

import ase
import numpy as np
from ase.mep import NEB
from ase.optimize import BFGS, FIRE, MDMin
from numpy.linalg import norm

from rlsim.utils import suppress_print


class Environment:
    def __init__(self, 
                 atoms: ase.Atoms | Dict[str, List[List[float]] | Tuple[int, bool]] | os.PathLike | str,
                 calc_params={"platform": "mace", "device": "cuda", "max_iter": 100, "relax_log": "log", "cutoff": 4.0}):
        if isinstance(atoms, ase.Atoms):
            self.atoms = atoms
        elif isinstance(atoms, os.PathLike):
            self.atoms = ase.io.read(atoms, format='vasp')
        elif isinstance(atoms, str):
            self.atoms = ase.io.read(atoms, format='vasp')
        elif isinstance(atoms, Dict):
            self.atoms = ase.Atoms(
                atoms["element"],
                cell=atoms["box"],
                pbc=atoms["pbc"],
                scaled_positions=atoms["pos"])*atoms["supercell"]
        else:
            raise "Atoms should be properly given"
        self.n_atom = len(self.atoms)
        self.pos = self.positions("cartesion").tolist()
        self.output = StringIO()
        self.relax_log = calc_params["relax_log"]
        self.max_iter = calc_params["max_iter"]
        self.cutoff = calc_params["cutoff"]
        self.calc_params = calc_params
        self.atoms.calc = self.get_calculator(**self.calc_params)
        self.device = self.calc_params["device"]

    def get_calculator(self,
                       platform: str = 'mace',
                       device: str = "cuda",
                       potential_id: str = '2018--Choi-W-M-Jo-Y-H-Sohn-S-S-et-al--Co-Ni-Cr-Fe-Mn',
                       **kwargs):

        if platform == 'mace':
            from mace.calculators import mace_mp

            # Suppress print statements in mace_mp function
            with suppress_print():
                calculator = mace_mp(model="medium", dispersion=False, default_dtype="float32", device=device)
                
        elif platform == 'kimpy':
            from ase.calculators.kim.kim import KIM
            calculator = KIM(potential_id);
        
        elif platform =='ase':
            from ase.calculators.eam import EAM
            calculator = EAM(potential=potential_id);
        elif platform == 'matlantis':
            from pfp_api_client.pfp.calculators.ase_calculator import \
                ASECalculator
            from pfp_api_client.pfp.estimator import Estimator
            estimator = Estimator(model_version="v4.0.0");
            calculator = ASECalculator(estimator);
        else:
            raise 'Error: platform should be set as either matlantis or kimpy'
        return calculator

    def set_atoms(self, pos: list, convention='frac', slab=False):    
        # set atomic positions in the configuration by fraction coordinate list pos = [\vec x_1, ..., \vec x_N]
        element = self.atoms.get_atomic_numbers()
        cell = self.atoms.cell
        pbc = self.atoms.pbc
        if slab:
            cell[2,2] *= 1 + 10/norm(cell[2,2]);
        if convention == 'frac':
            self.atoms = ase.Atoms(element, cell=cell, pbc=pbc, scaled_positions=pos);
        else:
            self.atoms = ase.Atoms(element, cell=cell, pbc=pbc, positions=pos);
        self.atoms.calc = self.get_calculator(**self.calc_params)
    
    def remove_atom(self, atom_index):
        element = self.atoms.get_atomic_numbers().tolist()
        cell = self.atoms.cell
        pbc = self.atoms.pbc
        pos = self.atoms.get_scaled_positions().tolist()
        del pos[atom_index]
        del element[atom_index]
        self.atoms = ase.Atoms(element, cell=cell, pbc=pbc, scaled_positions=pos)
        self.atoms.calc = self.get_calculator(**self.calc_params)
        return self.atoms

    def add_atom(self, frac_coords, atomic_number):
        pos = self.atoms.get_scaled_positions().tolist()
        pos.append(frac_coords)
        element = self.atoms.get_atomic_numbers().tolist()
        element.append(atomic_number)
        cell = self.atoms.cell
        pbc = self.atoms.pbc
        self.atoms = ase.Atoms(element,
              cell=cell,
              pbc = pbc,
              scaled_positions=pos)
        self.atoms.calc = self.get_calculator(**self.calc_params)
        return self.atoms

    def positions(self, f='frac'):
        if f[0] == 'f':
            return self.atoms.get_scaled_positions()
        else:
            return self.atoms.get_positions()

    def forces(self):  # compute force on each atom f = [\vec f_1, ..., \vec f_N]. unit: eV/A
        return self.atoms.get_forces()

    def potential(self):   # compute total potential energy. unit: eV
        return self.atoms.get_potential_energy()

    def freq(self, delta=0.05):
        Hlist = np.argwhere(self.atoms.get_atomic_numbers()==1).T[0]
        pos = self.get_positions(f='cartesian')
        f0 = self.forces()
        log_niu_prod = 0
        for i in Hlist:
            Hessian = np.zeros([3,3])
            for j in range(3):
                pos1 = pos.copy()
                pos1[i][j] += delta
                self.atoms.set_positions(pos1);
                Hessian[j] = -(self.forces()-f0)[i]/delta
            prod = np.prod(np.linalg.eigvalsh((Hessian+Hessian.T)/2))
            log_niu_prod += np.log(prod)/2 # + 3*np.log(1.55716*10);
            
        self.atoms.set_positions(pos)
        return log_niu_prod

    def relax(self, accuracy=0.05):
        self.atoms.set_constraint(
            ase.constraints.FixAtoms(mask=[False] * self.n_atom)
        )
        dyn = MDMin(self.atoms, logfile=self.relax_log)
        with redirect_stdout(self.output):
            converge = dyn.run(fmax=accuracy, steps=self.max_iter)
        self.pos = self.atoms.get_positions().tolist()

        return converge

    def step(self, action: list = [], accuracy=0.05):
        self.action = action
        self.pos_last = self.positions(f="cartesion")
        self.initial = self.atoms.copy()

        self.act_atom = self.action[0]
        self.act_displace = np.array(
            [[0, 0, 0]] * self.act_atom
            + [self.action[1:]]
            + [[0, 0, 0]] * (self.n_atom - 1 - self.act_atom)
        )
        self.pos = (self.positions(f="cartesion") + self.act_displace).tolist()
        self.set_atoms(self.pos, convention="cartesion")

        converge = self.relax(accuracy=accuracy)
        fail = int(norm(self.pos_last - self.positions(f="cartesion")) < 0.2) + (
            not converge
        )
        E_next = self.potential()

        return E_next, fail

    def revert(self):
        self.set_atoms(self.pos_last, convention="cartesion")

    def mask(self):
        dist = self.atoms.get_distances(range(self.n_atom), self.act_atom)
        mask = [d > self.cutoff for d in dist]

        return mask

    def normalize_positions(self):
        pos_frac = self.atoms.get_scaled_positions() % 1
        self.atoms.set_scaled_positions(pos_frac)

    def saddle(
        self, moved_atom=-1, accuracy=0.1, n_points=10, r_cut=4
    ):
        self.atoms.set_constraint(
            ase.constraints.FixAtoms(mask=[False] * self.n_atom)
        )
        self.initial.set_constraint(
            ase.constraints.FixAtoms(mask=[False] * self.n_atom)
        )
        images = [self.initial]
        images += [self.initial.copy() for i in range(n_points - 2)]
        images += [self.atoms]

        neb = NEB(images)
        neb.interpolate()

        for image in range(n_points):
            images[image].calc = self.get_calculator(**self.calc_params)
            images[image].set_constraint(ase.constraints.FixAtoms(mask=self.mask()))
        with redirect_stdout(self.output):
            optimizer = MDMin(neb, logfile=self.relax_log)

        res = optimizer.run(fmax=accuracy, steps=self.max_iter)
        res = True
        if res:
            Elist = [image.get_potential_energy() for image in images]
            E = np.max(Elist)

            def log_niu_prod(input_atoms, NN, saddle=False):
                delta = 0.05

                mass = input_atoms.get_masses()[NN]
                mass = np.array([mass[i // 3] for i in range(3 * len(NN))])
                mass_mat = np.sqrt(mass[:, None] * mass[None, :])
                Hessian = np.zeros([len(NN) * 3, len(NN) * 3])
                f0 = input_atoms.get_forces()
                pos_init = input_atoms.get_positions()
                for u in range(len(NN)):
                    for j in range(3):
                        pos1 = pos_init.copy()
                        pos1[NN[u]][j] += delta
                        input_atoms.set_positions(pos1)
                        Hessian[3 * u + j] = (
                            -(input_atoms.get_forces() - f0)[NN].reshape(-1) / delta
                        )

                freq_mat = (Hessian + Hessian.T) / 2 / mass_mat
                if saddle:
                    prod = np.prod(np.linalg.eigvals(freq_mat)[1:])
                else:
                    prod = np.linalg.det(freq_mat)
                output = np.log(np.abs(prod)) / 2

                return output

            max_ind = np.argmax(Elist)
            r_cut = 3
            NN = self.initial.get_distances(
                moved_atom, range(len(self.initial)), mic=True
            )
            NN = np.argwhere(NN < r_cut).T[0]
            self.initial.calc = self.get_calculator(**self.calc_params)
            self.initial.set_constraint(
                ase.constraints.FixAtoms(mask=[False] * self.n_atom)
            )
            images[max_ind].calc = self.get_calculator(**self.calc_params)
            images[max_ind].set_constraint(
                ase.constraints.FixAtoms(mask=[False] * self.n_atom)
            )
            log_niu_min = log_niu_prod(self.initial, NN)
            log_niu_s = log_niu_prod(images[max_ind], NN, saddle=True)

            log_attempt_freq = log_niu_min - log_niu_s + np.log(1.55716 * 10)

        else:
            E = 0
            log_attempt_freq = 0
        self.atoms.set_constraint(
            ase.constraints.FixAtoms(mask=[False] * self.n_atom)
        )

        pos_frac = self.atoms.get_scaled_positions() % 1
        self.atoms.set_scaled_positions(pos_frac)

        return E, log_attempt_freq, int(not res)