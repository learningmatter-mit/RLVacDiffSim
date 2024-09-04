# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 10:10:21 2022

@author: 17000
"""

import json

import ase
import numpy as np
from ase import io


class Memory:
    # This class records the trajectory and critical coefficients along the trajectory
    # Alternatively apply add_minimum and add_saddle, passing 'configuration' objects to the methods
    # Trajectories can be output as lammps output files, which can be visualized by OVITO or AtomEye

    def __init__(self, k_s, k_min, T=0.0):
        self.k_s = k_s
        self.k_min = k_min
        self.states = []
        self.next_states = []
        self.actions = []
        self.act_space = []
        self.action_probs = []
        self.rewards = []
        self.freq = []

        self.E_min = []
        self.barrier = []
        self.E_s = []
        self.E_next = []
        self.fail = []
        self.R = []
        self.S = []
        self.T = T
        self.fail_panelty = -0.5
        self.kb = 1.380649 / 1.602 * 10**-4
        self.meV_to_Hz = 1.602 / 6.626 * 10**12

    def add(self, info):
        self.states.append(info["state"])
        self.actions.append(info["act"])
        self.act_space.append(info["act_space"])
        self.action_probs.append(info["act_probs"])
        self.fail.append(info["fail"])
        self.next_states.append(info["next"])
        self.freq.append(info["log_freq"])
        self.E_s.append(info["E_s"])
        self.E_min.append(info["E_min"])
        self.E_next.append(info["E_next"])

        if not info["fail"]:
            self.barrier.append(info["E_s"]-info["E_min"])
            self.rewards.append(
                self.k_s
                * (-self.barrier[-1] + self.kb * self.T * self.freq[-1])
                + self.k_min * (self.E_min[-1] - self.E_next[-1])
            )
        else:
            self.barrier.append(0.0)
            self.rewards.append(self.fail_panelty)

    def HTST(self, T):
        self.t_list = []
        for i in range(len(self.E_s)):
            if type(self.freq_s[i]) == type(None) or type(self.freq_min[i]) == type(
                None
            ):
                raise (
                    "Error: frequency has not been calculated, so time cannot be evaluated."
                )
            f_s = np.log(np.sqrt(np.sort(self.freq_s[i][1:])))
            f_m = np.log(self.freq_min[i])
            exp_term = np.exp(-(self.E_s[i] - self.E_min[i]) / self.kb / T)
            freq_term = np.exp(np.sum(f_m) - np.sum(f_s)) / 2 * np.pi * self.meV_to_Hz
            self.t_list.append(1 / (exp_term * freq_term))
        return self.t_list

    def to_file(self, filename, animation=False):
        io.write(filename, self.states, format="vasp-xdatcar")
        if animation:
            io.write(filename, self.states, format="mp4")

    def save(self, filename):
        keys = ["numbers", "positions", "cell", "pbc"]
        to_list = [
            self.k_s,
            self.k_min,
            [{key: u.todict()[key].tolist() for key in keys} for u in self.states],
            self.E_min,
            [{key: u.todict()[key].tolist() for key in keys} for u in self.next_states],
            self.actions,
            [[[int(v[0])] + v[1:] for v in u] for u in self.act_space],
            self.action_probs,
            self.rewards,
            self.E_min,
            self.barrier,
            self.E_s,
            self.E_next,
            self.fail,
            self.freq,
        ]

        with open(filename + ".json", "w") as file:
            json.dump(to_list, file)

    def load(self, filename):
        with open(filename + ".json", "r") as file:
            data = json.load(file)
        [
            self.k_s,
            self.k_min,
            states,
            self.E_min,
            next_states,
            self.actions,
            self.act_space,
            self.action_probs,
            self.rewards,
            self.E_min,
            self.barrier,
            self.E_s,
            self.E_next,
            self.fail,
            self.freq,
        ] = data
        # self.trajectory = []
        for i in range(len(self.actions)):
            self.states.append(ase.Atoms.fromdict(states[i]))
            self.next_states.append(ase.Atoms.fromdict(next_states[i]))
