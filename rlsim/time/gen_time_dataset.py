import json
import os
import sys
from itertools import product

import numpy as np
import torch
from ase import io
from rgnn.common.registry import registry

sys.path.append("/home2/hojechun/github_repo/RL_H_diffusion")

from rlsim.action_space_v3 import actions as actions_v3
from rlsim.configuration import configuration
from rlsim.step import environment
from rlsim.train_graph_v2 import select_action
from rlsim.utils.logger import setup_logger

task = "/home2/hojechun/00-research/14-time_estimation/900_4"
if task not in os.listdir():
    os.makedirs(task, exist_ok=True)

log_filename = f"{task}/logger.log"  # Define your log filename
logger = setup_logger("Deploy", log_filename)
# sro = "0.8"
n_episodes = 100
horizon = 30
original_n_atoms = 108
T = 900
kT = T * 8.617 * 10**-5
vacancy = 2

model_path = "/home2/hojechun/github_repo/RL_H_diffusion/dev/Vrandom_DQN_new_sum"
model = registry.get_model_class("dqn_v2").load(f"{model_path}/model/model_trained")

q_params = {
    "temperature": T,
    "alpha": 1.0,
    "beta": 0.0,
    "dqn": False,
}
Tl = []
Cl = []

pool = [(j,k) for j,k in product(range(10), range(1200))]
def get_atoms(pool_ind):
    # T = pool_ind[0];
    traj = pool_ind[0];
    frame = pool_ind[1];
    atoms = io.read("paper/DQN_"+str(original_n_atoms-vacancy)+"_atoms_" + str(T) + "K/XDATCAR" + str(traj), index = frame);
    return atoms;

#new_pool = []
#for filename in pool:
#    atoms = io.read(filename)
#    if len(atoms) < original_n_atoms:
#        new_pool.append(filename)
#logger.info(f"Original pool num: {len(pool)}, Filtered pool num: {len(new_pool)}")

for u in range(n_episodes):
    conf = configuration()
    file = pool[np.random.randint(len(pool))]
    conf.atoms = get_atoms(file);
    conf.set_potential(platform="mace")
    env = environment(conf, max_iter=100, logfile=task + "/log")
    env.relax(accuracy=0.1)

    filename = str(task) + "/XDATCAR" + str(u)
    io.write(filename, conf.atoms, format="vasp-xdatcar")
    tlist = [0]
    clist = [conf.atoms.get_positions()[-1].tolist()]
    logger.info(f"Episode: {u}")
    for tstep in range(horizon):
        action_space = actions_v3(conf)
        act_id, act_probs, Q = select_action(model, conf.atoms, action_space, q_params)
        Gamma = float(torch.sum(torch.exp(Q/kT)))
        dt = 1 / Gamma * 10**-6
        tlist.append(tlist[-1] + dt)
        action = action_space[act_id]
        E_next, fail = env.step(action, accuracy=0.1)
        io.write(filename, conf.atoms, format="vasp-xdatcar", append=True)
        clist.append(conf.atoms.get_positions()[-1].tolist())
        if tstep % 10 == 0:
            logger.info(
                f"Step: {tstep}, T: {q_params['temperature']:.2f}, E: {conf.potential():.3f}"
            )
    Tl.append(tlist)
    Cl.append(clist)
    with open(str(task) + "/diffuse.json", "w") as file:
        json.dump([Tl, Cl], file)
