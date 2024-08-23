import os
import warnings

import numpy as np
from ase import io
from rgnn.models.dqn import ReactionDQN
from rgnn.models.reaction_models import PaiNN

from rlsim.actions.action import get_action_space
from rlsim.configuration import configuration
from rlsim.memory import Memory
from rlsim.step import environment
from rlsim.train_graph import ContextBandit
from rlsim.utils.logger import setup_logger

warnings.filterwarnings("ignore", category=UserWarning)

task = "data/Vrandom_mace_256/traj0"
horizon = 20
n_traj = 100

species = ["Cr", "Co", "Ni"]

# gcnn = PaiNN(species=species)
# reaction_model = ReactionDQN(gcnn)
# trainer = ContextBandit(reaction_model, temperature=1000, lr=5e-5)

pool = ["data/POSCARs_256/POSCAR_" + str(i) for i in range(0, 100)]

traj_list = []
if task not in os.listdir():
    os.makedirs(task, exist_ok=True)
# Configure logging
log_filename = f"{task}/logger.log"  # Define your log filename
opt_log_filenmae = f"{task}" + "/log"
if os.path.exists(opt_log_filenmae):
    os.remove(opt_log_filenmae)

logger = setup_logger("RL", log_filename)
# new_pool = []
# for filename in pool:
#     atoms = io.read(filename)
#     if len(atoms) < 500:
#         new_pool.append(filename)
# logger.info(f"Original pool num: {len(pool)}, Filtered pool num: {len(new_pool)}")


if "traj" not in os.listdir(task):
    os.mkdir(task + "/traj")
if "model" not in os.listdir(task):
    os.mkdir(task + "/model")

for epoch in range(n_traj):
    conf = configuration()
    file = pool[np.random.randint(len(pool))]
    conf.load(file)
    logger.info("epoch = " + str(epoch) + ":  " + file)
    conf.set_potential(platform="mace")
    env = environment(conf, logfile=task + "/log", max_iter=100)
    env.relax(accuracy=0.1)
    traj_list.append(Memory(1, 0))
    for tstep in range(horizon):
        action_space = get_action_space(conf)
        act_id = np.random.choice(len(action_space))
        action = action_space[act_id]
        info = {
            "act": act_id,
            "act_probs": [],
            "act_space": action_space,
            "state": conf.atoms.copy(),
            "E_min": conf.potential(),
        }

        E_next, fail = env.step(action, accuracy=0.05)
        if not fail:
            E_s, freq, fail = env.saddle(
                action[0], n_points=8, accuracy=0.07, platform="mace"
            )
            info["E_s"], info["log_freq"] = E_s, freq
        else:
            info["E_s"] = 0
            info["log_freq"] = 0
            logger.info("fail step 1")

        info["next"], info["fail"], info["E_next"] = conf.atoms.copy(), fail, E_next
        traj_list[-1].add(info)
        if fail:
            logger.info("fail")
        if tstep % 10 == 0 and tstep > 0:
            logger.info("    t = " + str(tstep))

    try:
        traj_list[epoch].save(task + "/traj/traj" + str(epoch))
    except:
        logger.info("saving failure")
