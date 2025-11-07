import json
import os

import numpy as np
import toml
from rgnn.common.registry import registry

from rlsim.drl.simulator import RLSimulator
from rlsim.environment import Environment


def deploy_RL(task, logger, config, atoms_traj=None):
    logger.info(f"Deploy DRL in: {os.path.realpath(task)}")
    toml.dump(config, open(f"{task}/config_copied.toml", "w"))
    deploy_config = config["deploy"]
    model_config = config.get("model", None)
    if model_config is not None:
        model = registry.get_model_class(model_config["@name"]).load(f"{model_config['model_path']}")
        model_params = model_config["params"]
    else:
        model = None
        model_params = None

    calc_params = deploy_config.pop("calc_info")
    calc_params.update({"relax_log": f"{task}/{calc_params['relax_log']}"})
    horizon = deploy_config.pop("horizon")
    simulation_mode = deploy_config.pop("mode")
    simulation_params = deploy_config.pop("simulation_params")

    sro_pixel0 = deploy_config.pop("sro_pixel0", None)
    sro_pixel1 = deploy_config.pop("sro_pixel1", None)
    sro_pixel2 = deploy_config.pop("sro_pixel2", None)
    sro_pixel_length = deploy_config.pop("sro_pixel_length", None)
    if sro_pixel0 is not None and sro_pixel1 is not None and sro_pixel2 is not None and sro_pixel_length is not None:
        sro_pixel = (sro_pixel0, sro_pixel1, sro_pixel2, sro_pixel_length)
    else:
        sro_pixel = None

    if atoms_traj is not None:
        pool = atoms_traj
    else:
        poscar_dir = deploy_config.pop("poscar_dir")
        n_poscars = deploy_config.pop("n_poscars")
        pool = [f"{poscar_dir}/POSCAR_" + str(i) for i in range(0, n_poscars)]
    if deploy_config.get("all_episodes", False):
        n_episodes = len(pool)
        logger.info(f"Running {n_episodes} (Serial) episodes in {simulation_mode} mode")
    else:
        n_episodes = deploy_config.pop("n_episodes")
        logger.info(f"Running {n_episodes} (Random) episodes in {simulation_mode} mode")

    # if simulation_mode == "lss" or simulation_mode == "mcmc":
    El = []
    output_file = str(task) + "/converge.json"
    if simulation_mode != "mcmc":
        Ql = []
        output_file_q = str(task) + "/q_values.json"
    if simulation_mode == "tks":
        Tl = []
        Cl = []
        output_file_tks = str(task) + "/diffuse.json"
    for u in range(n_episodes):
        if deploy_config.get("all_episodes", False):
            logger.info(f"Episode: {u} (Serial)")
            file = pool[u]
        else:
            logger.info(f"Episode: {u} (Random)")
            file = pool[np.random.randint(len(pool))]
        env = Environment(file, calc_params=calc_params)
        env.relax()
        simulator = RLSimulator(environment=env,
                                model=model,
                                q_params=model_params,
                                sro_pixel=sro_pixel)
        atoms_traj = str(task) + "/XDATCAR" + str(u)
        outputs = simulator.run(horizon=horizon,
                                logger=logger,
                                atoms_traj=atoms_traj,
                                mode=simulation_mode,
                                **simulation_params)
        El.append(outputs[0])
        with open(output_file, "w") as file:
            json.dump(El, file)
        if simulation_mode != "mcmc":
            Ql.append(outputs[1])
            with open(output_file_q, "w") as file:
                json.dump(Ql, file)
        if simulation_mode == "tks":
            Tl.append(outputs[2])
            Cl.append(outputs[3])
            with open(output_file_tks, "w") as file:
                json.dump([Tl, Cl], file)
