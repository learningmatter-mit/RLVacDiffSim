import argparse
import json
import os

import numpy as np
import toml
from rgnn.common.registry import registry

from rlsim.drl.simulator import RLSimulator
from rlsim.environment import Environment
from rlsim.utils.logger import setup_logger


def deploy_RL(settings, atoms_traj=None):
    with open(settings, "r") as f:
        config = toml.load(f)
        task = config["task"]
        logger_config = config["logger"]
        deploy_config = config["deploy"]
        model_config = config["model"]

    if task not in os.listdir():
        os.makedirs(task, exist_ok=True)
    log_filename = f"{task}/{logger_config['filename']}.log"
    logger = setup_logger(logger_config["name"], log_filename)

    model = registry.get_model_class(model_config["@name"]).load(f"{model_config['model_path']}")

    calc_params = deploy_config.pop("calc_info")
    calc_params.update({"relax_log": f"{task}/{calc_params['relax_log']}"})
    n_episodes = deploy_config.pop("n_episodes")
    horizon = deploy_config.pop("horizon")
    simulation_mode = deploy_config.pop("mode")
    simulation_params = deploy_config.pop("simulation_params")
    if atoms_traj is not None:
        pool = atoms_traj
    else:
        poscar_dir = deploy_config.pop("poscar_dir")
        n_poscars = deploy_config.pop("n_poscars")
        pool = [f"{poscar_dir}/POSCAR_" + str(i) for i in range(0, n_poscars)]

    if simulation_mode == "lss":
        El = []
        output_file = str(task) + "/converge.json"
    elif simulation_mode == "tks":
        Tl = []
        Cl = []
        output_file = str(task) + "/diffuse.json"
    for u in range(n_episodes):
        logger.info(f"Episode: {u}")
        file = pool[np.random.randint(len(pool))]
        env = Environment(file, calc_params=calc_params)
        env.relax()
        simulator = RLSimulator(environment=env,
                                model=model,
                                model_params=model_config["params"],
                                params=deploy_config)
        atoms_traj = str(task) + "/XDATCAR" + str(u)
        outputs = simulator.run(horizon=horizon,
                                logger=logger,
                                atoms_traj=atoms_traj,
                                mode=simulation_mode,
                                **simulation_params)
        if simulation_mode == "lss":
            El.append(outputs[0])
        elif simulation_mode == "tks":
            Tl.append(outputs[0])
            Cl.append(outputs[1])
    if simulation_mode == "lss":    
        with open(output_file, "w") as file:
            json.dump(El, file)
    elif simulation_mode == "tks":
        with open(output_file, "w") as file:
            json.dump([Tl, Cl], file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DRL model')
    parser.add_argument('--config', required=True, help='config file path')

    args = parser.parse_args()
    deploy_RL(args.config)
