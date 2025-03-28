import os

import click
import toml
from ase import io

from rlsim.drl.deploy import deploy_RL
from rlsim.drl.train import train_DQN
from rlsim.time.train import TimeTrainer
from rlsim.utils.logger import setup_logger


@click.command()
@click.argument("simulation")
@click.option("-c", "--config_name", help="Name config file in toml", required=True)
def main(simulation, config_name):
    """
    This function serves as the entry point for running various reinforcement learning (RL) 
    simulations and training tasks via the command line. The specific task to be run is 
    determined by the `simulation` argument.

    Parameters:
    simulation (str): The type of operation to perform. Should be one of the following:
        - "rl-train": To train a Deep Q-Network (DQN) using the specified configuration file.
        - "rl-deploy": To deploy and run an RL simulation based on the specified configuration file.
        - "time-train": To train a time estimator using the specified configuration file.

    config_name (str): The name of the configuration file (in TOML format) to be used for 
    the selected operation. This file should contain the necessary parameters for training 
    or deployment.

    Raises:
    NotImplementedError: If an unsupported `simulation` argument is provided.

    Examples:
    Run a training session using the configuration in 'config.toml':
        $ rlsim rl-train -c config.toml

    Deploy and run an RL simulation using the configuration in 'config.toml':
        $ rlsim rl-deploy -c config.toml
    Train a time estimator using the configuration in 'config.toml':
        $ rlsim time-train -c config.toml
    """
    with open(config_name, "r") as f:
        config = toml.load(f)
        task = config.pop("task")
        logger_config = config.pop("logger")
    if task not in os.listdir():
        os.makedirs(task, exist_ok=True)
    log_filename = f"{task}/{logger_config['filename']}.log"
    logger = setup_logger(logger_config["name"], log_filename)
    if simulation == "rl-train":
        train_DQN(task, logger, config)
    elif simulation == "rl-deploy":
        if "atoms_list" in config["deploy"].keys():
            atoms_list = config["deploy"].pop("atoms_list")
            atoms_traj = []
            for atoms_file in atoms_list:
                atoms = io.read(atoms_file)
                atoms_traj.append(atoms)
            deploy_RL(task, logger, config, atoms_traj=atoms_traj)
        else:
            deploy_RL(task, logger, config)
    elif simulation == "time-train":
        t_trainer = TimeTrainer(task, logger, config)
        t_trainer.train()
    else:
        raise click.UsageError(f"Unsupported simulation type: {simulation}. Please use 'rl-train', 'rl-deploy' or 'time-train'.")


