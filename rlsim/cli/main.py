import click

from rlsim.drl.deploy import deploy_RL
from rlsim.drl.train import train_DQN
from rlsim.time.train import train_Time


@click.command()
@click.argument("simulation")
@click.option("-c", "--config_name", help="Name config file in toml", required=True)
def main(simulation, config_name):
    """
    Main command-line interface for RL simulation and training.

    This function serves as the entry point for running various reinforcement learning (RL) 
    simulations and training tasks via the command line. The specific task to be run is 
    determined by the `simulation` argument.

    Parameters:
    simulation (str): The type of operation to perform. Should be one of the following:
        - "rl-train": To train a Deep Q-Network (DQN) using the specified configuration file.
        - "rl-deploy": To deploy and run an RL simulation based on the specified configuration file.

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
    """
    if simulation == "rl-train":
        train_DQN(config_name)
    elif simulation == "rl-deploy":
        deploy_RL(config_name)
    elif simulation == "time-train":


    else:
        raise click.UsageError(f"Unsupported simulation type: {simulation}. Please use 'rl-train' or 'rl-deploy'.")


