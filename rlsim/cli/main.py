import click

from rlsim.drl.deploy import deploy_RL
from rlsim.drl.train import train_DQN


@click.command()
@click.argument("mode")
@click.argument("simulation")
@click.option("-c", "--config_name", help="Name config file in toml")
def main(mode, simulation, config_name):
    if mode == "rl":
        if simulation == "train":
            train_DQN(config_name)
        elif simulation == "deploy":
            deploy_RL(config_name)
        else:
            NotImplementedError
    else:
        NotImplementedError

