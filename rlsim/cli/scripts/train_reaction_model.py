import os
import click
import torch
import toml
from rgnn.train.trainer import Trainer, test_model
from torch_geometric.loader import DataLoader
from rgnn.models.reaction_models.utils import get_scaler
from rgnn.common.registry import registry
from sklearn.model_selection import train_test_split
from rlsim.utils.logger import setup_logger


def train_reaction_model(settings):
    with open(settings, "r") as f:
        config = toml.load(f)
        task = config["task"]
        logger_config = config["logger"]
        train_config = config["train"]
        model_config = config["model"]

    if task not in os.listdir():
        os.makedirs(task, exist_ok=True)
    if "model" not in os.listdir(task):
        os.mkdir(task + "/model")
    batch_size = train_config.get("batch_size", 8)
    device = train_config.get("device", "cuda")
    num_epoch = train_config["num_epoch"]
    start_epoch = train_config.get("start_epoch", 0)
    save_model_name = train_config.get("save_model_name", "reaction_model.pth.tar")
    save_model_path = f"{task}/model"

    # Setup logger
    log_filename = f"{task}/{logger_config['filename']}.log"
    logger = setup_logger(logger_config["name"], log_filename)
    logger.info(f"Training reaction model in: {os.path.realpath(task)}")
    # Load dataset
    total_dataset = torch.load(train_config["trainset_path"])
    val_size = train_config.get("val_size", 0.2)
    trainset, valset = train_test_split(total_dataset, test_size=val_size, random_state=42)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size)
    if train_config.get("testset_path", None) is not None:
        testset = torch.load(train_config["testset_path"])
    else:
        testset = None

    # Setup model
    train_labels = train_config["train_labels"]
    means, stddevs = get_scaler(train_labels, trainset)
    means_str = "\t".join([f"{key}: {value.item():.4f}" for key, value in means.items()])
    stddevs_str = "\t".join([f"{key}: {value.item():.4f}" for key, value in stddevs.items()])
    logger.info(f"Means: {means_str}")
    logger.info(f"Stddevs: {stddevs_str}")
    model_name = model_config.pop("@name")
    model_config.update({"means": means, "stddevs": stddevs})
    reaction_model = registry.get_reaction_model_class(model_name)(**model_config)
    trainable_params = filter(lambda p: p.requires_grad, reaction_model.parameters())
    
    # Setup loss function, optimizer and scheduler
    loss_params = train_config["loss_params"]
    loss_fn = registry.get_loss_class(train_config["loss_fn"])(**loss_params)
    optimizer_params = train_config.get("optimizer_params", {"lr": 3e-4, "weight_decay": 0.0})
    optimizer = registry.get_optimizer_class(train_config.get("optimizer", "adam"))(trainable_params, **optimizer_params)
    scheduler_params = train_config.get("scheduler_params", {"patience": 10, "factor": 0.1})
    scheduler = registry.get_scheduler_class(train_config.get("scheduler", "reduce_lr_on_plateau"))(optimizer=optimizer, **scheduler_params)

    trainer = Trainer(
        save_dir=save_model_path,
        model=reaction_model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        validation_loader=val_loader,
    )
    trainer.train(
        logger=logger,
        device=device,
        n_epochs=num_epoch,
        start_epoch=start_epoch,
        best_model_name=save_model_name,
    )

    if testset is not None:
        test_results = test_model(
            dataset=testset,
            model=reaction_model,
            keys=train_labels,
            device=device,
            batch_size=batch_size,
        )

        with open(f"{task}/test_results.txt", "a") as f:
            f.write("Test Results:\n")
            f.write("--------------------------------\n")
        logger.info("Test Results--------------------")
        for key, value in test_results.items():
            pred = value["pred"]
            label = value["target"]
            mae = torch.abs(pred - label).mean()
            mse = torch.square(pred - label).mean()
            rmse = torch.sqrt(mse)
            r2 = 1 - torch.square(pred - label).mean() / torch.square(label - label.mean()).mean()
            logger.info(f"{key}: MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
            with open(f"{task}/test_results.txt", "a") as f:
                f.write(f"{key}: MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}\n")
        with open(f"{task}/test_results.txt", "a") as f:
            f.write("--------------------------------\n")
    
@click.command()
@click.option("-c", "--config", required=True, help="config file path")
def main(config):
    """
    Train reaction model
    """
    train_reaction_model(config)
