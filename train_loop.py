import yaml
import argparse
import torch

from torch_geometric.loader import DataLoader

import numpy as np

from model import GNNmodel

from datasets import ECLDataset

from losses import l2_loss

from utils import EarlyStopping


def get_datasets(ratio, full_dataset):
    """split the dataset into train and validation set
    Args:
        ratio (float): ratio of validation set to full dataset
        full_dataset (torch_geometric.data.Dataset): full dataset
    """
    n_val = int(np.floor(ratio * len(full_dataset)))
    n_train = int(len(full_dataset) - n_val)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val]
    )

    return train_dataset, val_dataset


def validate(config, model, valloader):
    # implement a validation loop that sets the model to eval and calculates the validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in valloader:
            batch = batch.to(config["device"])
            pred = model(batch)
            loss = l2_loss(pred, batch)
            val_loss += loss.item()
        # calculate average loss over all batches
        val_loss = val_loss / len(valloader)

    return val_loss


# A training loop that runs over epochs and batches and evaluates the loss function
def train(config, model, trainloader, valloader, optimizer, lr_scheduler):
    """Train the model.
    Args:
        config (dict): Dictionary containing the configuration.
        model (torch.nn.Module): Model to be trained.
        trainloader (torch_geometric.loader.DataLoader): DataLoader for the training set.
        valloader (torch_geometric.loader.DataLoader): DataLoader for the validation set.
        optimizer (torch.optim): Optimizer for the training.
        lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler."""

    if config["early_stopping"]:
        early_stopping = EarlyStopping(
            patience=config["patience"],
            verbose=True,
            delta=config["delta"],
            path=config["checkpoint_path"],
            filename=config["checkpoint_filename"],
        )

    early_stopping_flag = False
    last_epoch = config["epochs"]
    for epoch in range(config["epochs"]):
        full_loss = 0.0

        model.train()
        for idx, batch in enumerate(trainloader):
            batch = batch.to(config["device"])
            pred = model(batch)
            loss = l2_loss(pred, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            full_loss += loss.item()

            if idx % config["val_step"] == 0:
                val_loss = validate(config, model, valloader)

                lr_scheduler.step(val_loss)

                full_loss = full_loss / config["val_step"]
                print(
                    f"Epoch: {epoch}, Batch: {idx}, Train Loss: {full_loss:.5f}, Val Loss: {val_loss:.5f}"
                )

                full_loss = 0.0

                early_stopping(val_loss, model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    early_stopping_flag = True
                    break

        if early_stopping_flag:
            last_epoch = epoch
            break

    return model, last_epoch
