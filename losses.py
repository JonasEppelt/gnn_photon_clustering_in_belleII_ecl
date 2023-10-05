import torch
from torch_geometric.nn import global_add_pool


def l2_loss(pred, batch):
    """Calculate the l2 loss.

    Args:
        pred (torch.tensor): Predicted values.
        batch (torch_geometric.data.Batch): Batch of graphs."""
    loss_func = (batch.y - pred) ** 2

    loss = torch.sum(
        global_add_pool(loss_func, batch.batch), dim=1
    )  # sum over all nodes and targets
    loss = torch.mean(loss, dim=0)  # average over all graphs in batch

    return loss
