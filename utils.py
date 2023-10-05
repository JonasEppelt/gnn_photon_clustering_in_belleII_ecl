import numpy as np
import torch
import os

import matplotlib.pyplot as plt


def resolution_plot(
    true,
    pred,
    label,
    xlabel=None,
    ax=None,
    color="#8eba42",
    hatch=r"\\\ ",
    fontsize=13,
    plot_range=[-0.5, 0.5],
    bins=100,
):
    """Plot the resolution of a prediction compared to the true value.

    Args:
        true (np.array): True values.
        pred (np.array): Predicted values.
        label (str): Label for the plot.
        color (str): Color for the plot.
        hatch (str): Hatch pattern for the hatched area.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    # calculate resolution (= relative error) and clipping to plot_range to have overflow bins
    rec_errs = (pred - true) / true
    rec_errs = np.clip(rec_errs, *plot_range)

    # PLOT HISTOGRAM
    values, bin_edges, _ = ax.hist(
        rec_errs,
        bins=bins,
        color=color,
        histtype="step",
        hatch=hatch,
        range=plot_range,
        label=str(label),
        alpha=0.6,
    )

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel("Number of events", fontsize=fontsize)
    ax.legend(loc="center right", fontsize=fontsize)

    return ax


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0.001,
        path="./",
        filename="checkpoint.pt",
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0.001
            path (str): Path for the checkpoint to be saved to.
                            Default: './'
            filename (str): Filename for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = os.path.join(path, filename)

    def __call__(self, val_loss, model):
        # compare the validation loss to the previous one, save model if validation loss is better than previous
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        # if the validation loss is not better than the previous one, increase the counter
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            # if the counter reaches the patience, stop training
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
