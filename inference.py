import argparse
import yaml
import pandas as pd
import numpy as np

from model import GNNmodel

from datasets import ECLDataset

import torch
from torch_geometric.loader import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default=None)
    parser.add_argument("-c", "--config", default="configs/one_photon_train_config.yml")
    parser.add_argument(
        "-m", "--model", default="trained_model.pt"
    )  # add path of trained model to config
    args = parser.parse_args()

    with open(args.config) as cfg_path:
        config = yaml.safe_load(cfg_path)

    config["device"] = args.device
    config["model_path"] = args.model

    model = GNNmodel(
        features=config["features"],
        n_photons=config["n_photons"],
        dense_layer_dim=config["dense_layer_dim"],
        feature_space_dim=config["feature_space_dim"],
        spatial_information_dim=config["spatial_information_dim"],
        k=config["k"],
        n_gravblocks=config["n_gravblocks"],
        batch_norm_momentum=config["batch_norm_momentum"],
    ).to(config["device"])

    model.load_state_dict(
        torch.load(config["model_path"]), map_location=config["device"]
    )

    inference_dataset = ECLDataset(
        raw_dir=config["raw_dir"],
        processed_dir=config["processed_dir"],
        raw_filename=config["raw_filename"],
        processed_filename=config["processed_filename"],
        gammas=config["gammas"],
        n_events=config["n_events"],
        features=config["features"],
    )

    inference_loader = DataLoader(
        inference_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    inf_data = np.empty((0, config["gammas"] + 1))

    with torch.no_grad():
        for sample in inference_loader:
            sample = sample.to(config["device"])
            pred = model(sample).detach().cpu().numpy()
            inf_data = np.concatenate((inf_data, pred))

        if config["gammas"] == 1:
            inf_df = pd.DataFrame(inf_data, columns=["c0_frac_pred", "bkg_frac_pred"])
        else:
            inf_df = pd.DataFrame(
                inf_data,
                columns=["c0_frac_pred", "c1_frac_pred", "bkg_frac_pred"],
            )

            # save inference data
        inf_df.to_parquet(config["inference_path"], compression="snappy")
