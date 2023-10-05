import os, datetime, itertools
from typing import List
import pandas as pd
from tqdm import tqdm

import torch
from torch_geometric.data import InMemoryDataset, Data


class ECLDataset(InMemoryDataset):
    """Create graph dataset from parquet files"""

    def __init__(
        self,
        root,
        raw_filename,
        processed_filename,
        n_photons=1,
        n_events=1000,
        features=[
            "rec_energy",
            "rec_time",
            "psd_norm",
            "chi2_norm",
            "fit_type",
            "theta_norm",
            "phi_sin_norm",
            "phi_cos_norm",
            "local_theta_norm",
            "local_phi_norm",
            "mass_norm",
            "rec_energy_norm",
            "center_id",
        ],
        transform=None,
        pre_transform=None,
    ):
        """
        Args:
            root (str): Root directory where the dataset should be saved.
            raw_filename (str): Name of the data file.
            processed_filename (str): Name of the processed file.
            n_photons (int): Number of photons in the event (1 or 2).
            n_events (int): Number of events to be processed.
            features (list): List of features to be used in the graph.
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
            pre_transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before being saved to
                disk.
        """
        self.gammas = n_photons
        self.n_events = n_events
        self.features = features

        self._raw_file_names = raw_filename
        self._processed_file_names = processed_filename
        self._root = root
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return self._root

    @property
    def raw_paths(self) -> List[str]:
        return [
            os.path.join(self.raw_dir, raw_filename)
            for raw_filename in self.raw_file_names
        ]

    @property
    def raw_file_names(self):
        return [self._raw_file_names]

    @property
    def processed_dir(self) -> str:
        return os.path.join(self._root, "processed")

    @property
    def processed_file_names(self):
        return [self._processed_file_names]

    @property
    def processed_paths(self) -> List[str]:
        return [
            os.path.join(self.processed_dir, processed_filename)
            for processed_filename in self.processed_file_names
        ]

    def process(self):
        total_events = 0
        data_list = []
        for raw_path in self.raw_paths:
            events = pd.read_parquet(raw_path)

            for uni_event_id, event in events.groupby("uni_event_id"):
                # convert to torch
                if "phi_norm" in self.features:
                    event["phi_norm"] = event["phi"] / 3.1415
                # add local maximum as input feature if it is part of the features
                if "center_id" in self.features:
                    event["center_id"] = (
                        event.index.get_level_values("cell_id") == event["center_id"]
                    ).astype(float)
                if "lm0" in self.features:
                    event["lm0"] = (
                        event.index.get_level_values("cell_id") == event["lm0_id"]
                    ).astype(float)
                if "lm1" in self.features:
                    event["lm1"] = (
                        event.index.get_level_values("cell_id") == event["lm1_id"]
                    ).astype(float)

                node_feats = torch.tensor(event[self.features].values).float()
                if self.gammas == 1:
                    grnd_truth = torch.tensor(
                        event[["c0_frac", "bkg_frac"]].values
                    ).float()
                elif self.gammas >= 2:
                    grnd_truth = torch.tensor(
                        event[["c0_frac", "c1_frac", "bkg_frac"]].values
                    ).float()

                graph = Data(x=node_feats, y=grnd_truth)

                # add unique event id for identification
                graph.uni_event_id = uni_event_id

                data_list.append(graph)

                total_events += 1

                if total_events == self.n_events:
                    break

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
