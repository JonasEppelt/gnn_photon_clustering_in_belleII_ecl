import torch
import torch.nn as nn
import torch.nn.functional as F


from torch_geometric.nn import GravNetConv, BatchNorm, global_mean_pool


class GNNmodel(nn.Module):
    def __init__(
        self,
        n_photons,
        features,
        dense_layer_dim=22,
        feature_space_dim=16,
        spatial_information_dim=6,
        k=14,
        n_gravblocks=4,
        batch_norm_momentum=0.01,
    ):
        """
        Args:
            n_photons (int): Number of photons in the event (1 or 2).
            features (list): List of features per node.
            dense_layer_dim (int): Number of neurons in the dense layers.
            feature_space_dim (int): Number of dimensions for the feature space.
            spatial_information_dim (int): Number of dimensions for the spatial information.
            k (int): Number of nearest neighbors in the gravnet layer.
            n_gravblocks (int): Number of gravnet blocks.
            batch_norm_momentum (float): Momentum for the batch normalization layers.
        """
        super().__init__()
        input_dim = len(features)  # input length can be inferred from features

        # first block to start with input dim
        self.blocks = nn.ModuleList(
            [
                # start with the first block according to input dimension
                nn.ModuleList(
                    [
                        nn.Linear(2 * input_dim, dense_layer_dim),
                        nn.Linear(dense_layer_dim, dense_layer_dim),
                        nn.Linear(dense_layer_dim, dense_layer_dim),
                        GravNetConv(
                            in_channels=dense_layer_dim,
                            out_channels=dense_layer_dim,
                            space_dimensions=spatial_information_dim,
                            k=k,
                            propagate_dimensions=feature_space_dim,
                        ),
                        BatchNorm(dense_layer_dim, momentum=batch_norm_momentum),
                    ]
                )
            ]
        )

        # loop over remaining blocks as they are currently built the same
        self.blocks.extend(
            nn.ModuleList(
                [
                    # add according to number of blocks
                    nn.ModuleList(
                        [
                            nn.Linear(dense_layer_dim, dense_layer_dim),
                            nn.Linear(dense_layer_dim, dense_layer_dim),
                            nn.Linear(dense_layer_dim, dense_layer_dim),
                            GravNetConv(
                                in_channels=dense_layer_dim,
                                out_channels=dense_layer_dim,
                                space_dimensions=spatial_information_dim,
                                k=k,
                                propagate_dimensions=feature_space_dim,
                            ),
                            BatchNorm(dense_layer_dim, momentum=batch_norm_momentum),
                        ]
                    )
                    for _ in range(n_gravblocks - 1)
                ]
            )
        )

        # final layers
        self.final1 = torch.nn.Linear(
            in_features=n_gravblocks * dense_layer_dim, out_features=64
        )
        self.final2 = torch.nn.Linear(in_features=64, out_features=n_photons + 1)
        self.final3 = torch.nn.Linear(
            in_features=n_photons + 1, out_features=n_photons + 1
        )

    def forward(self, data):
        x, edge_index, batch, num_graphs = (
            data.x,
            data.edge_index,
            data.batch,
            data.num_graphs,
        )

        feat = []

        # global exchange to append the average of each feature in each event to every node
        out = global_mean_pool(x, batch)
        x = torch.cat([x, out[batch]], dim=-1)
        for i, block in enumerate(self.blocks):
            x = F.elu(block[0](x))  # linear
            x = F.elu(block[1](x))  # linear
            x = torch.tanh(block[2](x))  # linear
            x = block[3](x, batch)  # gravnet
            x = block[4](x)  # batchnorm
            feat.append(x)  # skip connections

        x = torch.cat(feat, dim=1)

        # final layers
        x = F.relu(self.final1(x))  # linear
        x = F.relu(self.final2(x))  # linear
        x = self.final3(x)  # linear
        out = F.softmax(x, dim=-1)

        return out
