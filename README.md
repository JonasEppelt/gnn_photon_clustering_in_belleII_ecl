# Code for "Photon Reconstruction in the Belle II Calorimeter Using Graph Neural Networks"

This repository contains the code used in the paper "Photon Reconstruction in the Belle II Calorimeter Using Graph Neural Networks (https://arxiv.org/abs/2306.04179).

## Datasets

Two example datasets are provided in `./data` as pandas dataframes stored as `single_photon_data.parquet` and `two_photon_data.parquet`. The datasets contain, respectively, 100 events of isolated photons and 100 events of overlapping photons.

For each event all nodes in the ROI are stored with their relevant features and the global photon features.

## Training

Two training examples are provided in the jupyter notebooks `one_photon_example.ipynb` and `two_photon_example.ipynb` for training on single photon and overlapping photon events respectively. Configs for both examples are provided in `./configs`.

The GravNet model is implemented in `./model.py` while the graph dataset can be generated out of the dataframes by using `./datasets.py`.

## Inference

Trained models can be used for inference on events using the `./inference.py` script. We do not provide trained models.