""" There are two types of runs: demo, tuning, and benchmarking. All configurations are defined in the config directory
    demo: runs the training pipeline once
    tuning: runs the tuning pipeline the number of times defined in config file
    benchmarking runs the tuning pipeline multiple times and then runs the training pipeline once, with logs """
# TODO: Implement logger to log progress of code execution
# TODO: Create directory tree and requirements.txt with bash script
import os.path as osp
import argparse
import torch
import json

from read.preprocessing import GenericDataset, PrimaryLabelset
from nn.DGL_models import GenericGNNModel, GNNModel
from utils.holdout import Holdout
from ops.train import Trainer


if __name__ == "__main__":
    path = osp.join('data', 'biogrid')
    parser = argparse.ArgumentParser(description="Config file parser")
    parser.add_argument("-c", "--config", help="json config file", type=str)
    args = parser.parse_args()

    json_data: dict = json.load(open(args.config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use if-else to check if requested dataset and model type (from config file) is available
    if True:
        dataset: GenericDataset = None
        dataset = PrimaryLabelset(json_data["data_config"]).dataset.to(device)

    dataset.splits = Holdout(json_data["data_config"], dataset, bool_mask=True).balanced_split()

    # Models are defined in DGL_models.py. You may build you custom layer with DGL in DGL_layers.py or use an
    # Off-the-shelf layer from DGL. You many define a list of layer types to use in the json config file, otherwise
    # you must provide a string with the name of the layer to use for the entire model

    if True:
        model: GenericGNNModel = None
        model = GNNModel(json_data["model_config"], dataset, device, pooling=None).to(device)

    Trainer(json_data["train_config"], dataset, model, device).run()
