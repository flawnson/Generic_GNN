import os.path as osp
import argparse
import torch
import json

from read.preprocessing import GenericDataset, PrimaryLabelset
from nn.DGL_models import GenericGNNModel, GNNModel


if __name__ == "__main__":
    # TODO: Implement logger to log progress of code execution
    # TODO: Create directory tree and requirements.txt with bash script

    path = osp.join('data', 'biogrid')
    parser = argparse.ArgumentParser(description="Config file parser")
    parser.add_argument("-c", "--config", help="json config file", type=str)
    parser.add_argument("-b", "--benchmarking", help="benchmarking run config", type=str)
    args = parser.parse_args()
    bm = args.benchmarking

    json_data: dict = json.load(open(args.config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset: GenericDataset = None
    dataset = PrimaryLabelset(json_data["data_config"])

    # Models are defined in DGL_models.py. You may build you custom layer with DGL in DGL_layers.py or use an
    # Off-the-shelf layer from DGL. You many define a list of layer types to use in the json config file, otherwise
    # you must provide a string with the name of the layer to use for the entire model

    model: GenericGNNModel = None
    model = GNNModel(json["model_config"], dataset, device, pooling=None)

