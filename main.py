import os.path as osp
import argparse
import torch
import json

from read.preprocessing import GenericDataset, PrimaryLabelset


if __name__ == "__main__":
    # TODO: Implement logger to log progress of code execution
    # TODO: Create directory tree and requirements.txt with bash script
    # TODO: Add __init__.py files to each project directory

    path = osp.join('data', 'biogrid')
    parser = argparse.ArgumentParser(description="Config file parser")
    parser.add_argument("-c", "--config", help="json config file", type=str)
    parser.add_argument("-b", "--benchmarking", help="benchmarking run config", type=str)
    args = parser.parse_args()
    bm = args.benchmarking

    json_data: dict = json.load(open(args.config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = PrimaryLabelset(json_data["data_config"])
    print(dataset.dataset)

