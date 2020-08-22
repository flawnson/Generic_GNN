""" There are two types of runs: demo, tuning, and benchmarking. All configurations are defined in the config directory
    demo: runs the training pipeline once (Includes option to use pre-trained weights)
    tuning: runs the tuning pipeline the number of times defined in config file
    benchmarking runs the tuning pipeline multiple times and then runs the training pipeline once and logs run details
    Documentation is written Google-style (minus examples) with type annotations as per pep484 for most of this project
    """

# TODO: Fix holdout methods
# TODO: Implement tuning for model and layer sizes
# TODO: Implement optimizer/loss customization
# TODO: Implement basic multi-GPU support
# TODO: Implement basic unit testing for splits
import os.path as osp
import subprocess
import argparse
import torch
import json

from read.preprocessing import GenericDataset, PrimaryLabelset
from utils.holdout import Holdout
from utils.logger import set_file_logger, log
from ops.benchmark import Benchmarker
from ops.train import Trainer
from ops.tune import Tuner


if __name__ == "__main__":
    path = osp.join('data', 'biogrid')
    parser = argparse.ArgumentParser(description="Config file parser")
    parser.add_argument("-c", "--config", help="json config file", type=str)
    parser.add_argument("-d", "--device", help="device to use", type=bool)
    args = parser.parse_args()

    json_data: dict = json.load(open(args.config))
    set_file_logger(json_data)
    # DGL Overrides torch.tensor.to() and implements it's own to() method for its graph objects
    device = torch.device("cuda" if json_data["cuda"] and torch.cuda.is_available() else "cpu")
    log.info(f"Using {device} for compute")

    # See https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    # benchmark mode is good whenever your input sizes for your network do not vary
    torch.backends.cudnn.benchmark = True if not args.device and torch.cuda.is_available() else False

    git_hash = subprocess.check_output(["git", "describe", "--always"]).strip().decode()
    git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode()
    log.info(f"Git hash: {git_hash}, branch: {git_branch}")

    # Use if-else to check if requested dataset and model type (from config file) is available
    log.info(f"Creating {json_data['dataset']} dataset")
    dataset: GenericDataset = None
    if json_data["dataset"] == "primary_labelset":
        dataset = PrimaryLabelset(json_data["data_config"]).dataset.to(device)
    else:
        raise NotImplementedError(f"{json_data['dataset']} is not a dataset")  # Add to logger when implemented

    # You must use balanced split (auroc doesn't work otherwise)
    log.info("Creating splitsets")
    dataset.splits = Holdout(json_data["data_config"], dataset, bool_mask=True).balanced_split()

    # Runtype pipelines
    log.info(f"Executing {json_data['run_type']} pipeline")
    if json_data["run_type"] == "demo":
        Trainer(json_data, dataset, device).run_train()
    elif json_data["run_type"] == "tune":
        Tuner(json_data, dataset, device).run_tune()
    elif json_data["run_type"] == "benchmark":
        Benchmarker(json_data, dataset, device).run_benchmark()
    else:
        raise NotImplementedError(f"{json_data['run_type']} is not a valid run type")
