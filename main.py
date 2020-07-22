""" There are two types of runs: demo, tuning, and benchmarking. All configurations are defined in the config directory
    demo: runs the training pipeline once
    tuning: runs the tuning pipeline the number of times defined in config file
    benchmarking runs the tuning pipeline multiple times and then runs the training pipeline once and logs run details
    Documentation is written Google-style (minus examples) with type annotations as per pep484 for most of this project
    """
# TODO: Implement logger to log progress of code execution
# TODO: Implement tuning pipeline and benchmarking pipeline
# TODO: Implement optimizer customization
# TODO: Implement basic multi-GPU support
# TODO: Implement model saving and loading
# TODO: Implement basic unit testing for splits
# TODO: Implement VAE to generate dataset features
# TODO: Implement linear model to validate dataset features
# TODO: Implement TensorFlow summarywriter for logging training metrics
# TODO: Create directory tree and requirements.txt with bash script
import os.path as osp
import subprocess
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
    # See https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    # git_hash = subprocess.check_output(["git", "describe", "--always"]).strip().decode()
    # git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode()
    # log.info(f"Git hash: {git_hash}, branch: {git_branch}")

    # Use if-else to check if requested dataset and model type (from config file) is available
    if json_data.get("data_config")["category"]:
        dataset: GenericDataset = None
        dataset = PrimaryLabelset(json_data["data_config"]).dataset.to(device)
    else:
        print(f"{json_data['model']} is not a model")  #Add to logger when implemented

    dataset.splits = Holdout(json_data["data_config"], dataset, bool_mask=True).split()

    # Models are defined in DGL_models.py. You may build you custom layer with DGL in DGL_layers.py or use an
    # Off-the-shelf layer from DGL. You many define a list of layer types to use in the json config file, otherwise
    # you must provide a string with the name of the layer to use for the entire model

    # Use if-else to check if requested model type (from config file) is available
    if json_data.get("model_config")["model"]:
        model: GenericGNNModel = None
        model = GNNModel(json_data["model_config"], dataset, device, pooling=None).to(device)
    else:
        print(f"{json_data['model']} is not a model")

    Trainer(json_data["train_config"], dataset, model, device).run()
