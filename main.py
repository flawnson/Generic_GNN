""" There are two types of runs: demo, tuning, and benchmarking. All configurations are defined in the config directory
    demo: runs the training pipeline once (Includes option to use pre-trained weights)
    tuning: runs the tuning pipeline the number of times defined in config file
    benchmarking runs the tuning pipeline multiple times and then runs the training pipeline once and logs run details
    Documentation is written Google-style (minus examples) with type annotations as per pep484 for most of this project
    """

# TODO: Add sample datasets from public repositories as examples
# TODO: Use SciKitLearn's mechanisms to implement custom size splits
# TODO: Implement tuning for model and layer sizes
# TODO: Get screenshots of model training and rataset
# TODO: Turn Tuning into Training wrapper (check if tuning pipeline still functional)
# TODO: Implement optimizer/loss customization
# TODO: Implement basic multi-GPU support
# TODO: Implement basic unit testing for splits
import os.path as osp
import jsonschema
import torch

from read.preprocessing import GenericDataset, PrimaryLabelset
from utils.holdout import Holdout
from utils.helper import parse_arguments
from utils.logger import set_file_logger, log, timed
from ops.benchmark import Benchmarker
from ops.train import Trainer
from ops.tune import Tuner
from nn.DGL_models import GenericGNNModel, GNNModel


if __name__ == "__main__":
    json_data, device = parse_arguments()

    # Use if-else to check if requested dataset and model type (from config file) is available
    log.info(f"Creating {json_data['dataset']} dataset")
    dataset: GenericDataset = None
    if json_data["dataset"] == "primary_labelset":
        dataset = PrimaryLabelset(json_data).dataset.to(device)
    else:
        raise NotImplementedError(f"{json_data['dataset']} is not a dataset")  # Add to logger when implemented

    # Use if-else to check if requested dataset and model type (from config file) is available
    log.info(f"Creating {json_data['dataset']} dataset")

    # You must use balanced split (auroc doesn't work otherwise)
    log.info("Creating splitsets")
    dataset.splits = Holdout(json_data, dataset).split()

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
