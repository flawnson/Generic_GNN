""" This module contains miscellaneous code that takes up too much space in the main pipeline, hence have been abstracted and
    moved here for organizational reasons """
import numpy as np
import json
import torch
import argparse
import jsonschema
import subprocess
import os.path as osp
import torch.nn.functional as F

from typing import Tuple, Dict
from utils.logger import set_file_logger, log
from nn.DGL_models import GenericGNNModel
from sklearn.metrics import roc_auc_score
from collections.abc import Iterable
from torch.utils.tensorboard import SummaryWriter


def parse_arguments() -> Tuple[Dict, torch.device]:
    path = osp.join('data', 'biogrid')
    parser = argparse.ArgumentParser(description="Config file parser")
    parser.add_argument("-c", "--config", help="json config file", type=str)
    parser.add_argument("-s", "--scheme", help="json scheme file", type=str)
    args = parser.parse_args()

    json_data: dict = json.load(open(args.config))
    json_scheme: dict = json.load(open(args.scheme))
    set_file_logger(json_data)
    # DGL Overrides torch.tensor.to() and implements it's own to() method for its graph objects
    device = torch.device("cuda" if json_data["cuda"] and torch.cuda.is_available() else "cpu")
    log.info(f"Using {device} for compute")

    # See https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    # benchmark mode is good whenever your input sizes for your network do not vary
    torch.backends.cudnn.benchmark = True if not json_data.get("cuda", False) and torch.cuda.is_available() else False

    git_hash = subprocess.check_output(["git", "describe", "--always"]).strip().decode()
    git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode()
    log.info(f"Git hash: {git_hash}, branch: {git_branch}")

    if json_data["validate"]:
        try:
            jsonschema.validate(json_data, json_scheme)
        except jsonschema.ValidationError:
            log.warning("Tried to validate but failed, continuing anyway")

    return json_data, device



def loss_weights(dataset, agg_mask: np.ndarray, device: torch.device) -> torch.tensor:
    """ These weights are designed to compensate for class imabalance in the dataset (negated effects if class has
        undergone oversampling or undersampling) """
    imb_wc = torch.bincount(dataset.ndata["y"][agg_mask], minlength=int(dataset.ndata["y"].max())).float().clamp(
        min=1e-10, max=1e10) / dataset.ndata["y"][agg_mask].shape[0]
    weights = (1 / imb_wc) / (sum(1 / imb_wc))

    return weights.to(device)


def auroc_score(params: list, dataset, agg_mask: np.ndarray, split_mask, logits: torch.tensor, s_logits) -> float:
    """ Logic for calculating roc_auc_score using sklearn (different configurations needed depending on the labelset
        and task type """
    if len(np.unique(dataset.ndata["y"].numpy())) == 2:
        auroc = roc_auc_score(y_true=dataset.y[agg_mask].to('cpu').numpy(),
                              y_score=np.amax(s_logits[agg_mask].to('cpu').data.numpy(), axis=1),
                              average=params[0],  # Should be None in binary case
                              multi_class=params[1])  # Should be None in binary case
    else:
        output_mask = np.isin(list(range(dataset.ndata["y"][split_mask].max())),
                              np.unique(dataset.ndata["y"][split_mask].to('cpu').numpy()))
        test = np.apply_along_axis(func1d=lambda arr: arr[output_mask], axis=1,
                                   arr=logits[:, 1:].to('cpu').data.numpy())
        s_logits = F.softmax(input=torch.from_numpy(test), dim=1)  # Recalc of s_logits from outer scope

        auroc = roc_auc_score(y_true=dataset.ndata["y"][agg_mask].to('cpu').numpy(),
                              y_score=s_logits[agg_mask],
                              average=params[0],
                              multi_class=params[1])

    return auroc


def pretty_print(scores: dict) -> None:
    # Function to use for printing model scores in training pipeline
    for score_type, score_set in scores.items():
        try:
            for split_name, score in score_set.items():
                print(f"{score_type + '-' + split_name}: {round(score, 3)}")
            print("-" * 10)
        except TypeError:
            for split_name, score in score_set.items():
                print(f"{score_type + '-' + split_name}: {score}")
            print("-" * 10)


def save_model(config: dict, epoch: int, model: GenericGNNModel) -> None:
    if config.get("save_model", False) and epoch == config["epochs"]:
        torch.save(model.state_dict, osp.join(osp.dirname(__file__), "output", config["save_model"]))


def load_model(config: dict, model: GenericGNNModel, device: torch.device) -> GenericGNNModel:
    # When loading a model on a CPU that was trained with a GPU, pass torch.device('cpu')
    # to the map_location argument in the torch.load() function.
    # In this case, the storages underlying the tensors are dynamically remapped
    # to the CPU device using the map_location argument.
    if config.get("load_model", None):
        try:
            return model.load_state_dict(torch.load(osp.join("outputs", config["load_model"]), map_location=device))
        except:
            RuntimeError("Pretrained weights do not seem to fit the provided model; check if pretrained model exists")
    else:
        return model.to(device)

