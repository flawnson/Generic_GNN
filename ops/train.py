""" This module contains the code that defined the training loop of the pipeline. The train and test methods are executed
    on a per-epoch basis. Note that DGL accesses graph features and labels differently from PyTorch Geometric (DGL uses
    indexing of the ndata attribute whereas PyG allows access to attributes themselves)"""

import torch.nn.functional as F
import os.path as osp
import numpy as np
import torch
import dgl

from typing import Dict
from nn.DGL_models import GenericGNNModel, GNNModel
from utils.helper import loss_weights, auroc_score, save_model, pretty_print, load_model
from utils.scoring import Scores
from read.preprocessing import GenericDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

DEFAULT_SCORES = {"acc": None, "f1": ["macro"], "auc": ["macro", "ovo"]}
WRITE_SUMMARY = False


class Trainer(object):
    def __init__(self, config: dict, dataset: GenericDataset, device: torch.device):
        """ Class for training and testing loops

        Args:
            train_config: The configurations to use for training pipeline
            dataset: The DGL dataset object
            model: The DGL model object
            device: The device to use during training (either "gpu" or "cpu")

        Returns:

        """
        self.train_config = config["train_config"]
        self.dataset = dataset
        self.device = device
        self.model = self.get_model()
        self.params = self.model.parameters()
        self.optimizer = torch.optim.Adam(self.params, lr=self.train_config["lr"], weight_decay=self.train_config["wd"])
        self.writer = SummaryWriter("../logs" + self.train_config["run_name"])

    def train(self, epoch) -> torch.tensor:
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(self.dataset, self.dataset.ndata["x"])
        agg_mask = np.logical_and(list(self.dataset.splits.values())[0], self.dataset.known_mask)
        weights = loss_weights(self.dataset, agg_mask, self.device) if self.train_config["weighted_loss"] else None
        loss = F.cross_entropy(logits[agg_mask], self.dataset.ndata["y"][agg_mask].long().to(self.device), weight=weights)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        save_model(self.train_config, epoch, self.model)

        return loss

    @torch.no_grad()
    def test(self) -> Dict[str, list]:
        self.model.eval()
        logits = self.model(self.dataset, self.dataset.ndata["x"])
        score_dict = {score_type: {} for score_type, params in
                      self.train_config.get("scores", DEFAULT_SCORES.items()).items()}

        for split_name, mask in self.dataset.splits.items():
            scores = Scores(self.train_config.get("scores", DEFAULT_SCORES),
                            self.dataset,
                            logits,
                            mask,
                            self.dataset.known_mask).score()

            # Slightly incomprehensible; renames key and assigns it to score object (either float or iterable)
            for score_name, score in scores.items():
                score_dict[score_name][split_name] = score

        return score_dict

    def pred(self):
        # TODO: Implement prediction method and logging
        self.model.eval()
        logits = self.model(self.dataset, self.dataset.ndata["x"])

    def write(self, epoch: int, scores: dict) -> None:
        # TODO: Fix DGLGraph not iterable error
        self.writer.add_graph(self.model, self.dataset, verbose=True)
        for score_type, score_set in scores.items():
            for score_split in score_set:
                self.writer.add_scalar(score_type + score_split[0], score_split[1], epoch)
        self.writer.flush()

    def get_model(self) -> GenericGNNModel:
        # Models are defined in DGL_models.py. You may build you custom layer with DGL in DGL_layers.py or use an
        # Off-the-shelf layer from DGL. You many define a list of layer types to use in the json config file, otherwise
        # you must provide a string with the name of the layer to use for the entire model

        model: GenericGNNModel = None
        if self.train_config["model_config"]["model"] == "GAT":
            model = GNNModel(self.train_config["model_config"], self.dataset, self.device, pooling=None).to(self.device)
        else:
            raise NotImplementedError(f"{self.train_config['model_config']['model']} is not a model")  # Add to logger when implemented

        return load_model(self.train_config, model, self.device)

    def run_train(self):
        for epoch in range(self.train_config["epochs"]):
            print(f"Epoch: {epoch}", "-" * 20)

            loss = self.train(epoch)
            print(f"Loss: {loss}", "\n", "_" * 10)

            scores = self.test()
            pretty_print(scores)

            if self.train_config.get("write_summary", WRITE_SUMMARY):
                self.write(epoch, scores)
