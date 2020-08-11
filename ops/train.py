""" This module contains the code that defined the training loop of the pipeline. The train and test methods are executed
    on a per-epoch basis. Note that DGL accesses graph features and labels differently from PyTorch Geometric (DGL uses
    indexing of the ndata attribute whereas PyG allows access to attributes themselves)"""

import torch.nn.functional as F
import os.path as osp
import numpy as np
import torch
import dgl

from nn.DGL_models import GenericGNNModel, GNNModel
from utils.helper import loss_weights, auroc_score, save_model, pretty_print, load_model
from read.preprocessing import GenericDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score


class Trainer:
    def __init__(self, train_config: dict, dataset: GenericDataset, device: torch.device):
        """ Class for training and testing loops

        Args:
            train_config: The configurations to use for training pipeline
            dataset: The DGL dataset object
            model: The DGL model object
            device: The device to use during training (either "gpu" or "cpu")

        Returns:

        """
        self.train_config = train_config
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
        agg_mask = np.logical_and(self.dataset.splits["trainset"], self.dataset.known_mask)
        weights = loss_weights(self.dataset, agg_mask, self.device) if self.train_config["weighted_loss"] else None
        loss = F.cross_entropy(logits[agg_mask], self.dataset.ndata["y"][agg_mask].long().to(self.device), weight=weights)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        save_model(self.train_config, epoch, self.model)

        return loss

    @torch.no_grad()
    def test(self):
        self.model.eval()
        logits = self.model(self.dataset, self.dataset.ndata["x"])
        accs, auroc_scores, f1_scores = [], [], []
        s_logits = F.softmax(input=logits[:, 1:], dim=1)  # To account for the unknown class

        for mask in self.dataset.splits.items():
            agg_mask = np.logical_and(mask[1], self.dataset.known_mask)  # Combined both masks
            pred = logits[agg_mask].max(1)[1]

            accs.append((mask[0], pred.eq(self.dataset.ndata["y"][agg_mask].to(self.device)).sum().item() / agg_mask.sum().item()))
            f1_scores.append((mask[0], f1_score(y_true=self.dataset.ndata["y"][agg_mask].to('cpu'),
                                      y_pred=pred.to('cpu'),
                                      average='macro')))
            auroc_scores.append((mask[0], auroc_score(self.dataset, agg_mask, mask[1], logits, s_logits)))

        return {"acc": accs, "f1": f1_scores, "auc": auroc_scores}

    def pred(self):
        # TODO: Implement prediction method and logging
        self.model.eval()
        logits = self.model(self.dataset, self.dataset.ndata["x"])

    def write(self, epoch: int, scores: dict) -> None:
        self.writer.add_graph(self.model, self.dataset, verbose=True)
        for score_type, score_set in scores.items():
            for score_split in score_set:
                self.writer.add_scalar(score_type + score_split[0], score_split[1], epoch)
        # self.writer.flush()

    def get_model(self):
        model: GenericGNNModel = None
        if self.train_config["model_config"]["model"] == "GAT":
            model = GNNModel(self.train_config["model_config"], self.dataset, self.device, pooling=None).to(self.device)
        else:
            raise NotImplementedError(f"{self.train_config['model_config']['model']} is not a model")  # Add to logger when implemented

        load_model(self.train_config, model, self.device)

    def run_train(self):
        for epoch in range(self.train_config["epochs"]):
            print(f"Epoch: {epoch}", "-" * 20)

            loss = self.train(epoch)
            print(f"Loss: {loss}", "\n", "_" * 10)

            scores = self.test()
            pretty_print(scores)

            if self.train_config.get("write_summary", False):
                self.write(epoch, scores)
