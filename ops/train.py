""" This module contains the code that defined the training loop of the pipeline. The train and test methods are executed
    on a per-epoch basis. Note that DGL accesses graph features and labels differently from PyTorch Geometric (DGL uses
    indexing of the ndata attribute whereas PyG allows access to attributes themselves)"""
import torch.nn.functional as F
import numpy as np
import torch
import dgl

from utils.helper import loss_weights, auroc_score
from sklearn.metrics import f1_score
from nn.DGL_models import GNNModel


class Trainer:
    def __init__(self, train_config: dict, dataset: dgl.DGLGraph, model: GNNModel, device: torch.device):
        """ Class for training and testing loops

        Args:
            train_config:
            dataset:
            model:
            device:

        Returns:

        """
        self.train_config = train_config
        self.dataset = dataset
        self.model = model
        self.device = device
        self.params = model.parameters()
        self.optimizer = torch.optim.Adam(self.params, lr=self.train_config["lr"], weight_decay=self.train_config["wd"])

    def train(self) -> torch.tensor:
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(self.dataset, self.dataset.ndata["x"])
        agg_mask = np.logical_and(self.dataset.splits[0], self.dataset.known_mask)
        weights = loss_weights(self.dataset, agg_mask, self.device) if self.train_config["weighted_loss"] else None
        loss = F.cross_entropy(logits[agg_mask], self.dataset.ndata["y"][agg_mask].long().to(self.device),
                               weight=weights)
        loss.backward(retain_graph=True)
        self.optimizer.step()

        return loss

    @torch.no_grad()
    def test(self):
        self.model.eval()
        logits = self.model(self.dataset, self.dataset.ndata["x"])
        accs, auroc_scores, f1_scores = [], [], []
        s_logits = F.softmax(input=logits[:, 1:], dim=1)  # To account for the unknown class

        for mask in self.dataset.splits:
            agg_mask = np.logical_and(mask, self.dataset.known_mask)  # Combined both masks
            pred = logits[agg_mask].max(1)[1]

            acc = pred.eq(self.dataset.ndata["y"][agg_mask].to(self.device)).sum().item() / agg_mask.sum().item()
            f1 = f1_score(y_true=self.dataset.ndata["y"][agg_mask].to('cpu'), y_pred=pred.to('cpu'), average='macro')
            auroc = auroc_score(self.dataset, agg_mask, mask, logits, s_logits)

            accs.append(acc)
            f1_scores.append(f1)
            auroc_scores.append(auroc)

        return accs, f1_scores, auroc_scores

    def pred(self):
        # TODO: Implement prediction method and logging
        self.model.eval()
        logits = self.model(self.dataset, self.dataset.ndata["x"])

    def run(self):
        for epoch in range(self.train_config["epochs"]):
            print(f"Epoch: {epoch}", "-" * 20)
            loss = self.train()
            score = self.test()
