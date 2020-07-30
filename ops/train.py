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
            train_config: The configurations to use for training pipeline
            dataset: The DGL dataset object
            model: The DGL model object
            device: The device to use during training (either "gpu" or "cpu")

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
        loss = F.cross_entropy(logits[agg_mask], self.dataset.ndata["y"][agg_mask].long().to(self.device), weight=weights)
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

            accs.append(pred.eq(self.dataset.ndata["y"][agg_mask].to(self.device)).sum().item() / agg_mask.sum().item())
            f1_scores.append(f1_score(y_true=self.dataset.ndata["y"][agg_mask].to('cpu'),
                                      y_pred=pred.to('cpu'),
                                      average='macro'))
            auroc_scores.append(auroc_score(self.dataset, agg_mask, mask, logits, s_logits))

        return {"acc": accs, "f1": f1_scores, "auc": auroc_scores}

    def pred(self):
        # TODO: Implement prediction method and logging
        self.model.eval()
        logits = self.model(self.dataset, self.dataset.ndata["x"])

    def run_train(self):
        for epoch in range(self.train_config["epochs"]):
            print(f"Epoch: {epoch}", "-" * 20)
            loss = self.train()
            print(f'Loss: {loss}')
            scores = self.test()
            print(f'Train_acc: {round(scores["acc"][0], 3)}, Test_acc: {round(scores["acc"][2], 3)}')
            print(f'Train_f1: {round(scores["f1"][0], 3)}, Test_f1: {round(scores["f1"][2], 3)}')
            print(f'Train_roc: {round(scores["auc"][0], 3)}, Test_roc: {round(scores["auc"][2], 3)}')

