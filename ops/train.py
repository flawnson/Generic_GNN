import torch
import numpy as np
import torch.nn.functional as F

from read.preprocessing import GenericDataset
from utils.helper import loss_weights
from nn.DGL_models import GNNModel


class Trainer:
    def __init__(self, train_config: dict, dataset: GenericDataset, model: GNNModel, device: torch.device):
        self.train_config = train_config
        self.dataset = dataset
        self.model = model
        self.device = device
        self.params = model.parameters()
        self.optimizer = torch.optim.Adam(self.params, lr=self.train_config["lr"], weight_decay=self.train_config["wd"])

    @torch.no_grad()
    def train(self) -> torch.tensor:
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(self.dataset, self.dataset.ndata["x"])
        agg_mask = np.logical_and(self.dataset.train_mask, self.dataset.known_mask)
        weights = loss_weights(self.dataset, agg_mask, self.device) if self.train_config["weighted_loss"] else None
        loss = F.cross_entropy(logits[agg_mask], self.data.y[agg_mask].long().to(self.device), weight=weights)
        loss.backward(retain_graph=True)
        self.optimizer.step()

        return loss

    def test(self):
        self.model.test()

        return None

    def run(self):
        for epoch in range(self.train_config["epochs"]):
            print(f"Epoch: {epoch}", "-" * 20)
            loss = self.train()
            score = self.test()
