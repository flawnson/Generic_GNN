import torch

from read.preprocessing import GenericDataset
from nn.DGL_models import GNNModel


class Trainer:
    def __init__(self, train_config: dict, dataset: GenericDataset, model: GNNModel):
        self.train_config = train_config
        self.dataset = dataset
        self.model = model
        self.params = model.parameters()
        self.optim = torch.optim.Adam(self.params, lr=self.train_config["lr"], weight_decay=self.train_config["wd"])

    @torch.no_grad()
    def train(self) -> torch.tensor:
        self.model.train()
        loss = self.optim.step()

        return loss

    def test(self):
        self.model.test()

        return None

    def run(self):
        for epoch in self.train_config["epochs"]:
            loss = self.train()
            score = self.test()

