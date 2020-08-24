# This file is for code related to visualizing the run pipelines and the dataset
import torch
import numpy as np
import networkx as nx


class VisualizeData(object):
    def __init__(self, config, dataset, device):
        self.visual_config = config["visual_config"]
        self.dataset = dataset
        self.nodes = dataset.nodes() if not self.visual_config["subset"] else self.subset()
        self.device = device

    def subset(self):
        """Method to take a subset of nodes in the graph"""
        graph_item = self.dataset.nodes() if self.visual_config["subset"].keys() == "nodes" else self.dataset.edges()
        if isinstance(self.visual_config["subset"].value(), int):
            np.choice(graph_item, )
        elif isinstance(self.visual_config["subset"], float):
            np.choice(self.dataset, )


class VisualizeTraining(object):

