""" This file is for code related to visualizing the run pipelines and the dataset """
import numpy as np
import networkx as nx
from utils.logger import log


class VisualizeData(object):
    def __init__(self, config: dict, dataset: nx.Graph):
        self.visual_config = config["visual_config"]
        self.dataset = dataset
        self.graph = self.dataset if not self.visual_config["subset"] else self.subset()

    def _get_graph_size(self, graph_item: nx.Graph) -> np.array:
        """ Private method to get the graph size to be drawn """
        if isinstance(self.visual_config["size"], int):
            subset_graph = np.random.choice(graph_item, self.visual_config["size"])
        elif isinstance(self.visual_config["size"], float):
            size: int = self.visual_config["size"] * len(graph_item)
            subset_graph = np.random.choice(graph_item, int(size))
        else:
            raise TypeError("Defined size type not permitted")

        return subset_graph

    def subset(self) -> nx.Graph:
        """ Method to take a subset of nodes in the graph """
        if self.visual_config["subset"] == "nodes":
            graph_item = self.dataset.nodes()
            subset = self._get_graph_size(graph_item)
            graph = self.dataset.subgraph(subset)
        elif self.visual_config["subset"] == "edges":
            graph_item = self.dataset.edges()
            subset = self._get_graph_size(graph_item)
            graph = self.dataset.edge_subgraph(subset)
        else:
            log.info("Subset size not understood, using entire graph")
            graph = self.dataset

        return graph

    def draw(self):
        """ Drawing with built in networkx method """
        nx.drawing.draw_networkx(self.graph, **self.visual_config["kwargs"])

