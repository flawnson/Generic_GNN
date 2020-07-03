import networkx as nx
import os.path as osp
import pandas as pd
import numpy as np
import torch
import json
import dgl

from abc import abstractmethod, ABC


class GenericDataset(ABC):
    """
    Abstract base class from which subclasses will inherit from.
    Currently the intended use is for datasets with multiple labelsets.
    Modifications can be made to compensate for multiple graph edgelists, node features, etc.
    """
    @abstractmethod
    def __init__(self, config: dict):
        super(GenericDataset, self).__init__()
        self.data_config = config
        self.dataset = self.preprocessing()

    def get_edges(self) -> nx.DiGraph:
        edgelist_path = osp.join(osp.dirname(osp.dirname(__file__)),
                                 osp.join(*self.data_config["directory"]),
                                 self.data_config["edgelist_file"])

        # For PyG
        edgeframe = pd.read_csv(edgelist_path, header=0, index_col=False)

        # DGL will turn graph into directed graph regardless of networkx object type
        edges = nx.readwrite.edgelist.read_edgelist(edgelist_path, delimiter=",", create_using=nx.DiGraph)

        return edges

    @staticmethod
    def get_edge_features():
        # TODO: Add adapter (put in utils dir) for csv, json, and txt files
        # TODO: Make edge and node features optional (and implement for dummy)
        pass

    def get_node_features(self) -> dict:
        features_dict = json.load(open(osp.join(osp.dirname(osp.dirname(__file__)),
                                                *self.data_config["directory"],
                                                self.data_config["features_file"])))

        features_dict = {name: embed for name, embed
                         in zip(features_dict["gene"].values(),
                                features_dict["embeddings"].values())}

        if self.data_config["dummy_features"]:
            features_dict = {key: np.ones(self.data_config["dummy_features"], dtype=np.double) for key, value
                             in features_dict.items()}

        return features_dict

    def get_targets(self) -> dict:
        node_labels = pd.read_csv(osp.join(osp.dirname(osp.dirname(__file__)),
                                           *self.data_config["directory"],
                                           self.data_config["label_file"]), header=0)

        target_data = [node_labels[name].tolist() for name in self.data_config["label_names"]]

        return self.get_labels(target_data)

    @abstractmethod
    def get_labels(self, target_data: list):
        # Must be implemented by subclasses. File reading logic is held in get_targets()
        return {}

    def intersection(self) -> nx.DiGraph:
        nx_graph: nx.graph = self.get_edges()
        target_data: dict = self.get_targets()

        # Needed to compensate for differences between target set and edgelist (assign nodes without labels to unknown)
        target_data = {name: 0 if name not in target_data else target_data[name] for name in nx_graph.nodes()}

        nx.set_node_attributes(nx_graph, target_data, "y")
        nx.set_node_attributes(nx_graph, self.get_node_features(), "x")

        # Filter for nodes with embeddings
        [nx_graph.remove_node(n) for (n, d) in nx_graph.copy().nodes(data=True) if "x" not in d]

        return nx_graph

    def preprocessing(self) -> dgl.graph:
        # TODO: Logic for turning data objects into library-specific implementations
        # TODO: Check if node list is getting rearranged during conversion to dgl graph object
        nx_graph = self.intersection()
        dgl_graph = dgl.DGLGraph()
        dgl_graph.from_networkx(nx_graph, node_attrs=["x"])
        dgl_graph.y = nx_graph.nodes("y")

        dgl_graph.known_mask = np.array(list(dgl_graph.y)) != 0 if self.data_config["semi-supervised"] else torch.ones(
            dgl_graph.y.shape[0])

        return dgl_graph


class PrimaryLabelset(GenericDataset, ABC):
    def __init__(self, config: dict):
        super(PrimaryLabelset, self).__init__(config=config)

    @staticmethod
    def extract_labels(target_data) -> dict:
        # Labels starts from 1, since 0 is reserved for unknown class in the case of semi-supervised learning
        # Can manually create dictionary that maps from data to integer
        # Note that PyG automatically turns ints into onehot
        return dict(zip(np.unique(target_data[1]), list(range(1, len(np.unique(target_data[1]))))))

    def get_labels(self, target_data) -> dict:
        # Abstract method defined in GenericDataset
        # return {name: self.extract_labels(target_data) for name, label in zip(target_data[0], target_data[1])}
        return [self.extract_labels(target_data)[name] for name, label in zip(target_data[0], target_data[1])]
