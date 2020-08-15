""" This file contains all scoring methods for use throughout the project. Some are merely wrappers around SciKitLearn,
    others are custom scoring mechanisms for semi-supervised graph learning. The purpose of the class is to return one
    single score object containing all scores and str() and repr() methods for prettier printing"""

import numpy as np

from sklearn.metrics import f1_score
from utils.helper import auroc_score
from read.preprocessing import GenericDataset


class Scores:
    def __init__(self, score_config: dict, dataset: GenericDataset, prediction, masks: list):
        self.score_config = score_config
        self.prediction = prediction
        self.dataset = dataset
        self.mask = np.logical_and(*masks)

    def accuracy(self):
        return self.prediction.eq(self.dataset.ndata["y"][self.mask]).sum().item() / self.mask.sum().item()
