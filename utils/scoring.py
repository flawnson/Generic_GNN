""" This file contains all scoring methods for use throughout the project. Some are merely wrappers around SciKitLearn,
    others are custom scoring mechanisms for semi-supervised graph learning. The purpose of the class is to return one
    single score object containing all scores and str() and repr() methods for prettier printing"""

import numpy as np
import torch.functional as F

from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score, confusion_matrix
from utils.helper import auroc_score
from read.preprocessing import GenericDataset


class Scores:
    def __init__(self, score_config: dict, dataset: GenericDataset, logits, masks: list):
        self.score_config = score_config
        self.mask = np.logical_and(*masks)
        self.dataset = dataset
        self.logits = logits
        self.s_logits = F.softmax(input=logits[:, 1:], dim=1)  # To account for the unknown class
        self.prediction = logits[self.mask].max(1)[1]

    def accuracy(self, params):
        return self.prediction.eq(self.dataset.ndata["y"][self.mask]).sum().item() / self.mask.sum().item()

    def f1_score(self, params):
        return f1_score(y_true=self.dataset.ndata["y"][self.mask].to('cpu'),
                        y_pred=self.prediction.to('cpu'),
                        average=params[0])

    def auroc(self, params):
        return auroc_score(params=params,
                           dataset=self.dataset,
                           agg_mask=self.mask,
                           split_mask=self.mask[1],
                           logits=self.logits,
                           s_logits=self.s_logits)

    def confusion_mat(self, params):
        return confusion_matrix(y_true=self.dataset.ndata["y"][self.mask].to('cpu'),
                                y_pred=self.prediction.to('cpu'),
                                labels=None,
                                sample_weight=None,
                                normalize=None)

    def precision(self, params):
        return precision_score(y_true=self.dataset.ndata["y"][self.mask].to('cpu'),
                               y_pred=self.prediction.to('cpu'),
                               labels=None,
                               pos_label=1,
                               average=params[0],
                               sample_weight=None,
                               zero_division=params[1])

    def recall(self, params):
        return recall_score(y_true=self.dataset.ndata["y"][self.mask].to('cpu'),
                            y_pred=self.prediction.to('cpu'),
                            labels=None,
                            pos_label=1,
                            average=params[0],
                            sample_weight=None,
                            zero_division=params[1])

    def jaccard(self, params):
        return jaccard_score(y_true=self.dataset.ndata["y"][self.mask].to('cpu'),
                             y_pred=self.prediction.to('cpu'),
                             labels=None,
                             pos_label=1,
                             average=params[0],
                             sample_weight=None)

    def score(self):
        scoreset = {"acc": self.accuracy(self.score_config["acc"]),
                    "auc": self.auroc(self.score_config["auc"]),
                    "f1": self.f1_score(self.score_config["f1"]),
                    "con": self.confusion_mat(self.score_config["con"]),
                    "prec": self.precision(self.score_config["prec"]),
                    "rec": self.recall(self.score_config["rec"]),
                    "jac": self.jaccard(self.score_config["jac"])
                    }

        return [scoreset[score_type] for score_type, score_params, in self.score_config.items()]
