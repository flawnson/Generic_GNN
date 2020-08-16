""" This file contains all scoring methods for use throughout the project. Some are merely wrappers around SciKitLearn,
    others are custom scoring mechanisms for semi-supervised graph learning. The purpose of the class is to return one
    single score object containing all scores and str() and repr() methods for prettier printing"""

import numpy as np
import torch.functional as F

from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score
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

    def accuracy(self):
        return self.prediction.eq(self.dataset.ndata["y"][self.mask]).sum().item() / self.mask.sum().item()

    def f1_score(self):
        return f1_score(y_true=self.dataset.ndata["y"][self.mask].to('cpu'),
                       y_pred=self.prediction.to('cpu'),
                       average='macro')

    def auroc(self):
        return auroc_score(self.dataset, self.mask, self.mask[1], self.logits, self.s_logits)

    def tps(self):
        return [((self.prediction == i) & (target == i)).sum() for i in range(dataset)]

    def fps(self):
        pass

    def fns(self):
        pass

    def tns(self):
        pass

    def mean_iou(self):
        pass

    def precision(self):
        return precision_score(y_true=self.dataset.ndata["y"][self.mask].to('cpu'),
                               y_pred=self.prediction.to('cpu'),
                               labels=None,
                               pos_label=1,
                               average='binary',
                               sample_weight=None,
                               zero_division='warn')

    def recall(self):
        return recall_score(y_true=self.dataset.ndata["y"][self.mask].to('cpu'),
                            y_pred=self.prediction.to('cpu'),
                            labels=None,
                            pos_label=1,
                            average='binary',
                            sample_weight=None,
                            zero_division='warn')

    def jaccard(self):
        return jaccard_score(y_true=self.dataset.ndata["y"][self.mask].to('cpu'),
                             y_pred=self.prediction.to('cpu'),
                             labels=None,
                             pos_label=1,
                             average='binary',
                             sample_weight=None)

    def score(self):
        scoreset = {"acc": self.accuracy(),
                    "auc": self.auroc(),
                    "f1": self.f1_score(),
                    "tps": self.tps(),
                    "fps": self.fps(),
                    "fns": self.fns(),
                    "tns": self.tns(),
                    "iou": self.mean_iou(),
                    "prec": self.precision(),
                    "rec": self.recall(),
                    "jac": self.jaccard()
                    }

        return [scoreset[score_type(score_params)] for score_type, score_params, in self.score_config.items()]
