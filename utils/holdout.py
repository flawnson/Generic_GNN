"""This file contains the code used to partition the graph dataset into sets for each run type."""

import numpy as np
import torch

from typing import Dict
from dgl.data.utils import Subset
from itertools import accumulate
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from utils.logger import timed

DEFAULT_SPLITS = {"trainset": 0.8, "validset": 0.1, "testset": 0.1}


class Holdout:
    def __init__(self, config: dict, dataset, bool_mask=True):
        """ Class for sequestering splits for training and validation

        Args:
            data_config: The configurations to use for splitting the data
            dataset: The DGL dataset object
            bool_mask: Whether of not to turn list indices into boolean masks

        Returns:
            list

        """
        super(Holdout, self)
        self.config = config
        self.data_config = self.config["data_config"]
        self.dataset = dataset
        self.bool_mask = bool_mask

    def naive_split(self) -> dict:
        frac_list = np.asarray(list(self.data_config.get("splits", DEFAULT_SPLITS).values()))
        # assert np.allclose(np.sum(frac_list), 1.), f'Expected frac_list to sum to 1, got {np.sum(frac_list)}'

        num_data = len(self.dataset)
        lengths = (num_data * frac_list).astype(int)
        lengths[-1] = num_data - np.sum(lengths[:-1])

        if self.data_config["shuffle"]:
            indices = np.random.RandomState(seed=None).permutation(num_data)
        else:
            indices = np.arange(num_data)

        if self.bool_mask:
            return self.indices_to_mask([Subset(self.dataset, indices[offset - length:offset]) for offset, length in
                                         zip(accumulate(lengths), lengths)])
        else:
            # https://docs.dgl.ai/en/0.4.x/api/python/data.html?highlight=subset#dgl.data.utils.Subset
            return dict(zip(self.data_config["splits"].keys(), [Subset(self.dataset, indices[offset - length:offset])
                            for offset, length in zip(accumulate(lengths), lengths)]))

    def tri_split(self):
        """
        Temporary method to split dataset until rest of codebase is complete
        :return: train, test, val for DGL object
        """
        frac_list = np.asarray(list(self.data_config.get("splits", DEFAULT_SPLITS).values()))

        # Initialize empty boolean arrays
        booleans = []
        for frac in frac_list:
            boolean = np.zeros(len(self.dataset.ndata["y"]), dtype=bool)
            booleans.append(boolean)

        class_indices = []
        for class_label in range(len(np.unique(self.dataset.ndata["y"])) + 1):
            mask = self.dataset.ndata["y"].numpy() == class_label
            class_indices.append(np.where(mask)[0])

        train_idx = []
        test_idx = []
        val_idx = []

        for index_list in class_indices:
            train_len: int = int(round(frac_list[0] * len(index_list)))
            validation_len: int = int(round(frac_list[1] * len(index_list)))

            train_indices = np.random.choice(index_list, train_len, replace=False)
            leftover = np.setdiff1d(index_list, train_indices)
            val_indices = np.random.choice(leftover, validation_len, replace=False)
            test_indices = np.setdiff1d(leftover, val_indices)

            train_idx += train_indices.tolist()
            test_idx += test_indices.tolist()
            val_idx += val_indices.tolist()

        # Change all False to True at indices
        for boolean, indices in zip(booleans, [train_idx, test_idx, val_idx]):
            boolean[indices] = True

        # return {"train_mask": booleans[0], "test_mask": booleans[1], "val_mask": booleans[2]}
        return dict(zip(self.data_config["splits"], booleans))

    def stratified_split(self) -> dict:
        # See SciKitLearn's documentation for implementation details (note that this method enforces same size splits):
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
        split = StratifiedKFold(n_splits=len(self.data_config["splits"]), shuffle=self.data_config["shuffle"])
        # split = StratifiedShuffleSplit(n_splits=len(self.data_config["splits"]))
        masks = list(split._iter_test_masks(self.dataset.ndata["x"], self.dataset.ndata["y"]))
        # test = split.get_n_splits(self.dataset.ndata["x"], self.dataset.ndata["y"])

        return dict(zip(self.data_config["splits"].keys(), masks))

    def _iter_test_masks(self, X, y=None, groups=None):
        test_folds = self._make_test_folds(X, y)
        for i in range(self.n_splits):
            yield test_folds == i

    @timed
    def split(self) -> Dict[str, object]:
        if self.data_config["split_type"] == "naive":
            # Naive split sizes (does not compensate for class imbalance
            return self.naive_split()
        elif self.data_config["split_type"] == "tri":
            # Any size splits of evenly distributed classes, but fixed at 3 sets
            return self.tri_split()
        elif self.data_config["split_type"] == "stratified":
            # Any number of sets of evenly distributed classes, but equal split sizes
            return self.stratified_split()
