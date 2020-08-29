"""This file contains the code used to partition the graph dataset into sets for each run type."""

import numpy as np
import torch

from dgl.data.utils import Subset
from itertools import accumulate
from sklearn.model_selection import KFold, StratifiedKFold


class Holdout:
    def __init__(self, data_config: dict, dataset, bool_mask=True):
        """ Class for sequestering splits for training and validation

        Args:
            data_config: The configurations to use for splitting the data
            dataset: The DGL dataset object
            bool_mask: Whether of not to turn list indices into boolean masks

        Returns:
            list

        """
        super(Holdout, self)
        self.data_config = data_config
        self.dataset = dataset
        self.bool_mask = bool_mask

    def split(self) -> list:
        frac_list = self.data_config.get("split_sizes", [0.8, 0.1, 0.1])

        frac_list = np.asarray(frac_list)
        assert np.allclose(np.sum(frac_list), 1.), 'Expected frac_list to sum to 1, got {:.4f}'.format(
            np.sum(frac_list))

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
            return [Subset(self.dataset, indices[offset - length:offset]) for offset, length in
                    zip(accumulate(lengths), lengths)]

    def temp_split(self):
        """
        Temporary method to split dataset until rest of codebase is complete
        :return: train, test, val for DGL object
        """
        frac_list = np.asarray(list(self.data_config.get("split_sizes", [0.8, 0.1, 0.1]).values()))

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

        return {"train_mask": booleans[0], "test_mask": booleans[1], "val_mask": booleans[2]}

    def balanced_split(self):
        split = StratifiedKFold(n_splits=10, shuffle=True)
        # test = split.get_n_splits(self.dataset.ndata["x"], self.dataset.ndata["y"])
        test = list(split._iter_test_masks(self.dataset.ndata["x"], self.dataset.ndata["y"]))

        return test