"""This file contains the code used to partition the graph dataset into sets for each run type."""

import numpy as np
import torch

from typing import Dict
from dgl.data.utils import Subset
from itertools import accumulate
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils.validation import column_or_1d
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import type_of_target
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

    def _make_test_folds(self, X, y=None):
        rng = check_random_state(self.random_state)
        y = np.asarray(y)
        type_of_target_y = type_of_target(y)
        allowed_target_types = ('binary', 'multiclass')
        if type_of_target_y not in allowed_target_types:
            raise ValueError(
                'Supported target types are: {}. Got {!r} instead.'.format(
                    allowed_target_types, type_of_target_y))

        y = column_or_1d(y)

        _, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
        # y_inv encodes y according to lexicographic order. We invert y_idx to
        # map the classes so that they are encoded by order of appearance:
        # 0 represents the first label appearing in y, 1 the second, etc.
        _, class_perm = np.unique(y_idx, return_inverse=True)
        y_encoded = class_perm[y_inv]

        n_classes = len(y_idx)
        y_counts = np.bincount(y_encoded)
        min_groups = np.min(y_counts)
        if np.all(self.n_splits > y_counts):
            raise ValueError("n_splits=%d cannot be greater than the"
                             " number of members in each class."
                             % (self.n_splits))
        if self.n_splits > min_groups:
            warnings.warn(("The least populated class in y has only %d"
                           " members, which is less than n_splits=%d."
                           % (min_groups, self.n_splits)), UserWarning)

        # Determine the optimal number of samples from each class in each fold,
        # using round robin over the sorted y. (This can be done direct from
        # counts, but that code is unreadable.)
        y_order = np.sort(y_encoded)
        allocation = np.asarray(
            [np.bincount(y_order[i::self.n_splits], minlength=n_classes)
             for i in range(self.n_splits)])

        # To maintain the data order dependencies as best as possible within
        # the stratification constraint, we assign samples from each class in
        # blocks (and then mess that up when shuffle=True).
        test_folds = np.empty(len(y), dtype='i')
        for k in range(n_classes):
            # since the kth column of allocation stores the number of samples
            # of class k in each test set, this generates blocks of fold
            # indices corresponding to the allocation for class k.
            folds_for_class = np.arange(self.n_splits).repeat(allocation[:, k])
            if self.shuffle:
                rng.shuffle(folds_for_class)
            test_folds[y_encoded == k] = folds_for_class
        return test_folds

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
