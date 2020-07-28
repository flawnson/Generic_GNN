import numpy as np
import torch

from dgl.data.utils import Subset
from itertools import accumulate


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
            return [Subset(self.dataset, indices[offset - length:offset]) for offset, length in
                    zip(accumulate(lengths), lengths)]

    def temp_split(self):
        """
        Temporary method to split dataset until rest of codebase is complete
        :return: train, test, val for DGL object
        """
        frac_list = np.asarray(self.data_config.get("split_sizes", [0.8, 0.1, 0.1]))

        # Initialize empty boolean arrays
        train_bool = np.zeros(len(self.dataset.ndata["y"]), dtype=bool)
        test_bool = np.zeros(len(self.dataset.ndata["y"]), dtype=bool)
        val_bool = np.zeros(len(self.dataset.ndata["y"]), dtype=bool)

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
        train_bool[train_idx] = True
        test_bool[test_idx] = True
        val_bool[val_idx] = True

        return [train_bool, test_bool, val_bool]

    def balanced_split(self):
        """TODO: Check if split is performing properly"""
        """TODO: Refactor to clean method"""
        frac_list = np.asarray(self.data_config.get("split_sizes", [0.8, 0.1, 0.1]))

        # Initialize empty boolean arrays
        booleans = []
        for frac in frac_list:
            boolean = np.zeros(len(self.dataset.ndata["y"]), dtype=bool)
            booleans.append(boolean)

        class_indices = []
        for class_label in range(len(np.unique(self.dataset.ndata["y"])) + 1):
            mask = self.dataset.ndata["y"].numpy() == class_label
            class_indices.append(np.where(mask)[0])

        split_indices = []
        mark = 0
        for frac in frac_list:
            indices = []
            leftover = class_indices[mark]

            for index_list in class_indices:
                split_len = int(round(frac * len(leftover)))

                split_idx = np.random.choice(leftover, split_len, replace=False)
                leftover = np.setdiff1d(leftover, split_idx)
                indices += split_idx.tolist()

            split_indices.append(indices)
            mark += 1

        # Change all False to True at indices
        for boolean, indices in zip(booleans, split_indices):
            boolean[indices] = True

        return booleans

    def indices_to_mask(self, index_lists: list) -> list:
        masks = []

        for index_list in index_lists:
            mask = np.zeros(len(self.dataset), dtype=int)
            mask[index_list.indices] = 1
            masks.append(mask)

        return masks
