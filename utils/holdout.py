import numpy as np
import torch

from dgl.data.utils import Subset
from itertools import accumulate


class Holdout:
    def __init__(self, data_config: dict, dataset, bool_mask=True):
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

    def balanced_split(self):
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
        for frac in frac_list:
            indices = []

            for index_list in class_indices:
                split_len = int(round(frac * len(index_list)))

                split_idx = np.random.choice(index_list, split_len, replace=False)
                leftover = np.setdiff1d(index_list, split_idx)
                indices += split_idx.tolist()

            split_indices.append(indices)

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
