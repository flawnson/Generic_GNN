import numpy as np
import torch

from dgl.data.utils import Subset
from itertools import accumulate


class Holdout:
    def __init__(self, data_config, dataset, bool_mask=True):
        super(Holdout, self)
        self.data_config = data_config
        self.dataset = dataset

    def split(self):
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

    def indices_to_mask(self, index_lists):
        masks = []

        for index_list in index_lists:
            mask = np.zeros(len(self.dataset), dtype=int)
            mask[index_list.indices] = 1
            masks.append(mask)

        return masks
