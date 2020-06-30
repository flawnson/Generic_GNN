import numpy as np
import torch

from dgl.data.utils import Subset, split_dataset
from itertools import accumulate


class Holdout:
    def __init__(self, data_config, dataset):
        super(Holdout, self)
        self.data_config = data_config
        self.dataset = dataset

    def tri_split(self):
        splits = split_dataset(self.dataset, frac_list=self.data_config["split_sizes"],
                               shuffle=self.data_config["shuffle"])

        return splits

    def split(self):
        if self.data_config["split_sizes"] is None:
            frac_list = [0.8, 0.1, 0.1]
        else:
            frac_list = self.data_config["split_sizes"]

        frac_list = np.asarray(frac_list)
        assert np.allclose(np.sum(frac_list), 1.), \
            'Expect frac_list sum to 1, got {:.4f}'.format(np.sum(frac_list))

        num_data = len(self.dataset)
        lengths = (num_data * frac_list).astype(int)
        lengths[-1] = num_data - np.sum(lengths[:-1])

        if self.data_config["shuffle"]:
            indices = np.random.RandomState(
                seed=None).permutation(num_data)
        else:
            indices = np.arange(num_data)

        return [Subset(self.dataset, indices[offset - length:offset]) for offset, length in
                zip(accumulate(lengths), lengths)]
