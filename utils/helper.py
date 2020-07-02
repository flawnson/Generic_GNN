import numpy as np
import torch


def loss_weights(dataset, agg_mask, device):
    """These weights are designed to compensate for class imabalance in the dataset (negated effects if class has
    undergone oversampling or undersampling)"""
    imb_wc = torch.bincount(dataset.y[agg_mask], minlength=int(dataset.y.max())).float().clamp(
        min=1e-10, max=1e10) / dataset.y[agg_mask].shape[0]
    weights = (1 / imb_wc) / (sum(1 / imb_wc))

    return weights.to(device)
