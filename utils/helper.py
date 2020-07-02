import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score


def loss_weights(dataset, agg_mask, device):
    """These weights are designed to compensate for class imabalance in the dataset (negated effects if class has
    undergone oversampling or undersampling)"""
    imb_wc = torch.bincount(dataset.y[agg_mask], minlength=int(dataset.y.max())).float().clamp(
        min=1e-10, max=1e10) / dataset.y[agg_mask].shape[0]
    weights = (1 / imb_wc) / (sum(1 / imb_wc))

    return weights.to(device)


def auroc_score(dataset, agg_mask, split_mask, logits, s_logits):
    if np.unique(dataset.y.numpy()) == 2:
        auroc = roc_auc_score(y_true=dataset.y[agg_mask].to('cpu').numpy(),
                              y_score=np.amax(s_logits[agg_mask].to('cpu').data.numpy(), axis=1),
                              average=None,
                              multi_class=None)
    else:
        output_mask = np.isin(list(range(0, dataset.y[split_mask].max())),
                              np.unique(dataset.y[split_mask].to('cpu').numpy()))
        test = np.apply_along_axis(func1d=lambda arr: arr[output_mask], axis=1,
                                   arr=logits[:, 1:].to('cpu').data.numpy())
        s_logits = F.softmax(input=torch.from_numpy(test), dim=1)  # Recalc of s_logits from outer scope

        auroc = roc_auc_score(y_true=dataset.y[agg_mask].to('cpu').numpy(),
                              y_score=s_logits[agg_mask],
                              average='macro',
                              multi_class='ovo')

    return auroc
