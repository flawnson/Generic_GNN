""" This module contains miscellaneous code that takes up too much space in the main pipeline, hence have been abstracted and
    moved here for organizational reasons """
import numpy as np
import json
import torch
import argparse
import jsonschema
import subprocess
import os.path as osp
import torch.nn.functional as F

from typing import Tuple, Dict
from utils.logger import set_file_logger, log
from nn.DGL_models import GenericGNNModel
from sklearn.metrics import roc_auc_score
from collections.abc import Iterable
from torch.utils.tensorboard import SummaryWriter

import warnings
import sklearn
# from sklearn.model_selection._split import _BaseKFold
import sklearn.model_selection
from sklearn.utils.validation import column_or_1d
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import type_of_target


def parse_arguments() -> Tuple[Dict, torch.device]:
    path = osp.join('data', 'biogrid')
    parser = argparse.ArgumentParser(description="Config file parser")
    parser.add_argument("-c", "--config", help="json config file", type=str)
    parser.add_argument("-s", "--scheme", help="json scheme file", type=str)
    args = parser.parse_args()

    json_data: dict = json.load(open(args.config))
    json_scheme: dict = json.load(open(args.scheme))
    set_file_logger(json_data)
    # DGL Overrides torch.tensor.to() and implements it's own to() method for its graph objects
    device = torch.device("cuda" if json_data["cuda"] and torch.cuda.is_available() else "cpu")
    log.info(f"Using {device} for compute")

    # See https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    # benchmark mode is good whenever your input sizes for your network do not vary
    torch.backends.cudnn.benchmark = True if not json_data.get("cuda", False) and torch.cuda.is_available() else False

    git_hash = subprocess.check_output(["git", "describe", "--always"]).strip().decode()
    git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode()
    log.info(f"Git hash: {git_hash}, branch: {git_branch}")

    if json_data["validate"]:
        try:
            jsonschema.validate(json_data, json_scheme)
        except jsonschema.ValidationError:
            log.warning("Tried to validate but failed, continuing run anyway")

    return json_data, device


def loss_weights(dataset, agg_mask: np.ndarray, device: torch.device) -> torch.tensor:
    """ These weights are designed to compensate for class imabalance in the dataset (negated effects if class has
        undergone oversampling or undersampling) """
    imb_wc = torch.bincount(dataset.ndata["y"][agg_mask], minlength=int(dataset.ndata["y"].max())).float().clamp(
        min=1e-10, max=1e10) / dataset.ndata["y"][agg_mask].shape[0]
    weights = (1 / imb_wc) / (sum(1 / imb_wc))

    return weights.to(device)


def auroc_score(params: list, dataset, agg_mask: np.ndarray, split_mask, logits: torch.tensor, s_logits) -> float:
    """ Logic for calculating roc_auc_score using sklearn (different configurations needed depending on the labelset
        and task type """
    if len(np.unique(dataset.ndata["y"].numpy())) == 2:
        auroc = roc_auc_score(y_true=dataset.y[agg_mask].to('cpu').numpy(),
                              y_score=np.amax(s_logits[agg_mask].to('cpu').data.numpy(), axis=1),
                              average=params[0],  # Should be None in binary case
                              multi_class=params[1])  # Should be None in binary case
    else:
        output_mask = np.isin(list(range(dataset.ndata["y"][split_mask].max())),
                              np.unique(dataset.ndata["y"][split_mask].to('cpu').numpy()))
        test = np.apply_along_axis(func1d=lambda arr: arr[output_mask], axis=1,
                                   arr=logits[:, 1:].to('cpu').data.numpy())
        s_logits = F.softmax(input=torch.from_numpy(test), dim=1)  # Recalc of s_logits from outer scope

        auroc = roc_auc_score(y_true=dataset.ndata["y"][agg_mask].to('cpu').numpy(),
                              y_score=s_logits[agg_mask],
                              average=params[0],
                              multi_class=params[1])

    return auroc


def pretty_print(scores: dict) -> None:
    # Function to use for printing model scores in training pipeline
    for score_type, score_set in scores.items():
        try:
            for split_name, score in score_set.items():
                print(f"{score_type + '-' + split_name}: {round(score, 3)}")
            print("-" * 10)
        except TypeError:
            for split_name, score in score_set.items():
                print(f"{score_type + '-' + split_name}: {score}")
            print("-" * 10)


def save_model(config: dict, epoch: int, model: GenericGNNModel) -> None:
    if config.get("save_model", False) and epoch == config["epochs"]:
        torch.save(model.state_dict, osp.join(osp.dirname(__file__), "output", config["save_model"]))


def load_model(config: dict, model: GenericGNNModel, device: torch.device) -> GenericGNNModel:
    # When loading a model on a CPU that was trained with a GPU, pass torch.device('cpu')
    # to the map_location argument in the torch.load() function.
    # In this case, the storages underlying the tensors are dynamically remapped
    # to the CPU device using the map_location argument.
    if config.get("load_model", None):
        try:
            return model.load_state_dict(torch.load(osp.join("outputs", config["load_model"]), map_location=device))
        except:
            RuntimeError("Pretrained weights do not seem to fit the provided model; check if pretrained model exists")
    else:
        return model.to(device)


class StratifiedKFold(sklearn.model_selection._split._BaseKFold):
    """Stratified K-Folds cross-validator
    Provides train/test indices to split data in train/test sets.
    This cross-validation object is a variation of KFold that returns
    stratified folds. The folds are made by preserving the percentage of
    samples for each class.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.
    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.
    random_state : int or RandomState instance, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
        Otherwise, leave `random_state` as `None`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedKFold
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> skf = StratifiedKFold(n_splits=2)
    >>> skf.get_n_splits(X, y)
    2
    >>> print(skf)
    StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
    >>> for train_index, test_index in skf.split(X, y):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    TRAIN: [1 3] TEST: [0 2]
    TRAIN: [0 2] TEST: [1 3]
    Notes
    -----
    The implementation is designed to:
    * Generate test sets such that all contain the same distribution of
      classes, or as close as possible.
    * Be invariant to class label: relabelling ``y = ["Happy", "Sad"]`` to
      ``y = [1, 0]`` should not change the indices generated.
    * Preserve order dependencies in the dataset ordering, when
      ``shuffle=False``: all samples from class k in some test set were
      contiguous in y, or separated in y by samples from classes other than k.
    * Generate test sets where the smallest and largest differ by at most one
      sample.
    .. versionchanged:: 0.22
        The previous implementation did not follow the last constraint.
    See also
    --------
    RepeatedStratifiedKFold: Repeats Stratified K-Fold n times.
    """
    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)

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

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.
        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.
        groups : object
            Always ignored, exists for compatibility.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """
        y = check_array(y, ensure_2d=False, dtype=None)
        return super().split(X, y, groups)