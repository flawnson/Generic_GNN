"""This file contained the code for the tuning pipeline. Ray tune library is required to run this pipeline,
    and is only available on Linux and MacOS devices (beta wheels available for Windows)"""

import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from nn.DGL_models import GNNModel
import tensorflow as tf  # Needed to prevent get_global_worker attribute error

from sklearn.metrics import f1_score
from utils.helper import auroc_score
from utils.logger import log
from utils.stopper import Stop
from read.preprocessing import GenericDataset
from nn.optim import OptimizerObj, LRScheduler

try:
    import ray
    from ray import tune
except ModuleNotFoundError:
    log.info("Ray is not available, continuing run without benchmarking")


def tune_model(config: dict) -> None:
    """
    Tune function to be passed into ray.tune run function
    :param config: Tuning config dict
    """
    # tune.track.init()

    dataset = config.get('dataset')

    if config.get("model_config")["model"] == "GAT":
        model: GenericGNNModel = GNNModel(config.get("model_config"),
                                          dataset,
                                          config.get('device'),
                                          pooling=None).to(config.get('device'))
    else:
        raise NotImplementedError(f"{config.get('model_config')['model']} is not a model")  # Add to logger when implemented

    optimizer = OptimizerObj(config, model.parameters())

    early_stopper = Stop(config)  # Early stopping for loss and accuracy

    for epoch in range(config.get("epochs")):
        model.train()
        optimizer.zero_grad()
        logits = model(dataset, dataset.ndata["x"])
        agg_mask = np.logical_and(list(self.dataset.splits.values())[0], self.dataset.known_mask)
        imb_Wc = torch.bincount(dataset.ndata["y"][agg_mask]).float().clamp(min=1e-10, max=1e10) / dataset.ndata["y"][agg_mask].shape[0]
        weights = (1 / imb_Wc) / (sum(1 / imb_Wc))

        loss = F.cross_entropy(logits[agg_mask], dataset.ndata["y"][agg_mask], weight=weights)
        _loss = loss.clone().detach().to("cpu").item()

        loss.backward()
        optimizer.step()

        model.eval()
        logits = model(dataset, dataset.ndata["x"])
        score_dict = {score_type: {} for score_type, params in self.train_config.get("scores", DEFAULT_SCORES.items()).items()}

        for split_name, mask in self.dataset.splits.items():
            scores = Scores(self.train_config.get("scores", DEFAULT_SCORES),
                            self.dataset,
                            logits,
                            mask,
                            self.dataset.known_mask).score()

            # Slightly incomprehensible; renames key and assigns it to score object (either float or iterable)
            for score_name, score in scores.items():
                score_dict[score_name][split_name] = score

        early_stopper.early_stopping(train_acc, "accs", True)
        early_stopper.early_stopping(train_acc, "loss", False)


class Tuner(object):
    def __init__(self, config: dict, dataset: GenericDataset, device: torch.device) -> dict:
        """
        :param dataset: PyG data object
        :param masks: Holdout validation split masks
        :param config: Tuning configuration dict
        :param device: cuda or cpu
        """

        self.tuning_config = config["tune_config"]
        self.tuning_config["device"] = device
        self.tuning_config["dataset"] = dataset

        self.tuning_config["lr"] = tune.sample_from(lambda _: tune.loguniform(0.00000001, 0.001))
        self.tuning_config["wd"] = tune.sample_from(lambda _: tune.loguniform(0.0000001, 0.0001))
        self.tuning_config["dropout"] = tune.sample_from(lambda _: tune.loguniform(0.01, 0.70))

    def run_tune(self) -> dict:
        # tune_log = logger.set_tune_logger("tune_logging", osp.join(osp.dirname(__file__), "tune_info_log.txt"))
        import multiprocessing

        cpus = int(multiprocessing.cpu_count())
        gpus = 1 if torch.cuda.device_count() >= 1 else 0

        ray.init()
        analysis = tune.run(
            tune_model,
            config=self.tuning_config,
            num_samples=1,
            local_dir=osp.join(osp.dirname(osp.dirname(__file__)),
                               "logs",
                               self.tuning_config.get("model_config")["model"] + "_tuning"),
            resources_per_trial={"cpu": cpus, "gpu": gpus},
            loggers=tune.logger.DEFAULT_LOGGERS,
        )

        # tune_log("Best config: {}".format(analysis.get_best_config(metric="train_accuracy")))
        # tune_log("Best config: {}".format(analysis.get_best_config(metric="test_accuracy")))
        # tune_log("Best config: {}".format(analysis.get_best_config(metric="val_accuracy")))
        # tune_log("Best config: {}".format(analysis.get_best_config(metric="loss", mode="min")))

        df = analysis.dataframe()
        df.to_csv(path_or_buf=osp.join(osp.dirname(osp.dirname(__file__)),
                                       "logs",
                                       "x_classification" + "_experiment.csv"))

        return analysis.get_best_config(metric="train_f1_score")  # needs to return config for best model