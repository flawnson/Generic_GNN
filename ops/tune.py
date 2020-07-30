import torch
import numpy as np
import os.path as osp
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import f1_score
from utils.helper import auroc_score
from nn.DGL_models import GNNModel

try:
    from ray import tune
except ModuleNotFoundError:
    print("Ray is not available, continuing run without tuning")


class Tuner:
    def __init__(self, tuning_config, dataset, model, device):
        self.tuning_config = tuning_config
        self.dataset = dataset
        self.model = model
        self.device = device

    def executable(self) -> None:
        tune.track.init()

        if self.tuning_config['model'] == "GAT":
            model = GNNModel(self.tuning_config,
                             self.dataset,
                             self.device).to(self.device)
        else:
            print(f"{self.tuning_config['model']} is not a model")

        optimizer = optim.Adam(model.parameters(), lr=self.tuning_config["lr"], weight_decay=self.tuning_config["wd"])

        loss_state, accs_state = 0, 0
        loss_no_improve, accs_no_improve = 0, 0

        for epoch in range(self.tuning_config.get("epochs")):
            model.train()
            optimizer.zero_grad()
            log_data, logits = model()
            alpha = np.logical_and(self.dataset.splits[0], self.dataset.known_mask)
            imb_Wc = torch.bincount(self.dataset.ndata["y"][alpha]).float().clamp(min=1e-10, max=1e10) / \
                     self.dataset.ndata["y"][alpha].shape[0]
            weights = (1 / imb_Wc) / (sum(1 / imb_Wc))

            loss = F.cross_entropy(logits[alpha],
                                   self.dataset.ndata["y"][alpha],
                                   weight=weights)
            _loss = loss.clone().detach().to("cpu").item()
            tune.track.log(loss=_loss)

            loss.backward()
            optimizer.step()

            model.eval()
            logits = model()
            accs, auroc_scores, f1_scores = [], [], []
            s_logits = F.softmax(input=logits[:, 1:], dim=1)

            for mask in self.dataset.splits:
                agg_mask = np.logical_and(mask, self.dataset.known_mask)
                pred = logits[alpha].max(1)[1]

                accs.append(pred.eq(self.dataset.ndata["y"].y[alpha]).sum().item() / alpha.sum().item())
                f1_scores.append(f1_score(y_true=self.dataset.ndata["y"][alpha].to('cpu'),
                                          y_pred=pred.to('cpu'),
                                          average='macro'))
                auroc_scores.append(auroc_score(self.dataset, agg_mask, mask, logits, s_logits))

    def run_tune(self):
        # tune_log = logger.set_tune_logger("tune_logging", osp.join(osp.dirname(__file__), "tune_info_log.txt"))
        import multiprocessing

        cpus = int(multiprocessing.cpu_count())
        gpus = 1 if torch.cuda.device_count() >= 1 else 0

        analysis = tune.run_train(
            self.executable,
            config=self.tuning_config,
            num_samples=1,
            local_dir=osp.join(osp.dirname(osp.dirname(__file__)),
                               "logs",
                               self.tuning_config["model"] + "_tuning_" + self.tuning_config.get("task") ),
            resources_per_trial={"cpu": cpus, "gpu": gpus}
        )

        # tune_log("Best config: {}".format(analysis.get_best_config(metric="train_accuracy")))
        # tune_log("Best config: {}".format(analysis.get_best_config(metric="test_accuracy")))
        # tune_log("Best config: {}".format(analysis.get_best_config(metric="val_accuracy")))
        # tune_log("Best config: {}".format(analysis.get_best_config(metric="loss", mode="min")))

        df = analysis.dataframe()
        df.to_csv(path_or_buf=osp.join(osp.dirname(osp.dirname(__file__)),
                                       "logs",
                                       self.tuning_config.get("task") + "_experiment.csv"))

        return analysis.get_best_config(metric="train_f1_score")  # needs to return config for best model