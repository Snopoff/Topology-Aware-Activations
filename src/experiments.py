from .datasets import Dataset, Circles, Tori, Disks
from .models import LightningModel
from .utils import mkdir_p

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner

from typing import List
import numpy as np
import hydra
import wandb
import time


class Experiments:
    def __init__(self, config: dict):
        self.config = config
        self.dataset = hydra.utils.instantiate(config["data"])
        self.model = hydra.utils.instantiate(config["model"])
        wandb.require("core")

    def run(self):
        seed_everything(self.config["experiment"]["seed"] + 1, workers=True)
        mkdir_p(self.config["experiment"]["wandb_dir"])
        wandb_logger = WandbLogger(
            project=self.config["experiment"]["wandb_project"],
            name=f'{self.dataset.name}_model_{self.config["logging"]["model"]}_hidden_{self.config["model"]["num_of_hidden"]}_dim_{self.config["model"]["dim_of_hidden"]}_activation_{self.config["logging"]["activation"]}_experiment_{self.config["experiment"]["seed"]+1}',
            group=f'{self.dataset.name}_hidden_{self.config["model"]["num_of_hidden"]}_dim_{self.config["model"]["dim_of_hidden"]}_activation_{self.config["logging"]["activation"]}',
            offline=self.config["experiment"]["offline"],
            save_dir=self.config["experiment"]["wandb_dir"],
        )
        wandb_logger.experiment.config.update({key: value for key, value in self.config["logging"].items()})
        lightning_model = LightningModel(model=self.model, learning_rate=self.config["trainer"]["optimizer_lr"])
        trainer = Trainer(
            deterministic=True,
            logger=wandb_logger,
            accelerator=self.config["experiment"]["accelerator"],
            devices="auto",
            log_every_n_steps=1,
            max_epochs=self.config["trainer"]["epochs"],
            # precision="bf16-mixed",
            default_root_dir="./logs/lightning_logs",
        )

        train_loader, test_loader = self.dataset.train_test_split(
            test_ratio=self.config["trainer"]["test_ratio"],
            batch_size=self.config["trainer"]["batch_size"],
        )

        if self.config["trainer"]["auto_lr_find"]:
            tuner = Tuner(trainer)
            tuner.lr_find(
                lightning_model,
                train_dataloaders=train_loader,
                val_dataloaders=test_loader,
            )

        trainer.fit(
            model=lightning_model,
            train_dataloaders=train_loader,
            val_dataloaders=test_loader,
        )
        wandb_logger.experiment.finish()


class ActivationExperiments(Experiments):
    def __init__(self, config: dict):
        super().__init__(config)
        self.activation = hydra.utils.instantiate(config["activation"])
        self.model = hydra.utils.instantiate(config["model"], activation=self.activation)


class TopologyChangeExperiments:
    """
    We firstly train a model (via train_eval_loop),
    then we pick val data and run it through the model again,
    but now we track the topology changes
    """

    def __init__(
        self,
        model,
        datasets: List,
        model_config={"num_of_hidden": 1, "dim_of_hidden": 3},
        n_experiments=30,
        list_of_activations=[
            "split_tanh",
            "split_sign",
            "split_sincos",
            "relu",
        ],
        test_ratio=0.2,
        epochs=5000,
        title_name=None,
    ) -> None:
        self.model = model
        self.datasets = datasets
        self.model_config = model_config
        self.n_experiments = n_experiments
        self.list_of_activations = list_of_activations
        self.test_ratio = test_ratio
        self.epochs = epochs
        self.title_name = (
            title_name if title_name is not None else "_n_{}_d_{}".format(model_config["num_of_hidden"], model_config["dim_of_hidden"])
        )

    def run_experiments(self, verbose=False, plot_results=True, homology_of_label=-1):
        results = {}
        for dataset in self.datasets:
            results[dataset] = {}
            if verbose:
                print("Working with dataset {}".format(dataset))
            for activation in self.list_of_activations:
                print("Working with activation {}".format(activation))
                topo_changes = np.zeros((self.n_experiments, self.model_config["num_of_hidden"] + 2))
                for i in range(self.n_experiments):
                    model = self.model(
                        dim_of_in=dataset.dim,
                        num_of_hidden=self.model_config["num_of_hidden"],
                        dim_of_hidden=self.model_config["dim_of_hidden"],
                        activation=activation,
                    )
                    res = train_eval_loop(
                        model,
                        dataset,
                        epochs=self.epochs,
                        test_ratio=self.test_ratio,
                        return_topo_changes=True,
                    )
                    topo_changes[i] = [d[homology_of_label] for d in res]
                mean_topo_change = np.mean(topo_changes, axis=0)
                std_topo_change = np.mean(topo_changes, axis=0)
                results[dataset][activation] = (mean_topo_change, std_topo_change)
                print("Mean topology changes = {}".format(mean_topo_change))

        if plot_results:
            self.plot_results(results)

        return results


def main_activations():
    circles, tori, disks = Circles(), Tori(), Disks()
    datasets = [circles, tori, disks]
    experiment = ActivationExperiments(
        model=ClassifierAL,
        datasets=datasets,
        n_experiments=2,
        num_of_hidden_layers=range(1, 2),
        dim_of_hidden_layers=(3, 4),
        list_of_activations=["split_tanh", "split_sincos", "relu"],
        verbose=False,
    )
    start_time = time.time()
    experiment.run_experiments()
    end_time = time.time()
    print("Time spent = {:.2f} min".format((end_time - start_time) / 60))


def main_topo_changes():
    datasets = [Circles()]
    experiment = TopologyChangeExperiments(
        model=ClassifierAL,
        datasets=datasets,
        n_experiments=2,
        list_of_activations=["relu", "split_tanh", "split_sincos"],
        model_config={"num_of_hidden": 3, "dim_of_hidden": 5},
    )
    start_time = time.time()
    experiment.run_experiments()
    end_time = time.time()
    print("Time spent = {:.2f} min".format((end_time - start_time) / 60))


if __name__ == "__main__":
    main_topo_changes()
