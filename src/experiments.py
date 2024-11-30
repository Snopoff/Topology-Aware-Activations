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

        #! === Computing TC ===
        topo_info_per_layer = lightning_model.compute_topological_complexity(
            self.dataset.tensor_dataset.tensors[0], labels=self.dataset.tensor_dataset.tensors[1], subsample_size=None
        )

        """tc_per_layer = [layer[-1] for layer in topo_info_per_layer]
        for tc in tc_per_layer:
            wandb_logger.experiment.log({"TC": tc})"""

        tc_per_layer_per_label = [[layer[0], layer[1]] for layer in topo_info_per_layer]

        for tc in tc_per_layer_per_label:
            wandb_logger.experiment.log({"TC_Label_0": tc[0], "TC_Label_1": tc[1]})

        wandb_logger.experiment.finish()


class ActivationExperiments(Experiments):
    def __init__(self, config: dict):
        super().__init__(config)
        self.activation = hydra.utils.instantiate(config["activation"])
        self.model = hydra.utils.instantiate(config["model"], activation=self.activation)


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
