import hydra
from hydra.utils import instantiate

from src.datasets import SplittedDataset, Dataset
from src.utils import mkdir_p
from src.models import *  # noqa: F403

import numpy as np
import wandb

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner


class ValidationalDatasets:
    def __init__(self, config: dict):
        self.i_split = config["i_split"]
        self.config = config
        self.config["data"]["_target_"] = "src.datasets.NestedRingsOld"
        self.config["data"]["base_n_points"] = 15000

    def create_dataset(self):
        return instantiate(self.config["data"])

    def prepare_datasets(self):
        dataset = self.create_dataset()
        splitted_datasets = SplittedDataset.split_dataset(
            dataset,
            n_splits=self.i_split,
            **self.config["splitting"]["splitting_kwargs"],
        ).splitted_datasets
        pick_the_middle_one = len(splitted_datasets) // 2
        picked_dataset = splitted_datasets[pick_the_middle_one]
        result = [picked_dataset] + SplittedDataset.split_dataset(
            picked_dataset,
            n_splits=1,
            **self.config["splitting"]["splitting_kwargs"],
        ).splitted_datasets
        return result

    def save_datasets(self):
        splitted_datasets = self.prepare_datasets()
        for i, dataset in enumerate(splitted_datasets):
            mkdir_p("data")
            np.save(f"data/{dataset.name}_X.npy", dataset.X)
            np.save(f"data/{dataset.name}_y.npy", dataset.y)


class ValidationalExperiments:
    def __init__(
        self,
        config: dict,
    ):
        self.config = config
        self.model_class = globals()[config["model"]["model_name"]]
        self.dataset = self._retrieve_dataset(config["dataname"])
        wandb.require("core")

    def _retrieve_dataset(self, dataname):
        return Dataset(X=np.load(f"data/{dataname}_X.npy"), y=np.load(f"data/{dataname}_y.npy"), name=dataname)

    def run_experiment(self, num_hidden: int, dim_hidden: int):
        """
        Run an experiment with the given dataset, number of hidden layers, and dimension of hidden layers.

        Parameters:
            dataset (Dataset): The dataset to use for the experiment.
            num_hidden (int): The number of hidden layers in the model.
            dim_hidden (int): The dimension of the hidden layers in the model.

        Returns:
            None
        """
        random_states = range(1, self.config["experiment"]["n_experiments"] + 1)
        for random_state in random_states:
            seed_everything(random_state, workers=True)
            mkdir_p(self.config["experiment"]["wandb_dir"])
            wandb_logger = WandbLogger(
                project=self.config["experiment"]["wandb_project"],
                name=f"{self.dataset.name}_hidden_{num_hidden}_dim_{dim_hidden}_experiment_{random_state}",
                group=f"{self.dataset.name}_hidden_{num_hidden}_dim_{dim_hidden}",
                offline=self.config["experiment"]["offline"],
                save_dir=self.config["experiment"]["wandb_dir"],
            )
            wandb_logger.experiment.config.update(
                {"num_hidden": num_hidden, "dim_hidden": dim_hidden, "base_n_points": self.config["logging"]["base_n_points"]}
            )
            if self.config["experiment"]["save_split_results"]:
                fig = self.dataset.plot_data()
                path_to_file = self.config["experiment"]["wandb_dir"] + f"{self.dataset.name}.png"
                fig.write_image(path_to_file)
                artifact = wandb.Artifact(f"{self.dataset.name}", type="plots")
                artifact.add_file(path_to_file)
                wandb.log_artifact(artifact)
            model = ClassifierModel(  # noqa: F405
                input_dim=self.config["model"]["input_dim"],
                hidden_dim=dim_hidden,
                num_hidden=num_hidden,
            )
            lightning_model = LightningModel(  # noqa: F405
                model=model, learning_rate=self.config["model"]["optimizer_lr"]
            )
            trainer = Trainer(
                deterministic=True,
                logger=wandb_logger,
                accelerator=self.config["experiment"]["accelerator"],
                devices="auto",
                log_every_n_steps=1,
                max_epochs=self.config["model"]["epochs"],
                precision="bf16-mixed",
                default_root_dir="./logs/lightning_logs",
            )

            train_loader, test_loader = self.dataset.train_test_split(
                test_ratio=self.config["model"]["test_ratio"],
                batch_size=self.config["model"]["batch_size"],
            )

            if self.config["model"]["auto_lr_find"]:
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

    def run_experiments(self):
        """
        Run experiments on the given dataset with different numbers of hidden layers and dimensions.

        This function iterates over the possible numbers of hidden layers and dimensions specified in the config,
        and for each combination, it runs an experiment on the given dataset and its splits.

        Parameters:
            None

        Returns:
            None
        """
        print("Datasets are splitted. Starting the experiments ...")
        for num_hidden in self.config["model"]["num_hidden"]:
            for dim_hidden in self.config["model"]["dim_hidden"]:
                self.run_experiment(num_hidden, dim_hidden)


def prepare_dataset(cfg):
    val_datasets = ValidationalDatasets(cfg)
    val_datasets.save_datasets()


def validate(cfg):
    val_experiments = ValidationalExperiments(cfg)
    val_experiments.run_experiments()


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="default_config",
)
def main(cfg):
    if cfg["task"] == "prepare_dataset":
        prepare_dataset(cfg)
    if cfg["task"] == "validate":
        validate(cfg)


if __name__ == "__main__":
    main()
