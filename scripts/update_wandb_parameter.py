import wandb
import hydra
from omegaconf import DictConfig


@hydra.main(config_name=None)
def update_parameter(cfg: DictConfig) -> None:
    print(cfg.value)

    api = wandb.Api()
    runs = api.runs(f"{cfg['entity_name']}/{cfg['project_name']}")

    for run in runs:
        run.config[cfg["param_name"]] = cfg["value"]
        run.update()

    print("Parameters updated successfully for all runs.")


if __name__ == "__main__":
    update_parameter()
