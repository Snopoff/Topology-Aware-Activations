import hydra
import hydra.utils as hu
from omegaconf import OmegaConf

from src.experiments import ActivationExperiments, TopologyChangeExperiments

OmegaConf.register_new_resolver("get_object", resolver=lambda obj: hu.get_object(obj))


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="default",
)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    activation = hydra.utils.instantiate(cfg["activation"])
    model = hydra.utils.instantiate(cfg["model"], activation=activation)
    print(model)
    print(cfg["logging"]["dim_hidden"])
    # experiments = Experiments(cfg)
    # experiments.run_experiments()


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="default_activation",
)
def run_activation_experiments(cfg):
    experiments = ActivationExperiments(cfg)
    experiments.run()


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="default_topological",
)
def run_topological_experiments(cfg):
    experiments = TopologyChangeExperiments(cfg)
    experiments.run()


if __name__ == "__main__":
    run_activation_experiments()
