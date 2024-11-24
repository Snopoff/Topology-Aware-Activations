CONFIG_NAME ?= default_config

run-experiments:
	python -m run_experiments --config-path configs --config-name $(CONFIG_NAME)

ENTITY_NAME ?= default
PROJECT_NAME ?= default
PARAM_NAME ?= default
VALUE ?= default

update_wandb_parameter:
	python -m scripts.update_wandb_parameter \
	hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled \
	+entity_name=$(ENTITY_NAME) +project_name=$(PROJECT_NAME) +param_name=$(PARAM_NAME) +value=$(VALUE)

WANDB_API_KEY ?= YOUR_API_TOKEN

export_wandb_token:
	export WANDB_API_KEY=${WANDB_API_KEY}