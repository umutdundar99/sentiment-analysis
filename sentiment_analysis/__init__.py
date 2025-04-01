__version__ = "0.1.0"
import os

import typer
from hydra import compose, initialize
from lightning.pytorch.loggers import WandbLogger

from sentiment_analysis.train.train_gpt2 import train_gpt2
from sentiment_analysis.train.train_nanogpt import train_nanogpt

CFG_DIR = os.path.join("config")
cli = typer.Typer(pretty_exceptions_enable=False, rich_markup_mode="markdown")


@cli.command()
def trainer(
    task: str = typer.Argument(..., help="The task to perform, e.g. 'sentiment'"),
    model: str = typer.Argument(..., help="The model to use, e.g. 'gpt2' or 'nanogpt'"),
):
    if model == "gpt2":
        trainer = train_gpt2
        cfg_path = "gpt2.yaml"
    elif model == "nanogpt":
        trainer = train_nanogpt
        cfg_path = "nanogpt.yaml"

    else:
        raise ValueError("Probably not implemented yet :)")

    with initialize(config_path=CFG_DIR):
        cfg = compose(config_name=cfg_path)
        wandb_logger = WandbLogger(
            name=cfg.wandb.name,
            offline=cfg.wandb.offline,
            project=cfg.wandb.project,
            experiment=cfg.wandb.experiment,
        )
        trainer(cfg, wandb_logger)


if __name__ == "__main__":
    cli()
