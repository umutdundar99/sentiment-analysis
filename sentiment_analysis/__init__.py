__version__ = "0.1.0"
import os
import typer
from sentiment_analysis.train.train_nanogpt import train_nanogpt_sentiment
from sentiment_analysis.train.train_gpt2 import train_gpt2_sentiment
from lightning.pytorch.loggers import WandbLogger
from hydra import compose, initialize

CFG_DIR = os.path.join("config")
cli = typer.Typer(pretty_exceptions_enable=False, rich_markup_mode="markdown")

@cli.command()
def trainer(
    task: str = typer.Argument(..., help="The task to perform, e.g. 'sentiment'"),
    model: str = typer.Argument(..., help="The model to use, e.g. 'gpt2' or 'nanogpt'")
):
    
    if model == "gpt2":
        trainer = train_gpt2_sentiment
        cfg_path = "gpt2.yaml"
    elif model == "nanogpt":
        trainer = train_nanogpt_sentiment
        cfg_path = "nanogpt.yaml"
    
    else:
        raise ValueError(f"Probably not implemented yet :)")

    with initialize(config_path=CFG_DIR):
        cfg = compose(config_name=cfg_path)
        wandb_logger = WandbLogger(cfg.logger.project, cfg.logger.name)
        trainer(cfg)  

if __name__ == "__main__":
    cli()