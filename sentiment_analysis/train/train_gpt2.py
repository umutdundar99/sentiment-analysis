import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from sentiment_analysis.data.loader import SentimentAnalysisDataModule
from sentiment_analysis.model.config import GPT2Config
from sentiment_analysis.model.module import GPTLightningModule
from sentiment_analysis.model.nn.gpt import GPT
from sentiment_analysis.utils.loss import FocalLoss


def train_gpt2(cfg: DictConfig, logger: WandbLogger):
    datamodule = SentimentAnalysisDataModule(
        data_dir=cfg.dataset.dir,
        batch_size=cfg.dataset.batch_size,
        max_length=cfg.dataset.max_length,
        encode_type=cfg.dataset.encode_type,
    )

    model_config = GPT2Config()
    model = GPT.from_pretrained(
        model_type="gpt2",
        GPTConfig=GPT2Config,
        num_classes=cfg.dataset.num_classes,
        override_args={"dropout": cfg.model.dropout},
    )
    criterion = FocalLoss()
    module = GPTLightningModule(
        model=model,
        config=model_config,
        criterion=criterion,
        weight_decay=cfg.optimizer.weight_decay,
        learning_rate=cfg.optimizer.lr,
        betas=cfg.optimizer.betas,
        device_type=cfg.trainer.device_type,
        num_classes=cfg.dataset.num_classes,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints/",
            filename="{epoch:02d}-gpt2",
            monitor="val/accuracy",
            mode="min",
            save_top_k=3,
            save_last=True,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]

    trainer = L.Trainer(
        logger=logger,
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        accelerator=cfg.trainer.accelerator,
        callbacks=callbacks,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        log_every_n_steps=1,
        gradient_clip_val=1.0,
        deterministic=True,
    )
    trainer.fit(module, datamodule)
    trainer.test(module, datamodule, ckpt_path="last")
