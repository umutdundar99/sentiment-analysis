import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from sentiment_analysis.data.loader import SentimentAnalysisDataModule
from sentiment_analysis.model.config import nanoGPTConfig
from sentiment_analysis.model.module import GPTLightningModule
from sentiment_analysis.model.nn.gpt import GPT
from sentiment_analysis.utils.loss import CrossEntropyLoss


def train_nanogpt(cfg: DictConfig, logger: WandbLogger):
    datamodule = SentimentAnalysisDataModule(
        data_dir=cfg.dataset.dir,
        batch_size=cfg.dataset.batch_size,
        max_length=cfg.dataset.max_length,
        encode_type=cfg.dataset.encode_type,
    )

    model_config = nanoGPTConfig()
    model = GPT(
        config=model_config,
        num_classes=cfg.dataset.num_classes,
        vocab_size=datamodule.train.vocab_size
        if cfg.dataset.encode_type == "char"
        else None,
    )
    criterion = CrossEntropyLoss()
    module = GPTLightningModule(
        model=model,
        config=model_config,
        criterion=criterion,
        weight_decay=cfg.optimizer.weight_decay,
        learning_rate=cfg.optimizer.lr,
        betas=cfg.optimizer.betas,
        device_type=cfg.trainer.device_type,
    )

    callbacks = [
        # ModelCheckpoint(
        #     dirpath="checkpoints/",
        #     filename="{epoch:02d}-{val_loss:.2f}",
        #     monitor="val_loss",
        #     mode="min",
        #     save_top_k=3,
        #     save_last=True,
        #     verbose=True,
        # ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]
    trainer = L.Trainer(
        # logger = logger,
        max_steps=cfg.trainer.max_steps,
        precision=cfg.trainer.precision,
        accelerator=cfg.trainer.accelerator,
        callbacks=callbacks,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        log_every_n_steps=50,
    )
    trainer.fit(module, datamodule)
