import logging
from lightning.pytorch.loggers import WandbLogger
import lightning as L 
from omegaconf import DictConfig
from sentiment_analysis.data.loader import SentimentAnalysisDataModule


def train_gpt2(cfg: DictConfig,
                         logger: WandbLogger):
      
      datamodule = SentimentAnalysisDataModule(
          data_dir= cfg.dataset.dir,
          batch_size= cfg.dataset.batch_size,
      )
    

