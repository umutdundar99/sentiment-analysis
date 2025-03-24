from typing import Dict
import torch.nn as nn
import lightning as L


class BaseModule(L.LightningModule):
    def __init__(self, ):
        super().__init__()
        #self.save_hyperparameters()
   

    def forward(self, x):
        logits, _ = self.model(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self.model(x, y)
        loss = self.criterion(logits, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self.model(x, y)
        loss = self.criterion(logits, y)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        return loss

