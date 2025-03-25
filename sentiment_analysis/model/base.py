from typing import Any, Dict

import lightning as L
import torch
import torchmetrics
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


class BaseModule(L.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()
        self.train_metrics = torchmetrics.MetricCollection(
            {
                "Accuracy": MulticlassAccuracy(num_classes=3),
                "Precision": MulticlassPrecision(num_classes=3),
                "Recall": MulticlassRecall(num_classes=3),
                "F1": MulticlassF1Score(num_classes=3),
            }
        )
        self.val_metrics = torchmetrics.MetricCollection(
            {
                "Accuracy": MulticlassAccuracy(num_classes=3),
                "Precision": MulticlassPrecision(num_classes=3),
                "Recall": MulticlassRecall(num_classes=3),
                "F1": MulticlassF1Score(num_classes=3),
            }
        )

    def calculate_metric(
        self, logits: torch.Tensor, y: torch.Tensor, metric: Dict[str, Any]
    ):
        """
        Calculates the accuracy of the model.

        Args:
            logits (torch.Tensor): The model predictions.
            y (torch.Tensor): The true labels.

        Returns:
            torch.Tensor: The accuracy of the model.
        """

        for name, _metric in metric.items():
            _metric = _metric.to(logits.device)
            _metric.update(logits, y)
            self.log(name, _metric, on_epoch=True, on_step=False)

    def reset_metrics(self, metric: Dict[str, Any]):
        for _metric in metric.values():
            _metric.reset()

    def forward(self, x):
        logits, _ = self.model(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x, y)
        loss = self.criterion(logits, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.calculate_metric(logits, y, self.train_metrics)
        self.log(
            "train/accuracy",
            self.train_metrics["Accuracy"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x, y)
        loss = self.criterion(logits, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log(
            "val/accuracy",
            self.val_metrics["Accuracy"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.calculate_metric(logits, y, self.val_metrics)
        return loss

    def on_train_epoch_end(self):
        self.reset_metrics(self.train_metrics)

    def on_validation_epoch_end(self):
        self.reset_metrics(self.val_metrics)
