from typing import Any, Optional

import lightning as L
import torch
import torchmetrics
from torchmetrics.classification import (
    ConfusionMatrix,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

import wandb


class BaseModule(L.LightningModule):
    def __init__(self, num_classes: Optional[int] = None):
        super().__init__()

        self.num_classes = num_classes

        metrics = {
            "accuracy": MulticlassAccuracy(num_classes=self.num_classes),
            "precision": MulticlassPrecision(num_classes=self.num_classes),
            "recall": MulticlassRecall(num_classes=self.num_classes),
            "f1": MulticlassF1Score(num_classes=self.num_classes),
        }

        self.train_metrics = torchmetrics.MetricCollection(metrics, prefix="train/")
        self.val_metrics = torchmetrics.MetricCollection(metrics, prefix="val/")
        self.test_metrics = torchmetrics.MetricCollection(metrics, prefix="test/")

        self.train_confmat = ConfusionMatrix(
            task="multiclass", num_classes=self.num_classes
        )
        self.val_confmat = ConfusionMatrix(
            task="multiclass", num_classes=self.num_classes
        )

        self.test_confmat = ConfusionMatrix(
            task="multiclass", num_classes=self.num_classes
        )

        self.class_labels = ["negative", "neutral", "positive"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return logits

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        train_results = self.train_metrics(preds, y)
        confmat = self.train_confmat(preds, y)

        self.log_dict(train_results, prog_bar=True)

        per_class_acc = confmat.diag() / confmat.sum(dim=1)

        for i, acc in enumerate(per_class_acc):
            self.log(f"train/acc_{self.class_labels[i]}", acc, prog_bar=True)

        self.log("loss/train", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.val_metrics.update(preds, y)
        self.val_confmat.update(preds, y)

        self.log("loss/val", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        val_results = self.val_metrics.compute()
        self.log_dict(val_results, prog_bar=True)

        confmat = self.val_confmat.compute()
        per_class_acc = confmat.diag() / confmat.sum(dim=1)

        for i, acc in enumerate(per_class_acc):
            self.log(f"val/acc_{self.class_labels[i]}", acc, prog_bar=True)

        confmat_table = wandb.Table(
            columns=["Predicted"] + self.class_labels,
            data=[
                [self.class_labels[i]] + confmat[i].tolist()
                for i in range(self.num_classes)
            ],
        )
        self.logger.experiment.log({"val/confusion_matrix": confmat_table})

        self.val_metrics.reset()
        self.val_confmat.reset()

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self.model(x)

        preds = torch.argmax(logits, dim=1)
        self.test_metrics.update(preds, y)
        self.test_confmat.update(preds, y)

    def on_test_epoch_end(self):
        test_results = self.test_metrics.compute()
        self.log_dict(test_results, prog_bar=True)

        confmat = self.test_confmat.compute()
        per_class_acc = confmat.diag() / confmat.sum(dim=1)

        for i, acc in enumerate(per_class_acc):
            self.log(f"test/acc_{self.class_labels[i]}", acc, prog_bar=True)

        confmat_table = wandb.Table(
            columns=["Predicted"] + self.class_labels,
            data=[
                [self.class_labels[i]] + confmat[i].tolist()
                for i in range(self.num_classes)
            ],
        )
        self.logger.experiment.log({"test/confusion_matrix": confmat_table})

        self.test_metrics.reset()
        self.test_confmat.reset()
