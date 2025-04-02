from typing import Dict, Tuple

import torch
import torch.nn as nn

from sentiment_analysis.model.base import BaseModule


class GPTLightningModule(BaseModule):
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, any],
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str,
        criterion,
        num_classes: int,
    ):
        super().__init__(num_classes=num_classes)

        self.config = config
        self.criterion = criterion
        self.model = model
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.betas = betas
        self.device_type = device_type

    def configure_optimizers(self):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.learning_rate, betas=self.betas
        )

        return optimizer
