from typing import Dict, Tuple
import torch.nn as nn
from sentiment_analysis.model.base import BaseModule
import torch
import inspect


class GPTLightningModule(BaseModule):
    def __init__(self, 
                 model: nn.Module,
                 config: Dict[str, any],
                 weight_decay: float,
                    learning_rate: float,
                    betas: Tuple[float, float],
                    device_type: str,
                 criterion):
        super().__init__()
       
        self.config= config
        self.criterion = criterion
        self.model = model
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.betas = betas
        self.device_type = device_type
        #self.save_hyperparameters()

    def configure_optimizers(self):
    
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and self.device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=self.betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer