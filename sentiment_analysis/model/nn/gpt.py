import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from sentiment_analysis.model.nn.blocks import Block, LayerNorm


class GPT(nn.Module):
    def __init__(
        self, config: Dict[str, Any], num_classes: int, vocab_size: Optional[int] = None
    ):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        config.vocab_size = vocab_size if vocab_size is not None else config.vocab_size
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        self.classifier_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.n_embd // 2, num_classes),
        )

        # Weight initialization
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Transformer forward
        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)  # (b, t, n_embd)

        # Pooling (literatüre uygun yöntem: son token kullanımı)
        pooled_x = x[:, -1, :]  # (b, n_embd)

        # Classification logits
        logits = self.classifier_head(pooled_x)  # (b, num_classes)

        return logits
