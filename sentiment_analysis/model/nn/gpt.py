import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from sentiment_analysis.model.nn.blocks import Block, LayerNorm


class GPT(nn.Module):
    def __init__(
        self,
        config: Dict[str, Any],
        num_classes: int = None,
        vocab_size: Optional[int] = None,
        is_pretrained: Optional[bool] = False,
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
        if not is_pretrained:
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

    def forward(self, idx):
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

        pooled_x = x[:, -1, :]  # (b, n_embd)

        # Classification logits
        logits = self.classifier_head(pooled_x)  # (b, num_classes)

        return logits

    @classmethod
    def from_pretrained(
        cls, model_type: str, GPTConfig, num_classes: int, override_args=None
    ):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config, num_classes=num_classes, is_pretrained=True)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        sd_keys = [k for k in sd_keys if not k.startswith("classifier_head")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.startswith("lm_head")]
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
