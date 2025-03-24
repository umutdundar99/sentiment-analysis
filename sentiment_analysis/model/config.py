from dataclasses import dataclass, asdict

@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50304 
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    

@dataclass
class nanoGPTConfig:
    block_size: int = 256
    vocab_size: int = 5000
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = False 
    