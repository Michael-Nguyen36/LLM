"""
Training and model configuration for the tiny decoder-only transformer.

We start VERY small (~10M params) so it trains fast on CPU/RX 6600.
Once the pipeline works, bump these values.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Decoder-only transformer hyperparameters.

    These defaults give a ~10M parameter model — enough to learn
    interesting patterns on TinyShakespeare, but trains in minutes on CPU.
    """
    vocab_size: int = 65              # TinyShakespeare has 65 unique chars
    d_model: int = 256                # Embedding / hidden dimension
    n_heads: int = 4                  # Number of attention heads
    n_layers: int = 6                 # Number of transformer blocks
    ffn_hidden_mult: int = 4          # FFN hidden = ~2/3 * d_model * mult (for SwiGLU)
    max_seq_len: int = 256            # Context window
    rope_base: float = 10000.0        # RoPE frequency base
    dropout: float = 0.0              # No dropout for tiny model


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 100
    max_iters: int = 5000
    eval_interval: int = 250
    eval_iters: int = 100
    grad_clip: float = 1.0
    log_interval: int = 10
    # Learning rate schedule
    lr_schedule: str = "cosine"       # cosine decay with warmup
    min_lr: float = 1e-5
    # Data
    data_path: str = "data"
    # Checkpoint
    ckpt_path: str = "checkpoints/model.pt"
    save_interval: int = 1000


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 1337
    device: str = "cpu"               # Auto-detected; override if needed
