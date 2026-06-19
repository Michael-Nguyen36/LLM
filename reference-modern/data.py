"""
Data loading for character-level TinyShakespeare dataset.

Downloads from the original source if not present.
This gives us a tiny, reproducible dataset for educational training.
"""

import os
import urllib.request

import torch


TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_tinyshakespeare(data_dir: str = "data"):
    """Download TinyShakespeare if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "tinyshakespeare.txt")
    if not os.path.exists(filepath):
        print(f"Downloading TinyShakespeare from {TINY_SHAKESPEARE_URL}...")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, filepath)
        print(f"Saved to {filepath}")
    return filepath


class CharTokenizer:
    """Simple character-level tokenizer.

    Maps each unique character to an integer. This is the simplest
    possible tokenizer — just for educational purposes.
    Real LLMs use BPE/SentencePiece with 32k-128k vocab.
    """

    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def load_tinyshakespeare(data_dir: str = "data") -> tuple[str, CharTokenizer]:
    """Load TinyShakespeare and return (text, tokenizer)."""
    filepath = download_tinyshakespeare(data_dir)
    with open(filepath, "r") as f:
        text = f.read()
    tokenizer = CharTokenizer(text)
    return text, tokenizer


def get_train_val_splits(text: str, tokenizer: CharTokenizer,
                         train_ratio: float = 0.9) -> tuple[torch.Tensor, torch.Tensor]:
    """Split encoded text into train and validation tensors."""
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(train_ratio * len(data))
    return data[:n], data[n:]


def get_batch(data: torch.Tensor, batch_size: int, block_size: int,
              device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (inputs, targets) from the data.

    Args:
        data: 1D tensor of token IDs
        batch_size: Number of sequences per batch
        block_size: Context length (max_seq_len)
        device: Target device

    Returns:
        x: (batch_size, block_size) — input tokens
        y: (batch_size, block_size) — target tokens (shifted by 1)
    """
    n = len(data)
    # Pick random starting positions
    starts = torch.randint(0, n - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)
