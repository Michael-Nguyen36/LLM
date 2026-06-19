"""
01_training_loop.py — Train VanillaDecoderOnlyTransformer on TinyShakespeare.

Usage:
    python3 code/06-training-generation/01_training_loop.py

Downloads TinyShakespeare if not present, trains for 1000 steps,
and saves the best checkpoint to checkpoints/vanilla_best.pt.

Expected runtime: ~5-10 minutes on CPU for 1000 steps.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path (stdlib has a 'code' module that shadows our directory)
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# Import the shared model module
from code.vanilla_transformer import VanillaDecoderOnlyTransformer

if __name__ == "__main__":
    # ── Load TinyShakespeare ────────────────────────────────────────

    DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_path = Path("data/tinyshakespeare.txt")
    data_path.parent.mkdir(exist_ok=True)

    if not data_path.exists():
        import urllib.request
        print("Downloading TinyShakespeare...")
        max_retries = 3
        base_delay = 2  # seconds
        for attempt in range(1, max_retries + 1):
            try:
                with urllib.request.urlopen(DATA_URL, timeout=30) as response:
                    data_path.write_bytes(response.read())
                print("Download complete.")
                break
            except Exception as e:
                if attempt == max_retries:
                    print(f"Download failed after {max_retries} attempts: {e}")
                    print("Create data/tinyshakespeare.txt manually with any text to train on.")
                    exit(1)
                delay = base_delay * (2 ** (attempt - 1))  # 2s, 4s, 8s
                print(f"  Attempt {attempt} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)

    text = data_path.read_text()
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

    print(f"Loaded {len(data):,} characters, vocab size {vocab_size}")

    # ── Train/val split ─────────────────────────────────────────────

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(data, batch_size=16, block_size=128):
        """Sample a random batch of (input, target) sequences."""
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        return x, y

    # ── Model and optimizer ─────────────────────────────────────────

    model = VanillaDecoderOnlyTransformer(vocab_size=vocab_size)
    total_params = sum(p.numel() for p in model.parameters())

    # AdamW with weight decay separation (best practice)
    # Apply weight decay only to matrix weights (dim >= 2), not biases/LayerNorm
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': 0.1},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=3e-4)

    print(f"Weight decay applied to {len(decay_params)} tensors ({sum(p.numel() for p in decay_params):,} params)")
    print(f"No weight decay for {len(no_decay_params)} tensors ({sum(p.numel() for p in no_decay_params):,} params)")

    # ── Training configuration ──────────────────────────────────────

    batch_size = 16
    block_size = 128
    eval_interval = 100
    eval_iters = 50
    best_loss = float('inf')

    print(f"Training {total_params:,}-param model on {len(data):,} tokens...")
    print(f"Config: B={batch_size}, T={block_size}, lr=3e-4")
    print()

    # ── Training loop ───────────────────────────────────────────────

    for step in range(1000):
        # ── Evaluation phase ──
        if step % eval_interval == 0:
            model.eval()
            losses = {}
            for split_name, split_data in [('train', train_data), ('val', val_data)]:
                losses_split = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    x, y = get_batch(split_data, batch_size, block_size)
                    with torch.no_grad():
                        _, loss = model(x, y)
                    losses_split[k] = loss.item()
                losses[split_name] = losses_split.mean().item()

            print(f"Step {step:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

            # Save checkpoint if val loss improved
            if losses['val'] < best_loss:
                best_loss = losses['val']
                Path("checkpoints").mkdir(exist_ok=True)
                torch.save(model.state_dict(), "checkpoints/vanilla_best.pt")
                print(f"  → saved best model (val loss {best_loss:.4f})")

            model.train()

        # ── Training step ──
        x, y = get_batch(train_data, batch_size, block_size)

        # 1. Forward pass: compute logits and loss
        _, loss = model(x, y)

        # 2. Backward pass: compute gradients
        optimizer.zero_grad()
        loss.backward()

        # 3. Gradient clipping: prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 4. Update weights
        optimizer.step()

    print(f"\nDone! Best val loss: {best_loss:.4f}")
    print(f"Checkpoint saved: checkpoints/vanilla_best.pt")
