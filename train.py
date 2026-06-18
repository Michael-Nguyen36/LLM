"""
Train the tiny decoder-only transformer on TinyShakespeare.

Usage:
    python train.py

Trains a ~10M param model from scratch. On CPU this takes ~5-10 minutes
for 5000 iterations on TinyShakespeare.
"""

import os
import time
import math

import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import Config
from model import DecoderOnlyTransformer
from data import load_tinyshakespeare, get_train_val_splits, get_batch


def estimate_loss(model: DecoderOnlyTransformer, train_data: torch.Tensor,
                  val_data: torch.Tensor, config: Config) -> dict:
    """Estimate train/val loss over eval_iters batches (no gradients)."""
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(config.training.eval_iters)
        for k in range(config.training.eval_iters):
            x, y = get_batch(data, config.training.batch_size,
                             config.model.max_seq_len, config.device)
            with torch.no_grad():
                _, loss, _ = model(x, targets=y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it: int, config: Config) -> float:
    """Cosine learning rate schedule with linear warmup."""
    warmup = config.training.warmup_steps
    max_iters = config.training.max_iters
    max_lr = config.training.learning_rate
    min_lr = config.training.min_lr

    # Linear warmup
    if it < warmup:
        return max_lr * (it + 1) / warmup

    # Cosine decay to min_lr
    if it > max_iters:
        return min_lr

    progress = (it - warmup) / (max_iters - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train(config: Config):
    """Main training loop."""
    # Setup
    device = torch.device(config.device)
    print(f"Device: {device}")
    print(f"Model params: ~{config.model.__dict__}")

    # Load data
    print("Loading TinyShakespeare...")
    text, tokenizer = load_tinyshakespeare(config.training.data_path)
    train_data, val_data = get_train_val_splits(text, tokenizer)
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens:   {len(val_data):,}")
    print(f"Vocab size:   {tokenizer.vocab_size}")

    # Update vocab size in config (in case it changed)
    config.model.vocab_size = tokenizer.vocab_size

    # Build model
    model = DecoderOnlyTransformer(config.model).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Optimizer — AdamW with weight decay on non-norm/bias params
    # This separation is standard: only apply weight decay to matrix weights.
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if p.dim() >= 2:  # weight matrices
            decay_params.append(p)
        else:  # biases, norms
            no_decay_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.training.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2),
    )

    # Create checkpoint directory
    os.makedirs(os.path.dirname(config.training.ckpt_path) or ".", exist_ok=True)

    # Training loop
    best_val_loss = float("inf")
    start_time = time.time()

    print(f"\nTraining for {config.training.max_iters} iterations...")
    pbar = tqdm(range(config.training.max_iters), desc="Training")

    for step in pbar:
        # Set learning rate (cosine schedule)
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluate loss
        if step % config.training.eval_interval == 0 or step == config.training.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, config)
            pbar.set_postfix({
                "train_loss": f"{losses['train']:.4f}",
                "val_loss": f"{losses['val']:.4f}",
                "lr": f"{lr:.2e}",
            })

            # Save best model
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "best_val_loss": best_val_loss,
                    "tokenizer": tokenizer,
                }, config.training.ckpt_path)

        # Training step
        x, y = get_batch(train_data, config.training.batch_size,
                         config.model.max_seq_len, config.device)

        logits, loss, _ = model(x, targets=y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping
        if config.training.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)

        optimizer.step()

    elapsed = time.time() - start_time
    print(f"\nDone! Trained {config.training.max_iters} iters in {elapsed:.1f}s")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {config.training.ckpt_path}")

    return model, tokenizer


if __name__ == "__main__":
    config = Config()
    # Auto-detect and use CUDA/ROCm if available
    if torch.cuda.is_available():
        config.device = "cuda"
    print(f"Using device: {config.device}")

    model, tokenizer = train(config)
