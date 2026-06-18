"""
02_generation.py — Autoregressive generation with temperature and top-k sampling.

Usage:
    python3 code/06-training-generation/02_generation.py

Loads the checkpoint from checkpoints/vanilla_best.pt (created by
01_training_loop.py) and generates Shakespeare-like text.

Expected output (without checkpoint): "No checkpoint found" error.
Expected output (with checkpoint): Generated Shakespeare-style text.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path (stdlib has a 'code' module that shadows our directory)
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn.functional as F


def generate(model, prompt_ids, max_new_tokens=200, temperature=1.0, top_k=40):
    """Autoregressive generation with temperature and top-k sampling.

    Args:
        model: VanillaDecoderOnlyTransformer in eval mode.
        prompt_ids: (1, T) tensor of token IDs to start generation.
        max_new_tokens: number of tokens to generate.
        temperature: scaling factor for logits (lower = more deterministic).
        top_k: keep only the top-k logits before sampling.

    Returns:
        (1, T + max_new_tokens) tensor with the generated sequence.
    """
    model.eval()
    idx = prompt_ids

    for _ in range(max_new_tokens):
        # Crop to model's maximum context window
        idx_cond = idx[:, -model.max_seq_len:]

        # Forward pass (no gradient needed for inference)
        with torch.no_grad():
            logits, _ = model(idx_cond)

        # Focus on the last token's logits
        # Temperature scaling happens BEFORE truncation
        logits = logits[:, -1, :] / temperature

        # Top-k: zero out everything outside the top k
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = float('-inf')

        # Convert to probabilities and sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append to sequence
        idx = torch.cat([idx, next_token], dim=1)

    return idx


# ── Demo ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from code.vanilla_transformer import VanillaDecoderOnlyTransformer

    # Build vocabulary from the training data
    text = open("data/tinyshakespeare.txt").read()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)

    # Initialize model
    model = VanillaDecoderOnlyTransformer(vocab_size=vocab_size)

    # Load checkpoint
    ckpt_path = Path("checkpoints/vanilla_best.pt")
    if not ckpt_path.exists():
        print("No checkpoint found. Run the training loop first to create checkpoints/vanilla_best.pt.")
        exit(1)

    state = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(state)
    print(f"Loaded checkpoint from {ckpt_path}")

    # Generate
    prompt = "ROMEO:"
    prompt_ids = torch.tensor([[stoi[ch] for ch in prompt]], dtype=torch.long)
    output_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=100,
        temperature=0.8,
        top_k=40,
    )
    output_text = "".join(itos[i.item()] for i in output_ids[0])

    print(f"Prompt: {prompt}")
    print("-" * 50)
    print(output_text)
    print("-" * 50)
    print(f"\nGenerated {len(output_ids[0]) - len(prompt_ids[0])} tokens")
