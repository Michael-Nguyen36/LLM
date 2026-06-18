"""
vanilla_transformer.py — Shared module (Phase 1, Chapter 06).

Self-contained: can be imported or run directly.
Contains: CausalMultiHeadAttention, VanillaDecoderBlock, VanillaDecoderOnlyTransformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ── Self-contained attention (same as Chapter 04) ──────────────────

class CausalMultiHeadAttention(nn.Module):
    """Multi-head causal self-attention. No RoPE yet — pure 2017 style."""
    def __init__(self, d_model=16, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)                                                 # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)                                    # each (B, T, C)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)     # (B, H, T, d_head)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)        # (B, H, T, T)

        # Causal mask: prevent attending to future tokens
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device)).view(1, 1, T, T)
        scores = scores.masked_fill(~mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(B, T, C)    # (B, T, C)
        return self.out(out)


# ── Vanilla decoder block (same as Chapter 05) ─────────────────────

class VanillaDecoderBlock(nn.Module):
    """One transformer decoder block with post-norm (2017 style).

    Layout:
        x = LayerNorm(x + MultiHeadAttention(x))   ← post-norm
        x = LayerNorm(x + FFN(x))                   ← post-norm
    """
    def __init__(self, d_model=16, n_heads=4):
        super().__init__()
        self.attn = CausalMultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Post-norm: LayerNorm AFTER residual addition
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
        return x


# ── Full transformer model ─────────────────────────────────────────

class VanillaDecoderOnlyTransformer(nn.Module):
    """Full decoder-only transformer: embeddings → N blocks → head.

    Sinusoidal positional encodings, post-norm LayerNorm, ReLU FFN.
    Weight init: N(0, 0.02) for all parameters with dim ≥ 2.
    """
    def __init__(self, vocab_size=65, d_model=128, n_heads=4, n_layers=4, max_seq_len=256):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # Sinusoidal positional encoding (fixed, not learned)
        pe = torch.zeros(1, max_seq_len, d_model)
        pos = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)
        inv_freq = 10000 ** (torch.arange(0, d_model, 2, dtype=torch.float) / d_model)
        pe[0, :, 0::2] = torch.sin(pos / inv_freq)
        pe[0, :, 1::2] = torch.cos(pos / inv_freq)
        self.register_buffer('pe', pe, persistent=False)

        self.blocks = nn.ModuleList([
            VanillaDecoderBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        # Standard N(0, 0.02) init for all matrix weights
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
        # Biases and LayerNorm params (dim < 2) keep default init (zeros/ones)

    def forward(self, idx, targets=None):
        """Forward pass.

        Args:
            idx: token IDs, shape (B, T)
            targets: optional target IDs, shape (B, T) — used for training

        Returns:
            logits: shape (B, T, vocab_size)
            loss: scalar or None
        """
        B, T = idx.shape
        assert T <= self.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"
        )
        # Token embeddings + positional encoding
        x = self.token_embed(idx) + self.pe[:, :T, :]

        # Pass through decoder blocks
        for block in self.blocks:
            x = block(x)

        # Final LayerNorm and projection to vocab
        logits = self.head(self.ln_final(x))

        # Loss computation (only during training)
        loss = None
        if targets is not None:
            # F.cross_entropy expects (N, C) logits and (N) targets
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss


# ── Standalone verification ────────────────────────────────────────

if __name__ == "__main__":
    model = VanillaDecoderOnlyTransformer()
    x = torch.randint(0, 65, (2, 32))
    logits, loss = model(x)  # no targets → loss is None
    print(f"Logits shape: {logits.shape}  (expected: [2, 32, 65])")
    print(f"Loss (no targets): {loss}")

    # With targets
    targets = torch.randint(0, 65, (2, 32))
    logits, loss = model(x, targets)
    print(f"Loss (with targets): {loss.item():.4f}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✓ VanillaDecoderOnlyTransformer works correctly")
