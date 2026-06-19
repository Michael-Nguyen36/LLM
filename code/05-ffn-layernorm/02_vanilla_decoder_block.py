import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalMultiHeadAttention(nn.Module):
    """Self-contained copy so this file runs standalone.

    Multi-head causal self-attention — 2017 style (no RoPE yet).
    """
    def __init__(self, d_model=16, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class VanillaDecoderBlock(nn.Module):
    """One transformer decoder block with post-norm (2017 style).

    Layout:
        x = LayerNorm(x + Dropout(MultiHeadAttention(x)))   ← post-norm
        x = LayerNorm(x + Dropout(FFN(x)))                   ← post-norm

    "Post" means LayerNorm comes AFTER the residual addition.

    Weight init: N(0, 0.02) for all parameters > 1D.
    """
    def __init__(self, d_model=16, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = CausalMultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            # Biases and LayerNorm params (dim < 2) keep default init

    def forward(self, x):
        # Post-norm: LayerNorm AFTER residual addition
        x = self.norm1(x + self.dropout(self.attn(x)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


if __name__ == "__main__":
    block = VanillaDecoderBlock(d_model=16, n_heads=4)
    x = torch.randn(1, 8, 16)
    out = block(x)
    print(f"Vanilla block: {x.shape} → {out.shape}")
    params = sum(p.numel() for p in block.parameters())
    print(f"Parameters: {params:,}")

    # Verify gradient flow through residual
    loss = out.sum()
    loss.backward()
    grads = [p.grad for p in block.parameters() if p.grad is not None]
    grad_norm = torch.norm(torch.stack([g.norm() for g in grads]))
    print(f"Gradient norm (all params): {grad_norm.item():.4f}")
    print("✓ Gradients flow through residual connections")

    # Verify layer count can be stacked
    print()
    print("Stacking 6 blocks...")
    blocks = nn.Sequential(*[
        VanillaDecoderBlock(d_model=16, n_heads=4) for _ in range(6)
    ])
    out_stack = blocks(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out_stack.shape}")
    total_params = sum(p.numel() for p in blocks.parameters())
    print(f"  Total parameters (6 blocks): {total_params:,}")
    print("✓ Stack of 6 decoder blocks works")
