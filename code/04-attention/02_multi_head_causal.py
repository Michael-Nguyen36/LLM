import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # Project to Q, K, V and reshape for multi-head
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


# Test
mha = CausalMultiHeadAttention(d_model=16, n_heads=4)
x = torch.randn(1, 6, 16)
out = mha(x)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}")
print(f"✓ Multi-head causal attention works")
