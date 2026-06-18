import torch
import torch.nn as nn

B, T, C = 2, 8, 16

# ── FFN: expand 4x → ReLU → compress ──────────────────────────────
# Attention = "team meeting" (tokens share info)
# FFN       = "individual work time" (each token processes alone)
ffn = nn.Sequential(
    nn.Linear(C, C * 4),    # expand 4x: 16 → 64
    nn.ReLU(),
    nn.Linear(C * 4, C),    # project back: 64 → 16
)
x = torch.randn(B, T, C)
out_ffn = ffn(x)
print(f"FFN: {x.shape} → {out_ffn.shape}")
print(f"  Hidden dimension (d_ff): {C * 4}")
print(f"  Activation: ReLU (modern models use SwiGLU — Phase 2)")
print()

# ── LayerNorm ──────────────────────────────────────────────────────
# Stabilizes activations: (x - mean) / std * gamma + beta
# Researchers now understand this as keeping activations in the
# linear regime of the activation function — not too hot or too cold.
ln = nn.LayerNorm(C)
out_ln = ln(x)
print(f"LayerNorm: {out_ln.shape}")
print(f"  Mean: {out_ln.mean().item():.4f}, Std: {out_ln.std().item():.4f}")
print(f"  (should be ~0.0 and ~1.0)")
print()

# ── Residual connection ────────────────────────────────────────────
# "Express lane on a highway" — gradient can skip the sublayer entirely
skip = x + out_ffn          # gradient flows through both paths
print(f"Residual x + FFN(x): {skip.shape}")
print()

# ── Build a mini block (FFN + LayerNorm + residual) ────────────────
class MiniFFNBlock(nn.Module):
    """Minimal post-norm block: FFN only (no attention)."""
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x):
        return self.norm(x + self.ffn(x))

mini = MiniFFNBlock(C)
out = mini(x)
print(f"Mini FFN block: {x.shape} → {out.shape}")
print(f"  Parameters: {sum(p.numel() for p in mini.parameters()):,}")

# Verify LayerNorm statistics
print()
print("Verification: LayerNorm output stats over 1000 random inputs...")
means, stds = [], []
for _ in range(1000):
    x_rand = torch.randn(16, 16) * 5.0 + 3.0  # random mean/std
    ln_out = nn.LayerNorm(16)(x_rand)
    means.append(ln_out.mean().item())
    stds.append(ln_out.std().item())
print(f"  Mean across runs: {torch.tensor(means).mean():.4f} (target: 0.0)")
print(f"  Std  across runs: {torch.tensor(stds).mean():.4f} (target: 1.0)")
print("✓ LayerNorm correctly normalizes any input distribution")
