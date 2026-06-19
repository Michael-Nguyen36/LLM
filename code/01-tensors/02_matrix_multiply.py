import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 02_matrix_multiply.py
# Goal: Build intuition for matrix multiplication — the single most
# important operation in neural networks (and LLMs).
#
# Key shape rule: (A, B) @ (B, C) → (A, C)
# The inner dimensions (B) must match; they "cancel out".
# ---------------------------------------------------------------------------

# --- Simulate data: 4 items, each described by 16 numbers (features) ---
X = torch.randn(4, 16)          # shape: (4, 16) — 4 samples, 16 features each
print(f"Input X shape: {X.shape}")

# --- Simulate a weight matrix: transforms 16 inputs → 8 outputs ---
W = torch.randn(16, 8)          # shape: (16, 8)
print(f"Weight W shape: {W.shape}")

print()

# --- Manual matrix multiplication: X @ W ---
# Rule: (4, 16) @ (16, 8) → (4, 8)
output = X @ W
print(f"X @ W = output of shape {output.shape} "
      f"(4 items, each now has 8 features)")
print(f"Manual matmul result (first 2 rows):\n{output[:2]}")

print()

# ---------------------------------------------------------------------------
# nn.Linear does the exact same thing, but also adds a bias term.
# ---------------------------------------------------------------------------
linear = nn.Linear(16, 8)        # Create a layer: 16 inputs → 8 outputs
# nn.Linear randomly initialises its own weight matrix (shape 8×16) and bias

out2 = linear(X)                 # Forward pass — internally does X @ W.T + bias
print(f"nn.Linear output shape: {out2.shape}")

# Note: X @ W and linear(X) use DIFFERENT random weights, so their output
# numbers won't match — only the SHAPES should match. In real code you
# use one or the other.

# We can grab the weight that nn.Linear created internally:
W_linear = linear.weight         # shape: (8, 16) — note: transposed compared to W!
print(f"  Internal weight shape: {W_linear.shape}")

print()

# ---------------------------------------------------------------------------
# Why shape (A, B) @ (B, C) → (A, C) works:
# Each output[i][j] = sum over k of input[i][k] * weight[k][j]
# i runs 0..A-1, k runs 0..B-1, j runs 0..C-1
# The inner dimension B must match, and it disappears.
# ---------------------------------------------------------------------------

# --- Batched input: process multiple groups at once ---
batched_X = torch.randn(2, 4, 16)   # 2 batches, each with 4 items, 16 features
print(f"Batched input shape: {batched_X.shape}")

batched_out = batched_X @ W          # (2, 4, 16) @ (16, 8) → (2, 4, 8)
print(f"Batched output shape: {batched_out.shape} "
      f"(2 batches of 4 items, each with 8 features)")

print()

# --- Compare manual vs nn.Linear outputs (they should be close) ---
# Make a fresh Linear with NO bias so it's a pure weight @ input
linear_no_bias = nn.Linear(16, 8, bias=False)
X_small = torch.randn(1, 16)

manual = X_small @ linear_no_bias.weight.T   # need .T because Linear stores (out, in)
pytorch_out = linear_no_bias(X_small)

print(f"Manual:     {manual}")
print(f"PyTorch:    {pytorch_out}")
print(f"Match?      {torch.allclose(manual, pytorch_out)}")
