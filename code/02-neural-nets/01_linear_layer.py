import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 01_linear_layer.py
# Goal: Show that a neural-network layer is just:
#           output = weight @ input + bias
# (with the weight matrix transposed for convenience).
# ---------------------------------------------------------------------------

# --- Single input: 4 numbers ---
x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])   # shape: (1, 4) — 1 sample, 4 features
print(f"Input shape: {x.shape}")
print(f"Input: {x}")

print()

# --- Create a linear layer: 4 inputs → 3 outputs ---
linear = nn.Linear(4, 3)          # internally creates weight (3×4) and bias (3,)
print(f"Linear layer weight shape: {linear.weight.shape}")   # (3, 4)
print(f"Linear layer bias shape:   {linear.bias.shape}")     # (3,)

# --- Forward pass using PyTorch ---
y = linear(x)                     # shape: (1, 3)
print(f"\nPyTorch forward pass: {y.shape}")
print(f"  Result: {y}")

print()

# --- Same thing done manually: y = x @ W^T + b ---
# nn.Linear stores weight as (out_features, in_features), so we transpose
W = linear.weight                 # (3, 4)
b = linear.bias                   # (3,)

y_manual = x @ W.T + b            # (1, 4) @ (4, 3) + (3,) → (1, 3)
print(f"Manual forward pass: {y_manual.shape}")
print(f"  Result: {y_manual}")

# Check they produce the same numbers
print(f"\nPyTorch and manual match? {torch.allclose(y, y_manual)}")

print()

# ---------------------------------------------------------------------------
# A neural network is just a sequence of these transformations
# with non-linear "activation functions" in between.
#
#   input → Linear → Activation → Linear → Activation → ... → output
#
# Without activations, stacking linear layers would be pointless
# (a stack of linears is still just one linear transformation).
# ---------------------------------------------------------------------------

# --- Processing a batch of inputs at once ---
x_batch = torch.randn(8, 4)       # 8 different samples, each with 4 features
print(f"Batch input shape: {x_batch.shape}")

y_batch = linear(x_batch)         # (8, 4) → (8, 3) — same layer, all 8 at once
print(f"Batch output shape: {y_batch.shape}")
print(f"Batch output (first 3 rows):\n{y_batch[:3]}")
