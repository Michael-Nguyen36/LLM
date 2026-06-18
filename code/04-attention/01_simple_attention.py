import torch
import torch.nn.functional as F

B, T, C = 1, 4, 8  # small enough to verify manually

# Random input: 1 batch, 4 tokens, 8-dim embedding each
x = torch.randn(B, T, C)

# Single-head attention from scratch
Wq = torch.randn(C, C)
Wk = torch.randn(C, C)
Wv = torch.randn(C, C)

Q = x @ Wq  # (1, 4, 8)
K = x @ Wk  # (1, 4, 8)
V = x @ Wv  # (1, 4, 8)

scores = Q @ K.transpose(-2, -1)            # (1, 4, 4)
scores = scores / (C ** 0.5)                # scale: prevent softmax saturation
weights = F.softmax(scores, dim=-1)         # (1, 4, 4)
output = weights @ V                         # (1, 4, 8)

print(f"Q shape:     {Q.shape}")
print(f"K shape:     {K.shape}")
print(f"V shape:     {V.shape}")
print(f"Scores:     {scores.shape}")
print(f"Weights:    {weights.shape}")
print(f"Output:     {output.shape}")

# Show the attention pattern for token 0
print(f"\nAttention weights from token 0 to all tokens:\n{weights[0, 0].detach().numpy().round(3)}")
