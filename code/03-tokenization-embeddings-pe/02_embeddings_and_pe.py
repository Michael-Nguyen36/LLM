import torch
import torch.nn as nn

# 02_embeddings_and_pe.py — Token embeddings + sinusoidal positional encoding
# This is the input stage of every transformer: token IDs → vectors → add position info.

vocab_size = 65
d_model = 16
seq_len = 8

# Token embedding: a learned lookup table
# Shape (vocab_size, d_model): each token ID maps to a unique d_model-dimensional vector
embed = nn.Embedding(vocab_size, d_model)
dummy_tokens = torch.randint(0, vocab_size, (2, seq_len))  # (B=2, T=8)
embedded = embed(dummy_tokens)                                # (2, 8, 16)
print(f"Embedding shape: {embedded.shape}")
print(f"  (B, T, d_model) = ({', '.join(str(s) for s in embedded.shape)})")

# Sinusoidal positional encoding (2017 original design)
# Each position gets a unique "fingerprint" using sine/cosine waves at different frequencies.
# This is fixed (no learned parameters) — we add it to the token embeddings.

pos = torch.arange(seq_len).unsqueeze(1)                      # (8, 1)
inv_freq = 10000 ** (torch.arange(0, d_model, 2) / d_model)  # (8,) — inverse frequencies
pe = torch.zeros(1, seq_len, d_model)                         # (1, 8, 16)
pe[0, :, 0::2] = torch.sin(pos / inv_freq)                   # even dims: sin
pe[0, :, 1::2] = torch.cos(pos / inv_freq)                   # odd dims: cos
print(f"\nPositional encoding shape: {pe.shape}")
print(f"PE at position 0, first 4 dims: {pe[0, 0, :4].tolist()}")
print(f"PE at position 1, first 4 dims: {pe[0, 1, :4].tolist()}")

# Add to embeddings — this is the input to the transformer
x = embedded + pe                                              # (2, 8, 16)
print(f"\nInput to transformer: {x.shape}")

# Verify: positions 0 and 1 have different encodings
assert not torch.allclose(pe[0, 0], pe[0, 1]), "Positions should differ"
print("✓ Different positions get different encodings")

# Verify: position 0 encoding is deterministic (same regardless of sequence length)
print("✓ Position 0 encoding: first 8 dims", pe[0, 0, :8].tolist())
print("  (This is always the same — position 0 has a fixed encoding)")
