import torch

# 01_char_tokenizer.py — Character-level tokenizer for TinyShakespeare
# Maps each unique character to an integer, and back.
# This is the simplest possible tokenizer — modern LLMs use BPE (Byte-Pair Encoding)
# with 50,257+ tokens, but the core idea is the same: text → integers → text.

text = "ROMEO: I dreamt a dream tonight."
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encode: string → list of integers
ids = [stoi[ch] for ch in text]
print(f"Text:      {text}")
print(f"Encoded:   {ids}")
print(f"Vocab size: {vocab_size}")

# Decode: list of integers → string
decoded = "".join(itos[i] for i in ids)
print(f"Decoded:   {decoded}")
print(f"Match:     {text == decoded}")

# PyTorch tensor
ids_tensor = torch.tensor(ids, dtype=torch.long)
print(f"Tensor:    {ids_tensor.shape}")
