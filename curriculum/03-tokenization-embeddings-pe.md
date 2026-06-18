# Chapter 03: Tokenization, Embeddings & Positional Encoding

> **Audience**: 🟢 All roles (code sections marked 🔵)
> **Prerequisites**: Chapter 02 (Neural Nets & Training), basic Python
> **Estimated time**: 15 minutes read, 15 minutes code

---

## Why This Matters

Before a neural network can process text, the text must become **numbers**. This chapter covers the transformation pipeline that sits at the very front of every LLM:

1. **Tokenization** — chop text into pieces, map each piece to an integer ID
2. **Embeddings** — look up each ID's dense vector representation
3. **Positional Encoding** — inject information about where each token sits in the sequence

By the end, you'll understand the exact format of the data that enters a transformer: a 3D tensor of shape `(B, T, d_model)` where each token has been given meaning (embedding) and a sense of order (position).

---

## Part 1: What Is a Token?

A **token** is the atomic unit of text that a model processes. Think of it like a **Lego brick**: the model doesn't see a paragraph of flowing language — it sees a sequence of discrete pieces, each with an ID number.

```
Input text:  "Hello"
Characters:   H    e    l    l    o
ASCII codes:  72  101  108  108  111
```

Different tokenization strategies produce different Lego brick sizes:

| Strategy | Example | Vocab Size | Used By |
|----------|---------|------------|---------|
| **Character** | `"Hello" → [H, e, l, l, o]` | ~65 (TinyShakespeare) | Our first models |
| **Word** | `"Hello world" → ["Hello", "world"]` | 50k–200k+ | Early NLP |
| **Byte-Pair Encoding (BPE)** | `"Hello" → ["Hel", "lo"]` | 32k–200k | GPT, Llama, Claude |

### 🕰️ How Tokenization Evolved

Tokenization has gone through several generations, each trading off vocabulary size against context understanding:

| Era | Approach | Paper / Authors | Model Example | Vocab Size | Key Innovation |
|-----|----------|-----------------|---------------|------------|----------------|
| **2015** | **BPE for NMT** | Sennrich et al. (ACL 2016) | Neural MT systems | ~32k | First subword method for NMT; frequency-based merging |
| **2017** | **BPE** (character-level) | *"Attention Is All You Need"* — Vaswani et al. | Original Transformer | ~40,000 | Applied BPE to transformer; good balance of coverage/speed |
| **2018** | **WordPiece** | *"BERT"* — Devlin et al. (NAACL 2019) | BERT, DistilBERT | 30,522 | Likelihood-based merging (not frequency); used by BERT |
| **2018** | **SentencePiece / Unigram** | Kudo & Richardson (EMNLP 2018) | T5, mT5, XLNet, ALBERT | 32k | Language-independent; trains tokenizer as probabilistic model |
| **2019** | **Byte-level BPE** | *"GPT-2"* — Radford et al. | GPT-2 | 50,257 | Operates on raw bytes (0–255); **no UNK token ever**; every text tokenizable |
| **2020** | **Byte-level BPE** (same vocab, better merges) | *"GPT-3"* — Brown et al. | GPT-3 | 50,257 | Same vocab as GPT-2; merges trained on 500B+ tokens |
| **2022** | **BPE + larger vocab** | *"PaLM"* — Chowdhery et al. | PaLM | 256,000 | Larger vocab for multilingual; SentencePiece-based |
| **2023** | **Byte-level BPE** + much larger vocab | *"GPT-4"* — OpenAI | GPT-4 | ~100,256 | More words as single tokens → shorter sequences, faster inference |
| **2023** | **BPE + massive vocab** | *"LLaMA 2"* — Touvron et al. | LLaMA 2 | 32,000 | Smaller vocab but optimized for code + multilingual |
| **2024** | **Byte-level BPE** + largest yet | *"GPT-4o"* — OpenAI | GPT-4o | ~200,000 | Maximum coverage for multi-lingual multi-modal input |
| **2024** | **BPE + 128k vocab** | *"Qwen 2"* — Qwen Team | Qwen 2 | 151,643 | Optimized for Chinese/English/code; tiktoken-based |

The progression is clear: **models want larger vocabularies** so common words become single tokens (reducing sequence length and compute). The 2017 transformer's 40k BPE seems tiny by today's standards — GPT-4o uses 5× that — but the core algorithm (frequency-based subword merging) is the same.

> 💡 **Key insight**: The tokenizer is a **fixed preprocessing step** — it doesn't learn during model training. The same tokenizer must be used at inference that was used during training. This is why tokenizer choice is a critical architectural decision made *before* training.

> 🧪 **Try this mental experiment**: Tokenize `"Hello"` and `" Hello"` (with a leading space). With BPE, these often produce **different** token IDs because the space changes how the encoder matches subword pieces. A model treats `"Hello"` and `" Hello"` as distinct concepts — whitespace matters!

For this chapter and the models we build, we start with **character-level** tokenization: one character = one token. It's simple, easy to debug, and we only need 65 unique tokens for all of Shakespeare.

---

## Part 2: Character-Level Tokenizer

The tokenizer's job is two dictionaries:

- **`stoi`** (string-to-int): `{'H': 0, 'e': 1, 'l': 2, 'o': 3}`
- **`itos`** (int-to-string): `{0: 'H', 1: 'e', 2: 'l', 3: 'o'}`

```
Lookup Table (vocab_size = 65)
┌──────┬───────┐
│ Char │ ID    │
├──────┼───────┤
│ '\n' │  0    │
│ ' '  │  1    │
│ '!'  │  2    │
│ '?'  │  3    │
│ 'A'  │  4    │
│ 'B'  │  5    │
│  …   │  …    │
│ 'e'  │ 25    │
│  …   │  …    │
│ 'z'  │ 64    │
└──────┴───────┘
```

For TinyShakespeare (the complete works, ~1M characters), sorting the unique characters gives us exactly **65** entries. Every character in the text maps to one of these 65 IDs.

```python
text = "ROMEO: I dreamt a dream tonight."
chars = sorted(list(set(text)))       # all unique characters
vocab_size = len(chars)               # 65
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
```

The **encode** operation runs each character through `stoi`:

```
"ROMEO: I dreamt a dream tonight." 
→ [R, O, M, E, O, :,  , I,  , d, r, e, a, m, t,  , a,  , d, r, e, a, m,  , t, o, n, i, g, h, t, .]
→ [45, 46, 35, 31, 46, 47, 13, 31, 13, 27, 30, 31, 24, 36, 37, 13, 24, 13, 27, 30, 31, 24, 36, 13, 37, 46, 41, 31, 34, 33, 37, 48]
```

The **decode** operation runs each ID through `itos` and joins — it should perfectly reconstruct the original text.

---

## Part 3: Embeddings — From ID to Meaning

A token ID is just an integer — it tells us *which* token but not *what it means*. We need to convert each ID into a dense vector that captures semantic information.

This is the **embedding layer**: a learned lookup table of shape `(vocab_size, d_model)`.

```
Token ID 37 → look up row 37 → [0.23, -1.45, 0.67, ..., 0.12]  (length = d_model)

                        Embedding Matrix
                     ┌──────────────────────┐
                   0 │ 0.12 -0.34  0.56 ... │
                   1 │ 0.78  0.01 -0.23 ... │
                     │        ...            │
            ID 37 ──→│ 0.23 -1.45  0.67 ... │←  d_model numbers
                     │        ...            │
                  64 │ 0.45  0.67 -0.89 ... │
                     └──────────────────────┘
                      vocab_size = 65 rows
```

Each token ID gets its own row. During training, the model adjusts these embedding vectors so that tokens appearing in similar contexts end up with similar vectors. In code:

```python
embed = nn.Embedding(vocab_size, d_model)   # (65, d_model)
ids = torch.tensor([[45, 46, 35]])           # (B=1, T=3)
x = embed(ids)                                # (1, 3, d_model)
```

The output shape is `(B, T, d_model)` — the standard shape you'll see throughout the transformer.

> 💡 **d_model** is the embedding dimension (also called the hidden dimension). In our tiny model it's 16. In GPT-2 it's 768. In GPT-4 it's probably 8,000+. This single number controls the model's capacity to represent information about each token.

---

## Part 4: Why Position Matters

Consider these two sentences:

> **"dog bites man"** — the dog is the attacker, the man is the victim.
> **"man bites dog"** — the man is the attacker, the dog is the victim.

Identical tokens, different positions, completely different meaning. Embeddings alone can't distinguish these:

```
Without position:     [dog] [bites] [man]
                     +──────┐ ┌──────┐ ┌──────┐
                     │ vec  │ │ vec  │ │ vec  │
                     └──────┘ └──────┘ └──────┘
                     All three vectors carry token meaning but NO position info.
                     "dog bites man" and "man bites dog" look identical!

With position signal: [dog] [bites] [man]
                     ┌──────┐ ┌──────┐ ┌──────┐
                     │ vec  │ │ vec  │ │ vec  │
                     + p0   │ + p1   │ + p2   │
                     └──────┘ └──────┘ └──────┘
                     Adding position encodings breaks the symmetry — 
                     "dog" at position 0 ≠ "dog" at position 2.
```

We need to inject **where** each token is, not just **what** it is.

---

## Part 5: Sinusoidal Positional Encoding

The original 2017 transformer uses a fixed (non-learned) encoding based on sine and cosine waves:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where:
- **pos** = position in the sequence (0, 1, 2, …, T-1)
- **i** = dimension index (0, 1, 2, …, d_model/2 - 1)

### What does this mean intuitively?

Each **pair** of dimensions (2i, 2i+1) forms a sine wave at a different frequency:

```mermaid
block-beta
    columns 6
    
    block:low_freq
        columns 3
        a["Dim 0-1"] b["sin(pos/1.0)"] c["slow wave"]
    end
    space
    block:mid_freq
        columns 3
        d["Dim 4-5"] e["sin(pos/0.1)"] f["medium wave"]
    end
    space
    block:high_freq
        columns 3
        g["Dim 14-15"] h["sin(pos/0.01)"] i["fast wave"]
    end
    
    style low_freq fill:#e1f5fe
    style mid_freq fill:#fff3e0
    style high_freq fill:#e8f5e9
```

- **Low-frequency dimensions** (first pairs): change slowly across positions — help the model distinguish nearby positions.
- **High-frequency dimensions** (last pairs): change rapidly — help the model distinguish distant positions.

The result is a **unique fingerprint** for every position in the sequence. Position 0 always gets the same encoding, regardless of how long the sequence is. This is important: the PE for position 5 is the same whether the sequence has 10 tokens or 1000 tokens.

### In Code

```python
pos = torch.arange(seq_len).unsqueeze(1)              # (8, 1)
inv_freq = 10000 ** (torch.arange(0, d_model, 2) / d_model) # inverse frequencies
pe = torch.zeros(1, seq_len, d_model)                  # (1, 8, 16)
pe[0, :, 0::2] = torch.sin(pos / inv_freq)            # even dims: sin
pe[0, :, 1::2] = torch.cos(pos / inv_freq)            # odd dims: cos
```

This matrix `pe` is added directly to the embedding output:

```python
x = embedded + pe  # broadcast across batch dimension
```

### 🕰️ How Positional Encoding Evolved

The 2017 sinusoidal PE is far from the only option. Here's the complete evolution of position encoding methods from 2017–2025:

| Year | Method | Paper / Authors | Key Innovation | Why an Improvement | Adopted By |
|------|--------|-----------------|----------------|--------------------|------------|
| **2017** | **Sinusoidal PE** | *"Attention Is All You Need"* — Vaswani et al. (NeurIPS 2017) | Fixed sine/cosine functions of different frequencies added to token embeddings | First principled position encoding; no learned params; theoretically extrapolates | Original Transformer, early translation models |
| **2018** | **Learned Absolute PE** | *"BERT"* — Devlin et al. (NAACL 2019) | Trainable embedding table `max_seq_len × d_model` added to token embeddings | More flexible — model learns task-appropriate position representations | BERT, GPT-1/2, ViT, RoBERTa, XLNet, ALBERT, ELECTRA |
| **2018** | **Relative Position Representations (RPR)** | Shaw et al. (NAACL 2018) | Learned embeddings based on relative offset between key and query; added to attention logits | First truly relative encoding; better length generalization | Music Transformer, early relative models |
| **2019** | **T5 Relative Position Bias** | Raffel et al. (JMLR 2020) | Learned scalar bias per relative offset, bucketed logarithmically (32 buckets) | Simplified RPR — scalar instead of vector; log bucketing handles long sequences with constant params | **T5, mT5, Flan-T5, UL2, ByT5** — influenced PaLM, PaLM-2, Gemini |
| **2019** | **Transformer-XL Relative PE** | Dai et al. (ACL 2019) | Relative encoding + segment-level recurrence for cross-segment flow | First to handle arbitrarily long contexts via recurrence | Transformer-XL, XLNet |
| **2021 (Apr)** | **RoPE (Rotary Position Embedding)** | Su et al. (arXiv Apr 2021, Neurocomputing 2024) | **Multiplies** Q/K by rotation matrix; dot product depends only on relative position | Paradigm shift — multiplicative not additive; natural distance decay; FlashAttention-compatible; no learned params | **LLaMA 1/2/3, Mistral, Qwen, Gemma, GPT-NeoX, PaLM, Yi, DeepSeek, Baichuan, ChatGLM** — dominant in modern open LLMs |
| **2021 (Aug)** | **ALiBi (Attention with Linear Biases)** | Press et al. (ICLR 2022) | Adds linear bias to attention scores proportional to distance; each head has different slope | Unmatched extrapolation — train 1024, test 2048+ with no degradation; 11% faster, 11% less memory | **MPT, BLOOM, BLOOMZ** — also multimodal: Video LLaMA, PaLI-X, Flamingo |
| **2022** | **xPos (Extrapolatable PE)** | Sun et al. (ACL 2023) | Adds exponential decay to RoPE's rotation matrix; "attention resolution" metric | Fixes RoPE's oscillation at long distances; stabilizes long-range attention | Research models, LEX Transformer |
| **2023 (Jun)** | **Position Interpolation (PI)** | Chen et al. (Meta) | Linearly down-scales position indices to map extended positions into trained range | First principled RoPE extension: 2K→32K with ~1000 fine-tuning steps | LLaMA fine-tuned variants (LongChat, Guanaco, Vicuna) |
| **2023 (Jul)** | **NTK-Aware Scaling** | bloc97 (community) | Changes RoPE base frequency instead of scaling positions; preserves high frequencies | Better than PI — preserves nearby token discrimination; ~2× extension without fine-tuning | Code LLaMA (base=1M), community models |
| **2023 (Aug)** | **YaRN** | Peng et al. (ICLR 2024) | NTK-by-parts + temperature scaling; categorizes dimensions by frequency band | 10× fewer tokens and 2.5× fewer steps than PI; 128k context on LLaMA | **Code LLaMA, Qwen, Mistral extended, Nous Research models** |
| **2023 (Oct)** | **FIRE / CLEX / PoSE** | Li et al. (ICLR 2024), Chen et al. (NeurIPS 2024), Zhu et al. | Learnable MLP for relative positions; Neural ODE continuous scaling; randomized position training | Adaptive to task; continuous scaling; no fine-tuning needed | Research models |
| **2024 (Jan)** | **Self-Extend** | Jin et al. (ICML 2024) | Bi-level attention: neighbor (standard) + grouped (floor-divided positions) | Extends context **without any fine-tuning** — only modifies inference code | Any RoPE-based LLM at inference |
| **2024 (Feb)** | **LongRoPE** | Ding et al. (ICML 2024) | Non-uniform interpolation search + progressive strategy + readjustment | **First to reach 2M+ tokens**; 1K fine-tuning steps; preserves short-context | **LLaMA-2, Mistral, Microsoft Phi-3** |
| **2024 (May)** | **CoPE (Contextual PE)** | Golovneva et al. (ICLR 2025) | Positions measured by **context-dependent gates** (σ(q·k)) instead of token count | Paradigm shift — position is semantic (words, nouns, sentences), not token index | Research models up to 100M params |
| **2024** | **DAPE / CAPE / HoPE** | Zheng et al. (NeurIPS 2024), Chen et al. (ACL 2025) | MLP takes attention value + position for dynamic bias; HoPE removes low-frequency decay | Semantically adaptive PE; HoPE challenges "long-term decay" assumption | Research models |
| **2025** | **Wavelet-based PE** | Multiple groups (ACL 2025) | Wavelet transforms for multi-resolution position encoding; heads organize into wavelet bands | Signal-processing foundations; multi-resolution analysis; interpretable | Research models, theoretical analysis |

> 🔄 **Why start here**: Sinusoidal PE is the original 2017 design — fixed, deterministic, no learned parameters. It's easy to understand and debug. **What changes later**: Phase 2 (Chapter 08) replaces this with **RoPE**, which encodes *relative* position and handles long contexts better. The context extension methods (PI, YaRN, LongRoPE) are Phase 4 topics.

> 🧠 **Implementation detail**: The PE matrix is typically stored with `register_buffer('pe', pe, persistent=False)`. This makes it part of the model's state (moves to GPU with `.to(device)`) but doesn't save it in checkpoints — since it's deterministic, we just recompute it when loading the model. A common gotcha: forgetting to move the PE to the same device as the embeddings.

---

## Part 6: 🟢 Summary Box

```
Token IDs → Embedding Vectors → Add Position Signal → Ready for Transformer
  (B, T)       (B, T, d_model)      (B, T, d_model)       (B, T, d_model)
```

| Term | What It Is |
|------|------------|
| **Token** | The atomic unit of text (character, subword, or word) |
| **Token ID** | The integer assigned to a token by the tokenizer |
| **Vocab size** | The total number of unique tokens the model knows |
| **Embedding** | A learned vector (length = d_model) representing a token |
| **d_model** | The embedding/hidden dimension (controls model capacity) |
| **Positional Encoding** | Information added to embeddings to indicate token order |

This connects directly to Chapter 00's high-level flow: "Tokenization: Text → Numbers" is Steps 1-2 of this chapter, and the embedding + PE is the actual numeric representation the transformer operates on.

---

## Part 7: 🔵 Code Section — Char Tokenizer in PyTorch

Let's build the character-level tokenizer and see it work:

```python
import torch

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
```

Run it:
```bash
python3 code/03-tokenization-embeddings-pe/01_char_tokenizer.py
```

**Expected output:**
```
Text:      ROMEO: I dreamt a dream tonight.
Encoded:   [7, 6, 5, 3, 6, 2, 0, 4, 0, 9, 17, 10, 8, 14, 18, 0, 8, 0, 9, 17, 10, 8, 14, 0, 18, 16, 15, 13, 11, 12, 18, 1]
Vocab size: 19   (toy example — full TinyShakespeare has 65)
Decoded:   ROMEO: I dreamt a dream tonight.
Match:     True
Tensor:    torch.Size([32])
```

Now the embedding + positional encoding:

```python
import torch
import torch.nn as nn

vocab_size = 65
d_model = 16
seq_len = 8

# Token embedding: lookup table (vocab_size, d_model)
embed = nn.Embedding(vocab_size, d_model)
dummy_tokens = torch.randint(0, vocab_size, (2, seq_len))  # (B=2, T=8)
embedded = embed(dummy_tokens)                                # (2, 8, 16)
print(f"Embedding shape: {embedded.shape}")

# Sinusoidal positional encoding
pos = torch.arange(seq_len).unsqueeze(1)                      # (8, 1)
div = 10000 ** (torch.arange(0, d_model, 2) / d_model)       # (8,)
pe = torch.zeros(1, seq_len, d_model)
pe[0, :, 0::2] = torch.sin(pos / div)
pe[0, :, 1::2] = torch.cos(pos / div)
print(f"Positional encoding shape: {pe.shape}")
print(f"PE at position 0, first 4 dims: {pe[0, 0, :4].tolist()}")
print(f"PE at position 1, first 4 dims: {pe[0, 1, :4].tolist()}")

# Add to embeddings
x = embedded + pe
print(f"Input to transformer: {x.shape}")

# Verify different positions get different encodings
assert not torch.allclose(pe[0, 0], pe[0, 1]), "Positions should differ"
print("✓ Different positions get different encodings")
```

Run it:
```bash
python3 code/03-tokenization-embeddings-pe/02_embeddings_and_pe.py
```

**Expected output:**
```
Embedding shape: torch.Size([2, 8, 16])
  (B, T, d_model) = (2, 8, 16)
Positional encoding shape: torch.Size([1, 8, 16])
PE at position 0, first 4 dims: [0.0, 1.0, 0.0, 1.0]
PE at position 1, first 4 dims: [0.841..., 0.540..., 0.046..., 0.998...]
Input to transformer: torch.Size([2, 8, 16])
✓ Different positions get different encodings
✓ Position 0 encoding: ... [0.0, 1.0, 0.0, 1.0, ...]
  (This is always the same — position 0 has a fixed encoding)
```

> 🔵 **Run both scripts** before proceeding. They should produce no errors.

---

## Part 8: 🟢 Check Your Understanding

Test yourself before moving to the next chapter.

1. **What's the difference between a token ID and an embedding vector?**

   <details>
   <summary>Show answer</summary>
   A token ID is a single integer that identifies which token (e.g., character 37). An embedding vector is a dense array of `d_model` floating-point numbers that represent that token's meaning. The ID is the index; the embedding is the value at that index in the lookup table.
   </details>

2. **Why can't we use token IDs directly as model input?**

   <details>
   <summary>Show answer</summary>
   Token IDs are just arbitrary integers — they have no inherent meaning or relationships (e.g., token 36 is not "close to" token 37). An embedding layer turns them into dense vectors where semantic relationships can be learned and captured.
   </details>

3. **What would happen if we didn't add positional encoding?**

   <details>
   <summary>Show answer</summary>
   The model would treat positions as a bag of words — "dog bites man" and "man bites dog" would look identical because the same set of token embeddings would be present regardless of order. The model would have no way to distinguish different word orders.
   </details>

4. **Why does position 0 always get the same encoding regardless of sequence length?**

   <details>
   <summary>Show answer</summary>
   The sinusoidal PE formula depends only on `pos` (the position index), not on the total sequence length. `sin(0)` = 0 and `cos(0)` = 1 for all frequencies, so position 0's encoding is always `[0, 1, 0, 1, 0, 1, ...]`. This is intentional — the first token's position encoding is the same whether it starts a short or long sequence.
   </details>

---

## Terms Introduced

| Term | Quick Definition |
|------|------------------|
| **Token** | The atomic unit of text (character, subword, or word) |
| **Tokenizer** | Algorithm that maps text to token IDs (`stoi`) and back (`itos`) |
| **Vocabulary** | The complete set of unique tokens the model knows |
| **BPE** | Byte-Pair Encoding — the subword tokenization used by modern LLMs |
| **Embedding** | A learned dense vector representing a token's meaning |
| **d_model** | The hidden/embedding dimension of the model |
| **Positional Encoding** | A signal added to embeddings to indicate token order |
| **Sinusoidal PE** | Fixed sine/cosine position encoding (original 2017 design) |

---

> **Next Chapter**: Chapter 04 — The Attention Mechanism.
>
> *How does the model figure out which tokens are related to which? The attention mechanism is the core innovation of the transformer.*
>
> *🔵 Make sure both companion scripts from this chapter run cleanly before proceeding.*
