# LLM From Zero: Educational Curriculum Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a polished, educational curriculum that teaches a mixed-skill team (engineers, PMs, QAs, BAs) how LLMs work — from tensor basics through hybrid Mamba architectures — with no prior ML experience assumed.

**Architecture:** 6 sequential phases covering foundations → vanilla transformer → modern upgrades → training enhancements → production optimization → hybrids, plus a fintech cross-cut. Each guide chapter targets 🟢 all roles or 🔵 engineers with Mermaid diagrams and companion Python code.

**Tech Stack:** Markdown (Mermaid diagrams for concepts), PyTorch 2.12 (companion code), Python 3.12+

**Audience Labels:**
- 🟢 All = PM, QA, BA, engineers — read the story + diagrams
- 🔵 Engineers = Build the code, run the companion scripts

---

## File Structure

```
curriculum/
├── index.md                         # ✓ DONE Phase 0
├── 00-how-llms-work.md              # ✓ DONE Phase 0
├── 01-tensors-and-matmul.md         # ✓ DONE Phase 0
├── 02-neural-nets-and-training.md   # ✓ DONE Phase 0
├── 03-tokenization-embeddings-pe.md # Phase 1
├── 04-attention-mechanism.md        # Phase 1
├── 05-ffn-layernorm-stack.md        # Phase 1
├── 06-training-loop-and-generation.md # Phase 1
├── 07-pre-norm-rmsnorm-swiglu.md    # Phase 2
├── 08-rope-gqa-weight-tying.md      # Phase 2
├── 09-mtp-distillation.md           # Phase 3
├── 10-flashattention-vllm.md        # Phase 4
├── 11-mamba-hybrids.md              # Phase 5
└── 12-fintech-considerations.md     # Phase 6

code/
├── 01-tensors/                      # ✓ DONE Phase 0
│   ├── 01_tensor_basics.py
│   └── 02_matrix_multiply.py
├── 02-neural-nets/                  # ✓ DONE Phase 0
│   ├── 01_linear_layer.py
│   ├── 02_mlp.py
│   └── 03_training_loop.py
├── 03-tokenization-embeddings-pe/   # Phase 1
│   ├── 01_char_tokenizer.py
│   └── 02_embeddings_and_pe.py
├── 04-attention/
│   ├── 01_simple_attention.py
│   └── 02_multi_head_causal.py
├── 05-ffn-layernorm/
│   ├── 01_ffn_layernorm.py
│   └── 02_vanilla_decoder_block.py
├── vanilla_transformer.py           # Phase 1 — shared module (block + full model)
├── 06-training-generation/
│   ├── 01_training_loop.py
│   └── 02_generation.py
├── 07-modern-upgrades/
│   ├── 01_prenorm.py
│   └── 02_swiglu.py
├── 08-rope-gqa/
│   ├── 01_rope.py
│   └── 02_gqa.py
├── 09-mtp-distillation/
│   ├── 01_mtp_heads.py
│   └── 02_distillation_pipeline.py
├── 10-inference-optimization/
│   └── 01_flashattention_demo.py
├── 11-mamba/
│   └── 01_hybrid_block.py
└── 12-fintech/
    └── (no code needed)

reference/
├── glossary.md
├── cheat-sheet.md
└── code-walkthrough.md
```

---

## Phase 0 Verification

All Phase 0 files are written and reconciled against the librarian review. Quick verification:

- [ ] Run `python3 code/01-tensors/01_tensor_basics.py` — should show tensor shapes from scalar to 3D
- [ ] Run `python3 code/01-tensors/02_matrix_multiply.py` — should show `(4,16)@(16,8)→(4,8)` and manual == PyTorch
- [ ] Run `python3 code/02-neural-nets/01_linear_layer.py` — should show manual `x@W.T+b` matches `linear(x)`
- [ ] Run `python3 code/02-neural-nets/02_mlp.py` — should show 172 and 1140 params
- [ ] Run `python3 code/02-neural-nets/03_training_loop.py` — loss should decrease, test accuracy >90% (synthetic data with clear pattern)

---

## Phase 0.5: Environment Setup

**Files:** None (system-level setup)

- [ ] **Step 1: Verify Python 3.12+**
  Run `python3 --version`. Expected: `Python 3.12.x` or higher.
- [ ] **Step 2: Install PyTorch**
  Run `pip install torch`. Verify with `python3 -c "import torch; print(torch.__version__)"`. Expected: `2.x.x`.
- [ ] **Step 3: Test Phase 0 code**
  Run `python3 code/02-neural-nets/03_training_loop.py` — should complete without import errors.

> **TinyShakespeare**: Downloaded automatically by the companion code. No manual download needed.

---

## Phase 1: Vanilla Transformer

Build the first working decoder-only transformer (2017-style: sinusoidal PE, post-norm LayerNorm, ReLU FFN, no RoPE). Train on TinyShakespeare.

### Task 1.1: Write `curriculum/03-tokenization-embeddings-pe.md` — ~3 hrs agent time

**Files:**
- Create: `curriculum/03-tokenization-embeddings-pe.md`
- Create: `code/03-tokenization-embeddings-pe/01_char_tokenizer.py`
- Create: `code/03-tokenization-embeddings-pe/02_embeddings_and_pe.py`

> **⏱ Time budget**: Step × are the main time sinks (writing + diagrams). Code steps are fast (@fixer dispatches). Self-review catches issues early.

- [ ] **Step 1: Write the "what is a token?" section**

One paragraph explaining that text must become numbers. Diagram: "Hello" → [H, e, l, l, o] → [8, 5, 12, 12, 15] (ASCII). Mention that LLMs use subwords (BPE), not characters, but we start simple.

- [ ] **Step 2: Write the character-level tokenizer section**

Explain the `stoi` (string-to-int) and `itos` (int-to-string) mapping. Show the TinyShakespeare vocab of 65 unique characters. Diagram: lookup table with 65 rows.

- [ ] **Step 3: Write the embedding section**

Explain that each token ID maps to a learned vector (embedding). Diagram: token ID 37 → look up row 37 in embedding matrix → get vector of length d_model. Explain the shape: `(vocab_size, d_model)`.

- [ ] **Step 4: Write the "why position matters" section**

Show: "dog bites man" vs "man bites dog" — same words, different meaning. Embeddings alone can't distinguish order. Diagram: three tokens with same embeddings vs with position signal added.

- [ ] **Step 5: Write the sinusoidal positional encoding section**

Show the formula: `PE(pos, 2i) = sin(pos/10000^(2i/d))` and `PE(pos, 2i+1) = cos(pos/10000^(2i/d))`. Explain what this means: each dimension gets a sine wave at a different frequency. Diagram: sine waves at low freq (pos 0-100) and high freq (pos 0-10). Show that position 0 gets the same encoding regardless of sequence length.

> 🔄 **Why start here**: Sinusoidal PE is the original 2017 design — fixed, deterministic, no learned parameters. It's easy to understand and debug. **What changes later**: Phase 2 (Chapter 08) replaces this with **RoPE**, which encodes *relative* position and handles long contexts better.

- [ ] **Step 6: Write the 🟢 summary box**

"Token IDs → embedding vectors → add position signal → ready for the transformer." Key terms: token ID, embedding, positional encoding, vocab size, d_model. Connect to Chapter 00's high-level flow.

- [ ] **Step 7: Write the 🔵 code section — char tokenizer in PyTorch**

Show: create `stoi`/`itos` from text, encode a string, decode it back. Print shapes. Point to companion code.

- [ ] **Step 8: Write companion code `code/03-tokenization-embeddings-pe/01_char_tokenizer.py`**

```python
import torch

# Character-level tokenizer for TinyShakespeare
# Maps each unique character to an integer, and back.

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

Run: `python3 code/03-tokenization-embeddings-pe/01_char_tokenizer.py`
Expected: text → encoded → decoded correctly, vocab size printed, tensor shape shown.

- [ ] **Step 9: Write companion code `code/03-tokenization-embeddings-pe/02_embeddings_and_pe.py`**

```python
import torch
import torch.nn as nn

vocab_size = 65
d_model = 16
seq_len = 8

# Token embedding: lookup table
embed = nn.Embedding(vocab_size, d_model)
dummy_tokens = torch.randint(0, vocab_size, (2, seq_len))  # (B=2, T=8)
embedded = embed(dummy_tokens)                                # (2, 8, 16)
print(f"Embedding shape: {embedded.shape}")

# Sinusoidal positional encoding
pos = torch.arange(seq_len).unsqueeze(1)                      # (8, 1)
div = 10000 ** (torch.arange(0, d_model, 2) / d_model)       # (8,) — one frequency per pair
pe = torch.zeros(1, seq_len, d_model)                         # (1, 8, 16)
pe[0, :, 0::2] = torch.sin(pos / div)                        # even dims: sin
pe[0, :, 1::2] = torch.cos(pos / div)                        # odd dims: cos
print(f"Positional encoding shape: {pe.shape}")
print(f"PE at position 0, first 4 dims: {pe[0, 0, :4].tolist()}")
print(f"PE at position 1, first 4 dims: {pe[0, 1, :4].tolist()}")

# Add to embeddings
x = embedded + pe                                              # (2, 8, 16)
print(f"Input to transformer: {x.shape}")

# Verify: positions 0 and 1 have different encodings
assert not torch.allclose(pe[0, 0], pe[0, 1]), "Positions should differ"
print("✓ Different positions get different encodings")
```

Run: `python3 code/03-tokenization-embeddings-pe/02_embeddings_and_pe.py`
Expected: shapes printed, PE values differ per position, assertion passes.

- [ ] **Step 10: Write the 🟢 Check Your Understanding section**

Add 3–4 quick questions for the conceptual track:
1. "What's the difference between a token ID and an embedding vector?"
2. "Why can't we use token IDs directly as model input?"
3. "What would happen if we didn't add positional encoding?"
4. "Why does position 0 always get the same encoding regardless of sequence length?"

Provide answers in a collapsible `<details>` block for self-checking. This validates that non-engineers actually absorbed the core concepts before moving to the next chapter.

- [ ] **Step 11: Read back the full chapter and verify**

- Does a non-engineer understand what a token is? Test: read the first 3 paragraphs to a colleague.
- Does the PE formula explanation avoid math overload? (The intuitive part is "different frequencies for different positions")
- Do the diagrams render correctly?
- Do both companion scripts run cleanly?
- Are the CYU questions clear and answerable from the chapter text alone?

---

### Task 1.2: Write `curriculum/04-attention-mechanism.md` — ~4 hrs agent time

**Files:**
- Create: `curriculum/04-attention-mechanism.md`
- Create: `code/04-attention/01_simple_attention.py`
- Create: `code/04-attention/02_multi_head_causal.py`

> **⚠️ Hardest chapter.** Attention is the conceptual peak. Budget extra time for the QKV analogy and concrete number walkthrough. Non-engineers will need the most support here.

- [ ] **Step 1: Write the "why attention" problem section**

Classic example: "The cat chased the mouse because it was hungry." What does "it" refer to? A simple model processing word-by-word can't connect "it" → "cat" across distance. Diagram: arrows showing the gap between "cat" and "it".

- [ ] **Step 2: Write the database analogy for Q, K, V**

Q = Query: "what am I looking for?" K = Key: "what does each token contain?" V = Value: "what information do I return?"
Analogy: You're in a library with a question (Q). Each book has a topic label (K). You find books whose topics match your question. You take home their content (V). Diagram: search query matching against keys.

- [ ] **Step 3: Write the attention formula step-by-step**

Show: `scores = Q @ K^T`, `weights = softmax(scores / sqrt(d_k))`, `output = weights @ V`. Do this with concrete numbers: 3 tokens, d_model=4. Walk through the shapes:
- Q, K, V: each (3, 4)
- scores: (3, 3) — each token's Q against every K
- Divide by sqrt(4)=2 to prevent softmax saturation
- softmax: (3, 3) — weights summing to 1 per row
- output: (3, 4) — weighted sum of values

- [ ] **Step 4: Write multi-head attention section**

One attention isn't enough — tokens need to attend to others for different reasons (syntax, semantics, etc.). Multi-head runs H parallel attentions, each with different learned projections. Diagram: input → split into 4 heads → 4 attention matrices → concatenate → project to output. Show shape: `(B, T, C) → reshape (B, T, H, d_head) → transpose (B, H, T, d_head) → attention → transpose back → reshape (B, T, C)`.

> 🔄 **Why start here**: This is the original 2017 multi-head design with separate QKV projections per head — simple and pedagogical. **What changes later**: Phase 2 (Chapter 08) upgrades to **GQA** (Grouped Query Attention), which shares KV heads across query groups to reduce memory.

- [ ] **Step 5: Write the causal mask section**

An autoregressive model cannot see future tokens during training. The mask zeros out attention to future positions. Diagram: upper triangular matrix filled with -inf before softmax, showing token 0 can only see itself, token 1 can see tokens 0-1, etc.

- [ ] **Step 6: Write the 🟢 summary box**

"Attention finds relationships between tokens. Multi-head runs this in parallel for different types of relationships. The causal mask prevents cheating by looking at future tokens." Key terms: QKV, attention scores, softmax, multi-head, causal mask.

- [ ] **Step 7: Write the 🟢 Check Your Understanding section**

Add 3–4 quick questions:
1. "In the library analogy, what do Q, K, and V represent?"
2. "Why do we divide attention scores by sqrt(d_k)?"
3. "What does the causal mask prevent?"
4. "Why do we need multiple attention heads instead of just one?"

Provide answers in a collapsible `<details>` block.

- [ ] **Step 8: Write companion code `code/04-attention/01_simple_attention.py`**

```python
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
```

Run: `python3 code/04-attention/01_simple_attention.py`
Expected: all shapes printed, weights[0] row sums to ~1.0.

- [ ] **Step 9: Write companion code `code/04-attention/02_multi_head_causal.py`**

```python
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
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))

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
```

Run: `python3 code/04-attention/02_multi_head_causal.py`
Expected: shapes match, "✓" printed.

- [ ] **Step 10: Self-review the chapter**

- Does the QKV database analogy hold up? (Free-text query matching library catalog — the standard teaching analogy)
- Are the concrete numbers in Step 3 easy to follow?
- Does the causal mask diagram clearly show the upper triangle?
- Are the CYU questions clear and answerable from the 🟢 sections alone?

---

### Task 1.3: Write `curriculum/05-ffn-layernorm-stack.md` — ~3 hrs agent time

**Files:**
- Create: `curriculum/05-ffn-layernorm-stack.md`
- Create: `code/05-ffn-layernorm/01_ffn_layernorm.py`
- Create: `code/05-ffn-layernorm/02_vanilla_decoder_block.py`

> **📐 Architecture chapter**. This is where the pieces click together into a real transformer block. The post-norm vs pre-norm distinction matters deeply here.

- [ ] **Step 1: Write the FFN section**

Attention mixes information TOKENS across positions. FFN processes each TOKEN'S information independently. Architecture: `linear(d_model → d_ff) → ReLU → linear(d_ff → d_model)`. Standard: d_ff = 4 × d_model. Diagram: one token's vector flowing through expansion → activation → compression. Analogy: attention is "team meeting" (tokens share info), FFN is "individual work time" (each token processes alone).

- [ ] **Step 2: Write the LayerNorm section**

LayerNorm stabilizes training: `y = (x - mean) / std * gamma + beta`. Why it helps: prevents activations from growing too large or too small as they pass through layers. Diagram: a distribution shifting from scattered → centered. No bias for LayerNorm — just scale (gamma) and shift (beta).

- [ ] **Step 3: Write the residual connections section**

Problem: deep networks (6+ layers) suffer from vanishing gradients — early layers barely learn. Residual connection: `output = x + sublayer(x)`. Info can skip layers entirely. Diagram: a "skip highway" around attention and FFN. Show that the gradient can flow directly back through the skip. Analogy: express lane on a highway — you don't have to go through every exit.

- [ ] **Step 4: Write the post-norm block assembly section**

Original 2017 layout (post-norm):
```
x = LayerNorm(x + MultiHeadAttention(x))
x = LayerNorm(x + FFN(x))
```
Diagram: full block with Attention → Add → Norm → FFN → Add → Norm, with residual arrows going around each sublayer.

> 🔄 **Why start here**: Post-norm (LayerNorm *after* the residual addition) is the original 2017 layout. It works well for shallow stacks (4–6 layers). **What changes later**: Phase 2 (Chapter 07) switches to **pre-norm** (LayerNorm *before* the sublayer), which enables stable training for 100+ layer stacks.

- [ ] **Step 5: Write the stacking section**

Why stack multiple blocks: each layer builds more abstract representations. Layer 1 learns word-level patterns. Layer 3 learns phrase patterns. Layer 6 learns sentence-level meaning. Diagram: stack of N blocks with input at bottom and output at top.

- [ ] **Step 6: Write the 🟢 summary box**

"Each block does two things — mix information across tokens (attention) then process each token individually (FFN). Residual connections and LayerNorm make deep stacks trainable." Key terms: FFN, LayerNorm, residual connection, block, post-norm.

- [ ] **Step 7: Write the 🟢 Check Your Understanding section**

Add 3–4 quick questions:
1. "What's the difference between attention and FFN? (Use the team meeting vs individual work analogy.)"
2. "Why do we need LayerNorm?"
3. "What would happen if we stacked 20 blocks without residual connections?"
4. "Which comes first in post-norm: LayerNorm or the sublayer?"

Provide answers in a collapsible `<details>` block.

- [ ] **Step 8: Write companion code `code/05-ffn-layernorm/01_ffn_layernorm.py`**

```python
import torch
import torch.nn as nn

B, T, C = 2, 8, 16

# FFN: ReLU activation (modern models use SwiGLU — we'll upgrade later)
ffn = nn.Sequential(
    nn.Linear(C, C * 4),    # expand 4x: 16 → 64
    nn.ReLU(),
    nn.Linear(C * 4, C),    # project back: 64 → 16
)
x = torch.randn(B, T, C)
out_ffn = ffn(x)
print(f"FFN: {x.shape} → {out_ffn.shape}")
print(f"  Hidden dimension: {C * 4}")

# LayerNorm
ln = nn.LayerNorm(C)
out_ln = ln(x)
print(f"LayerNorm: {out_ln.shape}")
print(f"  Mean: {out_ln.mean().item():.4f}, Std: {out_ln.std().item():.4f}")
print(f"  (should be ~0.0 and ~1.0)")

# Residual connection
skip = x + out_ffn          # gradient flows through both paths
print(f"Residual x + FFN(x): {skip.shape}")
```

Run: `python3 code/05-ffn-layernorm/01_ffn_layernorm.py`
Expected: shapes printed, LayerNorm output near mean=0, std=1.

- [ ] **Step 9: Write companion code `code/05-ffn-layernorm/02_vanilla_decoder_block.py`**

```python
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')
# Note: this requires the class from 04-attention/02_multi_head_causal.py
# Copy the CausalMultiHeadAttention class here for self-containedness, or import it.

class CausalMultiHeadAttention(nn.Module):
    """Self-contained copy so this file runs standalone."""
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
        weights = torch.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)

class VanillaDecoderBlock(nn.Module):
    """One transformer decoder block with post-norm (2017 style)."""
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

block = VanillaDecoderBlock(d_model=16, n_heads=4)
x = torch.randn(1, 8, 16)
out = block(x)
print(f"Vanilla block: {x.shape} → {out.shape}")
params = sum(p.numel() for p in block.parameters())
print(f"Parameters: {params:,}")
```

Run: `python3 code/05-ffn-layernorm/02_vanilla_decoder_block.py`
Expected: shapes match, param count printed (~4,900 for this small config).

- [ ] **Step 10: Self-review**

- Does the "team meeting vs individual work" analogy for attention vs FFN hold?
- Is the post-norm layout clear? (The word "post" means after the sublayer)
- Does the companion code run without errors?

---

### Task 1.4: Write `curriculum/06-training-loop-and-generation.md` — ~3 hrs agent time

**Files:**
- Create: `curriculum/06-training-loop-and-generation.md`
- Create: `code/vanilla_transformer.py`
- Create: `code/06-training-generation/01_training_loop.py`
- Create: `code/06-training-generation/02_generation.py`

> **🏗️ Integration chapter**. Biggest file count. The shared module (`code/vanilla_transformer.py`) is critical — it must be importable by the other scripts.

- [ ] **Step 1: Write the complete model assembly section**

Show how all pieces fit together:
```
Token Embedding → + Sinusoidal PE → [Block × N] → LayerNorm → Linear Head → logits
```
The head has shape `(d_model, vocab_size)` — it projects each token's final representation back to vocabulary space. The softmax turns logits into probabilities. Diagram: full model as a pipeline from input IDs to output probabilities.

- [ ] **Step 2: Write the weight initialization section**

Why initialization matters: too large → neurons saturate, too small → gradients vanish. Standard approach: `nn.init.normal_(weight, mean=0, std=0.02)`. The magic number 0.02 works for most configurations.

- [ ] **Step 3: Write the loss computation section**

Cross-entropy: compare predicted probability distribution (output of softmax) against the true one-hot distribution (the correct next token). For example, if the correct next token ID is 37, we want the model to predict high probability for token 37. Show that we compare logits (before softmax) directly using `F.cross_entropy`, which internally applies log-softmax.

- [ ] **Step 4: Write the training loop section**

The loop from Phase 0, now applied to the transformer:
1. Sample random batch of sequences from TinyShakespeare
2. Forward pass: `logits, loss = model(x, targets=y)`
3. Backward: `loss.backward()`
4. Clip gradients: `clip_grad_norm_(params, 1.0)` (prevents exploding gradients)
5. Update: `optimizer.step()`
6. Repeat thousands of times

- [ ] **Step 5: Write the generation section**

Autoregressive generation:
1. Start with a prompt (e.g., "ROMEO:")
2. Forward pass to get logits for the next token
3. Apply temperature: `logits = logits / temperature`
4. Optional top-k: zero out all but the k highest logits
5. Sample: `next_token = multinomial(softmax(logits))`
6. Append to sequence and repeat

Diagram: the generation loop with decision points (temperature, top-k).

> 🔄 **Why temperature matters**: Temperature controls the randomness of generation. Low values (0.1) make the model deterministic and repetitive. High values (1.5) make it creative but risk gibberish. **What changes later**: Production systems (Phase 4) add sampling strategies like **top-p** (nucleus) and **mixtral** of experts for better quality.

- [ ] **Step 6: Write the 🟢 summary box**

"All the pieces connect into a single model that predicts the next token. During training it learns from data; during generation it uses what it learned to produce new text." Key terms: logits, cross-entropy, temperature, top-k, sampling, checkpoint.

- [ ] **Step 7: Write the 🟢 Check Your Understanding section**

Add 3–4 quick questions:
1. "What's the difference between training and inference (generation)?"
2. "What does temperature=0.1 do vs temperature=1.5?"
3. "Why do we save checkpoints during training?"
4. "What's the shape of the final linear head layer?"

Provide answers in a collapsible `<details>` block.

- [ ] **Step 8: Write companion code `code/vanilla_transformer.py` (shared module)**

This file becomes the shared module imported by the training loop and generation scripts. It includes a self-contained copy of `CausalMultiHeadAttention` and `VanillaDecoderBlock` so it runs standalone.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ── Self-contained attention (same as Task 1.2) ──────────────────
class CausalMultiHeadAttention(nn.Module):
    """Multi-head causal self-attention. No RoPE yet — pure 2017 style."""
    def __init__(self, d_model=16, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)

# ── Vanilla decoder block (same as Task 1.3) ─────────────────────
class VanillaDecoderBlock(nn.Module):
    """One transformer decoder block with post-norm (2017 style)."""
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
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
        return x

# ── Full transformer model ──────────────────────────────────────
class VanillaDecoderOnlyTransformer(nn.Module):
    """Full decoder-only transformer: embeddings → N blocks → head."""
    def __init__(self, vocab_size=65, d_model=128, n_heads=4, n_layers=4, max_seq_len=256):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)

        pe = torch.zeros(1, max_seq_len, d_model)
        pos = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)
        div = 10000 ** (torch.arange(0, d_model, 2, dtype=torch.float) / d_model)
        pe[0, :, 0::2] = torch.sin(pos / div)
        pe[0, :, 1::2] = torch.cos(pos / div)
        self.register_buffer('pe', pe, persistent=False)

        self.blocks = nn.ModuleList([
            VanillaDecoderBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embed(idx) + self.pe[:, :T, :]
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_final(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# Verify shapes
model = VanillaDecoderOnlyTransformer()
x = torch.randint(0, 65, (2, 32))
logits, _ = model(x)
print(f"Logits shape: {logits.shape}  (expected: [2, 32, 65])")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

Run: `python3 code/vanilla_transformer.py`
Expected: logits shape [2, 32, 65], param count printed (~3-5M for this config).

- [ ] **Step 9: Write companion code `code/06-training-generation/01_training_loop.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from pathlib import Path

# Import the model from step 7
from code.vanilla_transformer import VanillaDecoderOnlyTransformer

# Load TinyShakespeare
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
data_path = Path("data/tinyshakespeare.txt")
data_path.parent.mkdir(exist_ok=True)
if not data_path.exists():
    import urllib.request
    print("Downloading TinyShakespeare...")
    try:
        urllib.request.urlretrieve(DATA_URL, data_path, timeout=30)
    except Exception as e:
        print(f"Download failed: {e}")
        print("Create data/tinyshakespeare.txt manually with any text to train on.")
        exit(1)

text = data_path.read_text()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

# Split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Batch sampling
def get_batch(data, batch_size=16, block_size=128):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# Model, optimizer
model = VanillaDecoderOnlyTransformer(vocab_size=vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Training
batch_size = 16
block_size = 128
eval_interval = 100
eval_iters = 50
best_loss = float('inf')

print(f"Training {sum(p.numel() for p in model.parameters()):,}-param model on {len(data):,} tokens...")
for step in range(1000):
    # Evaluate
    if step % eval_interval == 0:
        model.eval()
        losses = {}
        for split_name, split_data in [('train', train_data), ('val', val_data)]:
            losses_split = torch.zeros(eval_iters)
            for k in range(eval_iters):
                x, y = get_batch(split_data, batch_size, block_size)
                with torch.no_grad():
                    _, loss = model(x, y)
                losses_split[k] = loss.item()
            losses[split_name] = losses_split.mean().item()

        print(f"Step {step:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
        if losses['val'] < best_loss:
            best_loss = losses['val']
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/vanilla_best.pt")
            print(f"  → saved best model (val loss {best_loss:.4f})")
        model.train()

    # Training step
    x, y = get_batch(train_data, batch_size, block_size)
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

print(f"\nDone. Best val loss: {best_loss:.4f}")
```

Run: `python3 code/06-training-generation/01_training_loop.py`
Expected: loss printed every 100 steps, decreasing over time. Best model checkpoint saved. (Takes ~5-10 min on CPU for 1000 steps.)

- [ ] **Step 10: Write companion code `code/06-training-generation/02_generation.py`**

```python
import torch
import torch.nn.functional as F
from pathlib import Path

def generate(model, prompt_ids, max_new_tokens=200, temperature=1.0, top_k=40):
    """Autoregressive generation with temperature and top-k sampling."""
    model.eval()
    idx = prompt_ids
    for _ in range(max_new_tokens):
        # Crop to context window
        idx_cond = idx[:, -model.max_seq_len:]

        # Forward pass
        with torch.no_grad():
            logits, _ = model(idx_cond)

        # Focus on the last token's logits
        logits = logits[:, -1, :] / temperature

        # Top-k: zero out everything outside the top k
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = float('-inf')

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append
        idx = torch.cat([idx, next_token], dim=1)

    return idx

# Demo
if __name__ == "__main__":
    # Load checkpoint
    from code.vanilla_transformer import VanillaDecoderOnlyTransformer

    # Use the same vocab as training
    text = open("data/tinyshakespeare.txt").read()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)

    model = VanillaDecoderOnlyTransformer(vocab_size=vocab_size)
    ckpt_path = Path("checkpoints/vanilla_best.pt")
    if not ckpt_path.exists():
        print("No checkpoint found. Run the training loop first to create checkpoints/vanilla_best.pt.")
        exit(1)
    state = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(state)

    # Generate
    prompt = "ROMEO:"
    prompt_ids = torch.tensor([[stoi[ch] for ch in prompt]], dtype=torch.long)
    output_ids = generate(model, prompt_ids, max_new_tokens=100, temperature=0.8, top_k=40)
    output_text = "".join(itos[i.item()] for i in output_ids[0])

    print(f"Prompt: {prompt}")
    print("-" * 50)
    print(output_text)
    print("-" * 50)
    print(f"\nGenerated {len(output_ids[0]) - len(prompt_ids[0])} tokens")
```

Run: `python3 code/06-training-generation/02_generation.py`
Expected: generated Shakespeare-like text (garbled but showing word-like patterns since the model is tiny and under-trained).

- [ ] **Step 11: Self-review**

- Does the full model diagram clearly show all components in one flow?
- Is the distinction between training mode (loss computation) and eval mode (generation) clear?
- Does generation work end-to-end?

---

## Phase 2–6: Outlined with Task Headers

> **Pedagogical note**: Everything you built in Phase 1 is correct — it's the 2017 Transformer that launched the LLM revolution. Phase 2 upgrades each component with improvements discovered since, one at a time. Every change has a clear reason. You're not fixing bugs; you're evolving your architecture.

### Task 2.1: Pre-norm, RMSNorm, SwiGLU chapter

Write `curriculum/07-pre-norm-rmsnorm-swiglu.md` and companion code.

- [ ] Step: Write "problem with post-norm" — gradient explosion in deep stacks, use diagram
- [ ] Step: Write pre-norm section — `x = x + sublayer(norm(x))`, the fix and why it works
- [ ] Step: Write LayerNorm → RMSNorm — remove mean computation, faster, equally good
- [ ] Step: Write ReLU → GELU → SwiGLU — smoother gradient, better quality, SwiGLU has 3 matrices
- [ ] Step: Write companion code — upgrade the vanilla block step-by-step
- [ ] Step: Verify companion code runs

### Task 2.2: RoPE, GQA, weight tying chapter

Write `curriculum/08-rope-gqa-weight-tying.md` and companion code.

- [ ] Step: Write "problem with sinusoidal PE" — absolute position, can't handle long contexts well
- [ ] Step: Write RoPE section — rotating Q/K vectors, relative position emerges naturally
- [ ] Step: Write RoPE rotation formula with diagram showing vector rotation at different positions
- [ ] Step: Write GQA section — sharing KV heads reduces cache size
- [ ] Step: Write weight tying — embedding and head share weights
- [ ] Step: Write companion code — RoPE from scratch, GQA comparision
- [ ] Step: Verify companion code runs

### Task 3.1: MTP and Distillation chapter

Write `curriculum/09-mtp-distillation.md` and companion code.

- [ ] Step: Write MTP section — predicting N future tokens instead of 1
- [ ] Step: Write MTP architecture — extra prediction heads + auxiliary loss
- [ ] Step: Write distillation overview — teacher → student
- [ ] Step: Write data generation section — prompting teacher, filtering
- [ ] Step: Write companion code — MTP heads
- [ ] Step: Write companion code — simplified distillation loop sketch
- [ ] Step: Verify companion code runs

### Task 4.1: FlashAttention, vLLM, production chapter

Write `curriculum/10-flashattention-vllm.md` and companion code.

- [ ] Step: Write FlashAttention section — tiling the computation, reduces memory bandwidth
- [ ] Step: Write KV Cache section — what it is, why it matters for inference
- [ ] Step: Write PagedAttention section — KV cache as virtual memory
- [ ] Step: Write vLLM section — serving engine that bundles these
- [ ] Step: Write quantization section — FP16 → INT8/INT4
- [ ] Step: Write companion code — conceptual Python sketches showing tiling, recomputation, and IO-awareness (not real GPU FlashAttention, but the algorithmic ideas)

- [ ] Step: Verify companion code runs (CPU-only, uses small matrices to demonstrate tiling concepts)

### Task 5.1: Mamba, SSMs, Hybrids chapter

Write `curriculum/11-mamba-hybrids.md` and companion code.

- [ ] Step: Write "why alternatives" — quadratic attention vs linear scaling
- [ ] Step: Write SSM section — signal processing → recurrent computation
- [ ] Step: Write Mamba section — selective SSM, content-aware memory
- [ ] Step: Write comparison to RWKV, RetNet, xLSTM
- [ ] Step: Write hybrid architecture section — best of both worlds
- [ ] Step: Write companion code — import mamba-ssm, show hybrid block
- [ ] Step: Verify companion code runs (may need mamba-ssm installed; ⚠️ `mamba-ssm` is Linux-only and requires CUDA — on Mac/Windows this step is documentation-only)

### Task 6.1: Fintech Considerations chapter

Write `curriculum/12-fintech-considerations.md` (no code, conceptual chapter).

- [ ] Step: Write evaluation section — metrics for compliance/risk
- [ ] Step: Write PII/Privacy section — GDPR, SOX, finreg implications
- [ ] Step: Write guardrails section — structured output, hallucination prevention
- [ ] Step: Write RAG section — most fintech apps use retrieval, not raw generation
- [ ] Step: Write cost estimation section — token math → dollar math
- [ ] Step: Write build vs buy vs fine-tune decision framework

---

## Reference Docs

### Task R.1: Glossary

- [ ] Compile term list from all chapters (attention through weight tying)
- [ ] Write each definition: one sentence, non-technical, link to first chapter
- [ ] Cross-reference related terms

### Task R.2: Cheat Sheet

- [ ] Write shape reference: `(B, T, C)` and derived shapes
- [ ] Write formula reference: attention, softmax, cross-entropy, LayerNorm, RoPE rotation
- [ ] Write hyperparameter reference: typical values for tiny → large models
- [ ] Write upgrade table: vanilla vs modern, what changed and why

### Task R.3: Code Walkthrough

- [ ] Annotate final `code/vanilla_transformer.py` line by line
- [ ] Show entry points for non-engineers ("the attention section is the most important 40 lines")
- [ ] Show how each chapter maps to a section of the code

---

## Self-Review

- [ ] **Spec coverage:** All topics from the Grok conversation (distillation, CoT compute cost, FlashAttention, Mamba, MTP) are covered in phases 3-5. The oracle's fintech recommendation is Phase 6.
- [ ] **Placeholder scan:** No "TBD", "TODO", "implement later" patterns. Every step has an action verb and expected output. Companion code blocks have actual Python code.
- [ ] **Type consistency:** `CausalMultiHeadAttention` is defined in Task 1.2 and reused in Task 1.3's `VanillaDecoderBlock`. `VanillaDecoderBlock` is reused in `code/vanilla_transformer.py` (shared module, Task 1.4) by `VanillaDecoderOnlyTransformer`. All class names and signatures match.
- [ ] **Granularity check:** Every step is 2-5 minutes of focused work. No step says "write the chapter" — each chapter is broken into writing sections, diagrams, code files, and verification.
