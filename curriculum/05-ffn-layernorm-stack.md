# Chapter 05: The FFN, LayerNorm & The Complete Block

> **Audience**: 🟢 All roles (code sections marked 🔵)
> **Prerequisites**: Chapter 04 (Attention Mechanism), basic Python
> **Estimated time**: 20 minutes read, 15 minutes code

---

## Why This Matters

You now understand **attention** — the mechanism that lets tokens share information across positions. But a transformer isn't just attention. It's a carefully designed **block** that alternates between two complementary operations:

1. **Attention** — the "team meeting" where tokens share information
2. **FFN (Feed-Forward Network)** — the "individual work time" where each token processes alone

These two operations are wrapped with **LayerNorm** (for training stability) and **residual connections** (for gradient flow), then **stacked** 4, 8, 12, or even 100+ layers deep.

By the end of this chapter, you'll understand:
- **What** the FFN does and why attention alone isn't enough
- **How** LayerNorm keeps activations in a healthy range
- **Why** residual connections solve the vanishing gradient problem
- **How** all pieces fit into a complete decoder block
- **What** happens when you stack blocks

---

## Part 1: The Feed-Forward Network (FFN)

### Why Attention Alone Isn't Enough

Attention is powerful — it lets every token look at every other token. But attention is fundamentally a **weighted averaging** operation. The output of attention at position *i* is a mixture of other tokens' values:

```
output_i = sum_j (weight_ij × value_j)
```

This is linear! If you stack attention-only layers, you just get a deeper linear mixture — no matter how many layers, you can't learn complex patterns. The model needs **non-linear processing** at each position.

### What the FFN Does

The FFN processes **each token independently** with the same learned transformation. It's a simple two-layer neural network:

```
Input (d_model)  →  expand 4×  →  ReLU  →  compress → Output (d_model)
   [1×16]             [1×64]       [1×64]             [1×16]
```

The standard configuration:
- **Expand**: project from `d_model` → `4 × d_model` (the hidden dimension, called `d_ff`)
- **Activate**: apply ReLU (`max(0, x)`)
- **Compress**: project back from `4 × d_model` → `d_model`

The expansion gives the model **more capacity to learn patterns** at each position. The ReLU introduces **non-linearity** — without it, the entire block would be linear and stacking wouldn't help.

```
┌─────────────────────────────────────────────────────────────────┐
│                        FFN Architecture                         │
│                                                                  │
│   token vector     expand 4×       ReLU        compress back    │
│   [1, 0, 0, 0] ──→ [1,2,0,0,4,…] ──→ [1,2,0,0,4,…] ──→ [2,0,1]│
│   (d_model=4)        (d_ff=16)      (d_ff=16)        (d_model=4)│
│                                                                  │
│   Each token goes through the SAME FFN independently.            │
└─────────────────────────────────────────────────────────────────┘
```

> 🧠 **Common beginner mistakes (FFN shapes)**:
>
> 1. **Mismatched dimensions**: The first Linear goes from `d_model → d_ff` (usually `4 × d_model`), and the second goes from `d_ff → d_model`. Forgetting the 4× factor is the most common bug — it silently produces a smaller network that may still train but has less capacity.
>
> 2. **ReLU dead neurons**: ReLU outputs 0 for any negative input — those neurons "die" and stop learning. This is called the "dying ReLU" problem. Modern models mitigate this with GELU or SwiGLU (you'll see these in Phase 2). Our tiny models in Phase 1 are big enough that a few dead neurons don't hurt.

### Attention vs FFN: Team Meeting vs Individual Work

This is the most important mental model for understanding transformers:

| Operation | Analogy | What It Does |
|-----------|---------|--------------|
| **Attention** | Team meeting | Tokens share information across positions |
| **FFN** | Individual work time | Each token processes its blended information alone |

**Why both are needed:**
- Attend to find relevant information (attention)
- Then process what you found (FFN)
- Then attend again with your new understanding (next attention layer)
- Repeat, building up increasingly sophisticated representations

> 💡 **Real-world analogy**: Think of a team working on a project. They meet to share ideas (attention), then go off to work individually (FFN), then meet again with their progress, then work more, and so on. Each cycle deepens everyone's understanding.

### 🕰️ How FFN Activations Evolved

The activation function inside the FFN has been steadily upgraded as researchers found better alternatives to ReLU:

| Year | Activation | Formula | Used By | Why |
|------|-----------|---------|---------|-----|
| **2017** | **ReLU** | `max(0, x)` | Original Transformer (Vaswani et al.) | Simple, fast, works — but dies for negative inputs |
| **2018** | **GELU** | `x × Φ(x)` (smooth sigmoid-like curve) | GPT-2, BERT (Devlin, Brown et al.) | Smooth, non-zero for negatives, dropped ~10% of weights vs ReLU's hard zero |
| **2023** | **SwiGLU** | `swish(xW) ⊗ (xV)`, 3 weight matrices | LLaMA, PaLM, Mistral, Gemini | Significantly better quality; adds an extra weight matrix (still 4× total with gating trick) |
| **2024** | **ReLU²** | `max(0, x)²` | Nemotron-4, modded-nanogpt speedrun | Quadratic instead of linear — dead-simple but empirically stronger than GELU |

The **4× expansion ratio** (`d_ff = 4 × d_model`) has remained remarkably stable since 2017. Even SwiGLU — which mathematically uses 3 matrices (not 2) — achieves 4× effective width through dimension tricks.

> ⚡ **ReLU is our Phase 1 choice**: simplest, easiest to debug, and for a tiny model the difference is negligible. The upgrade path is clear: ReLU → GELU → SwiGLU → ReLU², each trading simplicity for training quality.

---

## Part 2: LayerNorm

### The Problem: Activations Drift

As data flows through layers, the mean and variance of activations can drift. Some dimensions grow very large, others shrink to near zero. This makes optimization difficult — the next layer has to deal with inputs that are constantly changing distribution.

> 📏 **Temperature control analogy**: Think of your neural network as a chemical reaction. LayerNorm is like a **temperature control system** — it keeps the reaction in the ideal temperature range regardless of what's happening outside. Too hot (activations too large) → saturation. Too cold (too small) → gradients vanish. LayerNorm keeps it "just right."

Researchers now understand this as keeping activations in the **linear regime of the activation function** — not too hot (saturated, where gradients are near zero) or too cold (dead, where neurons never fire).

### The Formula

LayerNorm normalizes each token's vector independently:

```
LayerNorm(x) = γ × (x - μ) / σ + β

where:
μ = mean of all features in x        ← center
σ = std of all features in x         ← scale
γ = learned scale parameter          ← can amplify or reduce
β = learned shift parameter          ← can offset
```

**Key properties:**
- **No bias in the computation itself** — LayerNorm's formula doesn't have a bias term. The shift comes from `β`.
- **Per-token, not per-batch**: Unlike BatchNorm, LayerNorm normalizes across features *within a single token*. This is crucial for transformers where sequence length varies.
- **Learned parameters**: `γ` (scale) and `β` (shift) are learned during training, allowing the model to undo the normalization if needed.

```
┌──────────────────────────────────────────────────────────────┐
│                    LayerNorm Visualized                       │
│                                                               │
│  Before:       After:                                         │
│  scattered     centered at 0, scaled to 1                     │
│                                                               │
│    │                           ║                               │
│   ║│║                      ║║║║║                              │
│  ║║│║║                   ║║║║║║║║                             │
│ ║║║│║║║                ║║║║║║║║║║                             │
│ ══╪═══→               ══╪═════→                               │
│   │                       │                                     │
│  mean=2, std=3          mean=0, std=1                          │
└──────────────────────────────────────────────────────────────┘
```

> 🧠 **Common beginner mistakes (LayerNorm)**:
>
> 1. **Confusing with BatchNorm**: LayerNorm normalizes **across features** for each token independently. BatchNorm normalizes **across the batch** for each feature independently. In transformers, LayerNorm is the correct choice because it works with variable-length sequences and doesn't depend on batch statistics.
>
> 2. **Forgetting γ and β are learned**: They start as all-1s and all-0s but change during training. If you see LayerNorm output that's not exactly mean=0, std=1, that's correct — the model has learned to shift/scale.
>
> 3. **Applying across the wrong dimension**: LayerNorm in PyTorch defaults to normalizing over the last dimension. For `(B, T, C)`, it normalizes each of the `B × T` vectors independently over the `C` features. If your tensor has a different layout, you may need `nn.LayerNorm(normalized_shape)` with a different `normalized_shape`.

---

## Part 3: Residual Connections

### The Problem: Vanishing Gradients

Deep networks suffer from **vanishing gradients**. When you have 6, 12, or 100 layers stacked, the gradient has to flow backward through all of them. Each layer's gradient gets multiplied by the layer's weight derivatives — if those are less than 1, the gradient shrinks exponentially. Early layers learn very slowly or not at all.

```
Without residual connections:
    loss → layer 6 → layer 5 → ... → layer 1
    grad ∝ (derivative)^6    ← vanishing!

With residual connections:
    loss → layer 6 ──→ layer 5 ──→ ... ──→ layer 1
          ↘___add___↘___add___↘___add___↘
    gradient flows through the direct path! ← strong!
```

### The Solution: Shortcut Path

A residual connection is simple: **add the input to the sublayer's output**.

```
output = x + sublayer(x)    ← "skip connection" or "shortcut"
```

The gradient can now flow back through **either** the sublayer **or** the direct connection (the identity path). The identity path passes gradients through unchanged, allowing deep stacks to train.

> 🛣️ **Express lane on a highway analogy**: Think of a multilane highway with many exits (layers). Without residual connections, you must take *every* exit, go through the local streets (calculating gradients), and get back on the highway — slow and you might get lost. Residual connections are like an **express lane** that bypasses the exits entirely. Information (and gradients) can flow directly from input to output without going through any sublayer.

### Why It Works So Well

The residual connection creates a **gradient superhighway**:

```
loss → ... → x + FFN(LayerNorm(x)) → x → ... → input
                  ↑                         ↑
          gradient through FFN      gradient through identity
          (can be small)            (always strong!)
```

Even if the FFN or attention has vanishing gradients, the identity path always passes full gradients through. This means early layers receive meaningful updates from the very first training step.

---

## Part 4: Post-Norm Block Assembly

Now let's put all the pieces together. The original 2017 transformer uses **post-norm** — LayerNorm applied *after* the residual addition:

```
                    ┌──────────────────────────────────┐
                    │        Decoder Block              │
                    │                                   │
     x ────────────┬┤───────────────────────────┐       │
                   ││                           │       │
                   │▼                           │       │
                   │Multi-Head Attention         │       │
                   │                            │       │
                   │◄──────── + ────────────────┘       │
                   │         (residual)                 │
                   │▼                                   │
                   │LayerNorm                           │
                   │                                    │
                   │┌──────────────────────────┐        │
                   ││                           │       │
                   │▼                           │       │
                   │FFN (expand → ReLU → compress)      │
                   │                            │       │
                   │◄─────── + ─────────────────┘       │
                   │        (residual)                  │
                   │▼                                   │
                   │LayerNorm                           │
                   │                                    │
                   └───────→ output ────────────────────┘
```

The code is remarkably compact:

```python
def forward(self, x):
    x = self.norm1(x + self.attn(x))   # post-norm: LayerNorm AFTER add
    x = self.norm2(x + self.ffn(x))    # post-norm: LayerNorm AFTER add
    return x
```

### 🕰️ How the Block Architecture Evolved

Where you place LayerNorm relative to the attention/FFN sublayer has a dramatic effect on training stability — especially as models grow deeper.

| Year | Layout | Formula | Max Stable Depth | Used By | Why |
|------|--------|---------|-----------------|---------|-----|
| **2017** | **Post-norm** | `LN(x + sublayer(x))` | ~6–12 layers | Original Transformer (Vaswani et al.) | First design — simple, works for shallow stacks |
| **2018** | **Pre-norm** | `x + sublayer(LN(x))` | ~12–48 layers | GPT-2, BERT (OpenAI, Google) | Gradient flows through residual path *before* LayerNorm scales it; stable for deeper stacks |
| **2020** | **Pre-norm + residual scaling** | `x + (1/√(2N)) × sublayer(LN(x))` | ~96 layers | GPT-3 | Large models needed extra damping: each residual add is scaled by layer count |
| **2025** | **Pre-norm + zero-init output projection** | Output proj starts at 0, block = identity initially | 100+ layers, no warmup | modded-nanogpt speedrun | Zero-initialized output means each block is identity at step 0 — gradients flow perfectly from the start |

The trend is clear: **initially, the block should do nothing**. Each upgrade made the block closer to an identity function at initialization, enabling deeper stacks to train without careful learning rate warmup.

> 🔄 **Why start here**: Post-norm (LayerNorm *after* the residual addition) is the original 2017 layout. It works well for shallow stacks (4–6 layers). **What changes later**: Phase 2 (Chapter 07) switches to **pre-norm** (LayerNorm *before* the sublayer), which enables stable training for 100+ layer stacks.

### Weight Initialization

How do we initialize all these parameters? The standard approach for Phase 1 is **N(0, 0.02)** — a normal distribution with mean 0 and standard deviation 0.02:

```python
def _init_weights(self):
    for p in self.parameters():
        if p.dim() >= 2:
            nn.init.normal_(p, mean=0.0, std=0.02)
```

Why 0.02? It's a "Goldilocks" value that's small enough to avoid saturating activations in the first forward pass, but large enough to break symmetry and allow learning. The 0.02 number became standard through experimentation in GPT-2.

> 🔄 **Modern upgrade**: Since 2025, many implementations zero-initialize the output projection of each sub-layer (the final Linear in attention and FFN) instead of using N(0,0.02). This improves stability and hyperparameter transfer (modded-nanogpt speedrun). We keep N(0,0.02) for simplicity — you'll see zero-init in Phase 2.

---

## Part 5: Stacking — Why Multiple Blocks?

A single transformer block is powerful, but the real magic comes from **stacking** them. Each layer builds on the previous layer's representations:

```
Output:    "ROMEO: I dreamt a dream tonight."
             ↑
Layer 6:  ┌─ Decoder Block ─┐    ↔ Sentence-level meaning
          │  Attn → FFN     │     "I" → "dreamt" (main clause)
          └─────────────────┘
             ↑
Layer 5:  ┌─ Decoder Block ─┐    ↔ Phrase-level structure
          │  Attn → FFN     │     "a dream" → compound noun
          └─────────────────┘
             ↑
Layer 4:  ┌─ Decoder Block ─┐    ↔ Syntax (SVO relationships)
          │  Attn → FFN     │     "ROMEO" → "dreamt" (subject-verb)
          └─────────────────┘
             ↑
Layer 3:  ┌─ Decoder Block ─┐    ↔ Word-level patterns
          │  Attn → FFN     │     "dreamt" → "dream" (root)
          └─────────────────┘
             ↑
Layer 2:  ┌─ Decoder Block ─┐    ↔ Local context (adjacent words)
          │  Attn → FFN     │     "a dream" → determiner + noun
          └─────────────────┘
             ↑
Layer 1:  ┌─ Decoder Block ─┐    ↔ Token features + position
          │  Attn → FFN     │     "R" "O" "M" "E" "O" ":"
          └─────────────────┘
             ↑
Input:    [Token Embeddings + Positional Encoding]
```

This hierarchy is why deep transformers work so well. Lower layers learn concrete patterns (words, syntax). Upper layers learn abstract patterns (semantics, discourse, intent).

> 🟢 **Key insight for non-engineers**: Think of it like reading a book. You start by recognizing letters (layer 1), then words (layer 2), then grammar (layer 3-4), then plot and meaning (layer 5+). Each level builds on the one below.

---

## Part 6: 🔵 Code Section — Implementing the Block

Let's build each piece step by step, then assemble them into a complete decoder block.

### FFN + LayerNorm + Residual Demo

First, let's verify each component independently:

```python
import torch
import torch.nn as nn

B, T, C = 2, 8, 16

# FFN
ffn = nn.Sequential(
    nn.Linear(C, C * 4),    # expand 4x: 16 → 64
    nn.ReLU(),
    nn.Linear(C * 4, C),    # compress: 64 → 16
)
x = torch.randn(B, T, C)
out_ffn = ffn(x)
print(f"FFN: {x.shape} → {out_ffn.shape}")

# LayerNorm
ln = nn.LayerNorm(C)
out_ln = ln(x)
print(f"LayerNorm: mean={out_ln.mean():.4f}, std={out_ln.std():.4f}")

# Residual
skip = x + out_ffn
print(f"Residual: {skip.shape}")
```

### Complete Decoder Block

Now combine attention (from Chapter 04), FFN, LayerNorm, and residual connections:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalMultiHeadAttention(nn.Module):
    """Self-contained copy — see Chapter 04 for details."""
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
        x = self.norm1(x + self.attn(x))   # post-norm
        x = self.norm2(x + self.ffn(x))    # post-norm
        return x

block = VanillaDecoderBlock(d_model=16, n_heads=4)
x = torch.randn(1, 8, 16)
out = block(x)
print(f"Vanilla block: {x.shape} → {out.shape}")
```

### Run the Companion Code

```bash
python3 code/05-ffn-layernorm/01_ffn_layernorm.py
python3 code/05-ffn-layernorm/02_vanilla_decoder_block.py
```

**Expected output — `01_ffn_layernorm.py`:**
```
FFN: torch.Size([2, 8, 16]) → torch.Size([2, 8, 16])
  Hidden dimension (d_ff): 64
  Activation: ReLU (modern models use SwiGLU — Phase 2)

LayerNorm: torch.Size([2, 8, 16])
  Mean: 0.0012, Std: 1.0023
  (should be ~0.0 and ~1.0)

Residual x + FFN(x): torch.Size([2, 8, 16])

Mini FFN block: torch.Size([2, 8, 16]) → torch.Size([2, 8, 16])
  Parameters: 2,128

Verification: LayerNorm output stats over 1000 random inputs...
  Mean across runs: 0.0000 (target: 0.0)
  Std  across runs: 1.0000 (target: 1.0)
✓ LayerNorm correctly normalizes any input distribution
```

**Expected output — `02_vanilla_decoder_block.py`:**
```
Vanilla block: torch.Size([1, 8, 16]) → torch.Size([1, 8, 16])
Parameters: 3,216
Gradient norm (all params): 10.2345
✓ Gradients flow through residual connections

Stacking 6 blocks...
  Input:  torch.Size([1, 8, 16])
  Output: torch.Size([1, 8, 16])
  Total parameters (6 blocks): 19,296
✓ Stack of 6 decoder blocks works
```

---

## Part 7: 🟢 Summary Box

```
Each transformer block does two things:
  1. Mix information across positions (attention)
  2. Process each token independently (FFN)

Residual connections and LayerNorm make deep stacks trainable.

  Attention = team meeting (tokens share info)
  FFN       = individual work time (each token processes alone)
  Residual  = express lane (gradients flow directly)
  LayerNorm = temperature control (keeps activations stable)

Post-norm (2017 layout):
  x = LayerNorm(x + Attention(x))
  x = LayerNorm(x + FFN(x))
```

| Term | What It Is |
|------|------------|
| **FFN** | Feed-Forward Network: expand 4× → ReLU → compress. Processes each token independently. |
| **d_ff** | Hidden dimension of the FFN, typically `4 × d_model` |
| **LayerNorm** | Normalizes each token's features to mean 0, std 1, then applies learned scale (γ) and shift (β) |
| **Residual connection** | `output = x + sublayer(x)` — allows gradients to bypass sublayers |
| **Post-norm** | LayerNorm applied *after* the residual addition (original 2017 layout) |
| **Block/Layer** | One complete unit: Attention → Add → Norm → FFN → Add → Norm |
| **Stacking** | Arranging multiple blocks sequentially, each building on the previous layer's representations |

> 💡 **Key insight**: The transformer block is a **complete computational unit** — it mixes (attention), processes (FFN), stabilizes (LayerNorm), and preserves (residual). A transformer is just this block repeated N times. Every modern LLM you've used is built from 8 to 100+ copies of this same fundamental structure.

---

## Part 8: 🟢 Check Your Understanding

Test yourself before moving to the next chapter.

1. **What's the difference between attention and FFN? (Use the team meeting vs individual work analogy.)**

   <details>
   <summary>Show answer</summary>
   Attention is the "team meeting" — tokens share information across positions by computing weighted mixtures of other tokens' values. FFN is "individual work time" — each token independently processes its own information through a non-linear transformation (expand → ReLU → compress). Both are needed: you find relevant information (attention), then you process what you found (FFN), then you attend again with your new understanding, and so on.
   </details>

2. **Why do we need LayerNorm?**

   <details>
   <summary>Show answer</summary>
   As data flows through layers, activation values can drift — some dimensions grow very large, others shrink. This makes optimization unstable because the next layer has to deal with constantly changing input distributions. LayerNorm recenters (subtract mean) and rescales (divide by std) each token's features, keeping activations in a stable range. The learned γ and β parameters allow the model to adapt the normalization as needed. Think of it as a temperature control system for your neural network.
   </details>

3. **What would happen if we stacked 20 blocks without residual connections?**

   <details>
   <summary>Show answer</summary>
   The gradients would vanish — early layers would receive near-zero gradient updates and effectively stop learning. With 6 layers it might still work (barely), but with 20+ layers the gradient signal is exponentially diluted as it passes through each layer's derivatives. Residual connections solve this by providing a "gradient superhighway" — the identity path lets gradients flow directly from output to input without passing through any sublayer, ensuring early layers get meaningful updates.
   </details>

4. **Which comes first in post-norm: LayerNorm or the sublayer?**

   <details>
   <summary>Show answer</summary>
   In **post-norm**, LayerNorm comes **after** the sublayer: `x = LayerNorm(x + sublayer(x))`. The word "post" means after — LayerNorm is applied to the sum of the input and the sublayer output. This is the original 2017 layout. Later (Phase 2, Chapter 07) you'll learn about **pre-norm**, where LayerNorm comes *before* the sublayer, which enables training much deeper stacks.
   </details>

---

## Part 9: Common Beginner Mistakes (Reference)

> **⚠️ Post-norm and training instability**: Post-norm works well for 4–6 layers but can become unstable with deeper stacks. If you try scaling to 12+ layers and training diverges, this is expected — you'll learn about the pre-norm fix in Phase 2.

> **⚠️ Forgetting the residual connection**: The most common bug when assembling a block. Without `x + attn(x)`, your block has no gradient shortcut. The block may still train at very shallow depths but will fail at any meaningful scale.

> **⚠️ LayerNorm applied to the wrong dimension**: `nn.LayerNorm(C)` normalizes over the last dimension `C`. If you accidentally normalize over the sequence dimension `T`, the model can't distinguish between different sequence positions. In `(B, T, C)` format, always normalize over `C`.

> **⚠️ FFN dimension mismatch**: The first Linear goes from `d_model → d_ff` (4×), the second from `d_ff → d_model`. If you accidentally use `d_model` for both (making the FFN just two projections with no expansion), the FFN loses its capacity to learn complex patterns.

> **⚠️ Weight initialization scale**: Using std=0.02 works for most Phase 1 configurations, but if your model is much wider (d_model > 1024) or deeper (layers > 12), you may need to adjust. The 0.02 number is a heuristic, not a law.

> **⚠️ Zero-init output projections (modern)**: Since 2025, many implementations zero-initialize the output projection (the final `nn.Linear` in both attention and FFN) instead of N(0,0.02). This means the block initially acts as the identity function (since the output projection starts at zero), which improves training stability. Our Phase 1 code uses N(0,0.02) for simplicity — you'll see zero-init in Phase 2.

> **⚠️ Counting parameters**: For a small block (d_model=16, n_heads=4, d_ff=64): Attention has ~800 params (QKV: 16×48, out: 16×16), FFN has ~2,080 params (16×64 + 64×16), LayerNorm has ~32 params (2×16). Total per block: ~2,900. For the full model with embedding: add vocab_size×d_model (65×16 = 1,040) and final LayerNorm (32). A 4-block model would be ~12,600 params. These numbers help you verify your implementation.

---

## Part 10: Further Learning

- **Next Chapter**: Chapter 06 — The Training Loop & Generation. *All the pieces you've built (tokenization, embedding, attention, FFN, LayerNorm, residual connections) come together into a complete model that can learn from data and generate text.*

- For an interactive deep-dive, see **edu-transformer** (2026) — a single-file implementation with 9 learning modes that trains in 3 seconds.

- A great visual reference is the [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) — an interactive visualization of the entire transformer block.

---

## Terms Introduced

| Term | Quick Definition |
|------|------------------|
| **FFN** | Feed-Forward Network: a two-layer neural network that processes each token independently |
| **d_ff** | Hidden dimension of the FFN (typically 4× d_model) |
| **ReLU** | Activation function: `max(0, x)` — introduces non-linearity |
| **LayerNorm** | Normalization: `γ × (x - μ)/σ + β` — stabilizes activations |
| **Residual connection** | Skip connection: `output = x + sublayer(x)` — prevents vanishing gradients |
| **Post-norm** | Architecture where LayerNorm is applied after the residual addition |
| **Block** | Complete transformer unit: Attention → Add → Norm → FFN → Add → Norm |
| **Stack** | Multiple blocks arranged sequentially |
| **Parameter count** | Total number of trainable weights in a model |

---

> **Next Chapter**: Chapter 06 — The Training Loop & Generation.
>
> *All the pieces you've built now connect into a complete model. You'll learn how training works end-to-end and how transformers generate text token by token.*
>
> *🔵 Make sure both companion scripts from this chapter run cleanly before proceeding.*
