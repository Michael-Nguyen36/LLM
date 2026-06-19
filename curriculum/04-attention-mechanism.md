# Chapter 04: The Attention Mechanism

> **Audience**: 🟢 All roles (code sections marked 🔵)
> **Prerequisites**: Chapter 03 (Tokenization, Embeddings & PE), basic Python, matrix multiply shapes
> **Estimated time**: 25 minutes read, 20 minutes code

---

## Why This Matters

The attention mechanism is the single most important idea in modern deep learning. It's the "secret sauce" that made transformers possible — and it's what every LLM you've used (GPT, Claude, Gemini, Llama) is built on.

> 🌟 **If you only deeply understand one chapter in this curriculum, make it this one.** Everything else builds on attention.

By the end of this chapter, you'll understand:
- **Why** attention exists (RNNs had a fundamental flaw)
- **How** attention works step-by-step with concrete numbers
- **What** Query, Key, and Value mean (with a vivid analogy you won't forget)
- **Why multiple heads are better** than one
- **How** the causal mask prevents cheating

---

## Part 1: The Problem Attention Solves

Consider this sentence:

> **"The cat chased the mouse because it was hungry."**

What does **"it"** refer to? Obviously, **the cat** (the cat is hungry, not the mouse). But how does your brain know that?

### The Sequential Processing Problem

Before transformers, the dominant architecture was the **RNN (Recurrent Neural Network)**. RNNs process tokens one at a time, maintaining a single "hidden state" that carries information forward:

```
Input:   The → cat → chased → the → mouse → because → it → was → hungry
State:   h0  → h1  →   h2   → h3 →   h4   →   h5   → h6 → h7  →  h8
```

The problem? By the time the RNN reaches "it" at position 6, the information from "cat" at position 1 has been mixed with everything in between. The hidden state is a blur — it remembers everything a little bit and nothing clearly.

> 🎭 **Spotlight on a Stage**: Imagine an RNN as a single spotlight following one actor across a dark stage. As the spotlight moves, the previous actor fades into darkness. By the time you reach "it", the spotlight has moved so far that "cat" is barely visible. **Attention** is like turning on ALL the lights at once — every actor (token) is fully illuminated, and you can look directly at any of them from anywhere on stage.

### What Attention Does Differently

Instead of compressing everything into one hidden state, **attention lets every token look at every other token directly**:

```
"it" ──→ looks at "cat" ◄── sees strong relevance
     ──→ looks at "chased" ◄── sees some relevance
     ──→ looks at "mouse" ◄── sees weak relevance
     ──→ looks at "the" ◄── sees almost no relevance
```

Each token produces a **weighted mixture** of all tokens' information, where the weights come from how relevant each other token is to it.

---

## Part 2: The QKV Analogy — A Library Search

To understand **Query (Q), Key (K), and Value (V)**, imagine you walk into a **library**:

| Component | Library Analogy | In Attention |
|-----------|-----------------|--------------|
| **Query (Q)** | Your search question: *"What books discuss Roman architecture?"* | What this token is looking for |
| **Key (K)** | Each book's topic label on the spine: *"Roman Architecture", "Cooking", "Physics"* | What each token contains / offers |
| **Value (V)** | The full content of each book you retrieve | The actual information you take away |

### The Search Process

```
Step 1: You enter with your question (Query)
        "What books discuss Roman architecture?"

Step 2: You scan each shelf, comparing your question to each book's label
        Query matches Key: "Roman Architecture" → high score
        Query vs Key: "Cooking" → low score
        Query vs Key: "Physics" → medium score

Step 3: You retrieve the most relevant books' content
        You take home: mostly from "Roman Architecture"
        A bit from: "Physics" (it discussed aqueducts)
        Nothing from: "Cooking"
```

Attention does the same thing in math: **compare Q against every K, turn those comparisons into weights, then take a weighted sum of V**.

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
                     ──┬───────────────┬──   ─┬─
                       │                │      └── Weighted sum of Values
                       │                └── Attention weights (how much to look at each token)
                       └── How relevant is each token to the query?
```

> 🕰️ **Before Scaled Dot-Product Attention**: The 2017 transformer wasn't the first attention mechanism. **Additive attention** (Bahdanau et al., 2014) used a small neural network to score query-key pairs: `score = v^T tanh(W_q q + W_k k)`. **Dot-product attention** (Luong et al., 2015) simplified this to `q · k` but lacked the scaling factor. The 2017 innovation was adding `1/√d_k` scaling to prevent gradient vanishing in high dimensions — making attention stable for large models.

### The Two Roles of Each Token

Every token plays **two roles simultaneously**:

1. **As a question-asker**: it has a Query vector — *"what am I looking for?"*
2. **As an answer-giver**: it has a Key vector — *"what information do I contain?"* — and a Value vector — *"here's my actual information to share"*

When token A looks at token B, it compares its **Q** against token B's **K** to decide if B is relevant, then takes B's **V** weighted by that relevance.

```
                  Q_from_A · K_from_B  →  relevance score from A to B
                  weights × V_from_B   →  information A takes from B
```

---

## Part 3: Attention Formula — Step by Step with Concrete Numbers

Let's work through attention with real numbers. We'll use:

- **3 tokens** (T=3) — tiny sequence
- **d_model = 4** — each token is a 4-dimensional vector

### Setup

```
Token 0: [1, 0, 0, 0]    (e.g., "cat")
Token 1: [0, 1, 0, 0]    (e.g., "chased")
Token 2: [0, 0, 1, 0]    (e.g., "mouse")
```

> For simplicity, we skip the learned W_q, W_k, W_v projections here and use the token embeddings directly as Q, K, V. **In practice, learned projections are essential** — they're what allow each head to learn different relationship types (syntax, semantics, position, etc.). Without them, all heads would compute identical attention patterns. The multi-head section below explains this.

### Step 3a: Compute Scores (Q @ K^T)

Each token compares its Query against every token's Key:

```
Q = [[1, 0, 0, 0],     K = [[1, 0, 0, 0],
     [0, 1, 0, 0],          [0, 1, 0, 0],
     [0, 0, 1, 0]]          [0, 0, 1, 0]]

Scores = Q @ K^T

        ┌─ Q0 → ─┐   ┌─     ─┐   ┌─                         ─┐
        │ 1 0 0 0 │   │ 1 0 0 │   │ Q0·K0  Q0·K1  Q0·K2  │   │ 1  0  0 │
        │ 0 1 0 0 │ × │ 0 1 0 │ = │ Q1·K0  Q1·K1  Q1·K2  │ = │ 0  1  0 │
        │ 0 0 1 0 │   │ 0 0 1 │   │ Q2·K0  Q2·K1  Q2·K2  │   │ 0  0  1 │
        └─       ─┘   │ 0 0 0 │   └─                         ─┘
                      └─     ─┘
```

Shape check: `(3, 4) @ (4, 3) → (3, 3)`

Each cell `[i, j]` = dot product of token i's Query and token j's Key = **how much token i should attend to token j**.

### Step 3b: Scale (divide by √d_model)

We divide by √d_model (here √4 = 2) to prevent the softmax from becoming too "peaky" (one weight near 1, all others near 0). This is called **temperature control** — scaling keeps the scores in a range where softmax works well.

```
Scores (raw):     [[1, 0, 0],         Scaled:  [[0.5, 0,  0 ],
                   [0, 1, 0],                  [0,  0.5, 0 ],
                   [0, 0, 1]]                  [0,  0,  0.5]]
```

### Step 3c: Softmax (turn scores into weights)

Softmax converts each row into a probability distribution (sums to 1):

```
For row 0: softmax([0.5, 0, 0]) 
         = [e^0.5/(e^0.5+e^0+e^0), e^0/(e^0.5+e^0+e^0), e^0/(e^0.5+e^0+e^0)]
         = [1.65/(1.65+1+1), 1/(1.65+1+1), 1/(1.65+1+1)]
         = [0.45, 0.27, 0.27]
```

```
Weights = softmax(Scaled Scores)
       = [[0.45, 0.27, 0.27],    ← Token 0 attends to itself (45%) and others (27% each)
          [0.27, 0.45, 0.27],    ← Token 1 attends mostly to itself
          [0.27, 0.27, 0.45]]    ← Token 2 attends mostly to itself
```

✅ **Each row sums to 1.** These are the attention weights.

### Step 3d: Compute Output (Weights @ V)

Each token's output is a weighted sum of all Value vectors:

```
V = [[1, 0, 0, 0],      Output = Weights @ V
     [0, 1, 0, 0],
     [0, 0, 1, 0]]      Token 0 output = 0.45×[1,0,0,0] + 0.27×[0,1,0,0] + 0.27×[0,0,1,0]
                                           = [0.45, 0.27, 0.27, 0]
                         Token 1 output = [0.27, 0.45, 0.27, 0]
                         Token 2 output = [0.27, 0.27, 0.45, 0]
```

> 🔍 **What just happened?** Token 0's output is mostly its own information (0.45) mixed with a bit from Token 1 (0.27) and Token 2 (0.27). In a real model with learned projections, these would be far more nuanced — capturing syntax, semantics, and long-range dependencies.

### The Full Formula

```python
# In code, this is the complete attention operation:
scores = Q @ K.transpose(-2, -1)    # (T, T) — similarity matrix
scores = scores / (d_k ** 0.5)       # scale to prevent softmax saturation
weights = F.softmax(scores, dim=-1)  # (T, T) — rows sum to 1
output = weights @ V                  # (T, d_k) — weighted sum of values
```

---

## Part 4: Multi-Head Attention

### Why One Head Isn't Enough

A single attention pattern is like **asking one question** of the entire sentence:

- "Which words relate to the subject?" ← this is one kind of relationship
- "Which words are verbs connected to?" ← this is a different kind
- "Which words modify which other words?" ← yet another

One attention head has to pick ONE way of looking at relationships. That's limiting.

> 👥 **Expert Panel Analogy**: Imagine you're trying to understand a complex problem. Instead of asking one person, you assemble a panel of experts:
> - **Expert 1** (Syntactic Head): "I focus on grammatical relationships — subjects, verbs, objects."
> - **Expert 2** (Semantic Head): "I focus on meaning — what's the topic of discussion?"
> - **Expert 3** (Positional Head): "I focus on distance — which tokens are close by?"
> - **Expert 4** (Entity Head): "I focus on named entities — people, places, things."
>
> Each expert looks at the same sentence but pays attention differently. Then you combine their opinions before making a decision.

### How Multi-Head Works

Multi-head attention runs **H parallel attentions**, each with its own learned projections. Here's the shape pipeline:

```
Start: (B, T, 512)                               ← input embeddings
          │
          ▼
Project Q, K, V separately for each head:
  ──→ Q₁: (B, T, 64), K₁: (B, T, 64), V₁: (B, T, 64)    ← Head 1
  ──→ Q₂: (B, T, 64), K₂: (B, T, 64), V₂: (B, T, 64)    ← Head 2
  ──→ ... (8 heads total)
  ──→ Q₈: (B, T, 64), K₈: (B, T, 64), V₈: (B, T, 64)    ← Head 8
          │
          ▼
Each head computes attention independently:
  Head 1: Attention(Q₁, K₁, V₁) → (B, T, 64)
  Head 2: Attention(Q₂, K₂, V₂) → (B, T, 64)
  ...
  Head 8: Attention(Q₈, K₈, V₈) → (B, T, 64)
          │
          ▼
Concatenate all heads: (B, T, 64×8) = (B, T, 512)
          │
          ▼
Project through W_O: (B, T, 512) → (B, T, 512)   ← final output
```

The key insight for the code is the **reshape and transpose dance**:

```
Input:  (B, T, C)            ← e.g., (B, T, 512)
Step 1: view → (B, T, H, d_head)    ← split last dim into heads  (512 = 8 × 64)
Step 2: transpose → (B, H, T, d_head)  ← bring head dim next to batch
Step 3: attention on (B, H, T, d_head) → (B, H, T, d_head)
Step 4: transpose back → (B, T, H, d_head)
Step 5: view → (B, T, C)     ← merge heads back
Step 6: W_O projection → (B, T, C)
```

### 🕰️ How Multi-Head Attention Evolved

The 2017 design — one query, one key, one value *per head* — has been optimized significantly as models grew from 110M (BERT-base) to 400B+ (LLaMA 3) parameters. The bottleneck: **the KV cache**. During generation, every head's K and V vectors must be cached for all past tokens. With 96 layers × 96 heads, that's ~2 GB *per sequence* — and you serve millions of sequences.

| Year | Method | Query Heads | Key/Value Heads | KV Cache Savings | Used By | Why |
|------|--------|-------------|-----------------|-------------------|---------|-----|
| **2017** | **MHA** (Multi-Head Attention) | H | H | 1× (baseline) | Vaswani et al. | Original design — max expressiveness |
| **2019** | **MQA** (Multi-Query Attention) | H | 1 | H× smaller | Shazeer (Google), PaLM | Single KV head shared across all Q heads; 3× faster inference for similar quality |
| **2023** | **GQA** (Grouped Query Attention) | H | G (usually 8) | ~H/G × smaller | LLaMA 2, LLaMA 3, Mistral, Gemini | Best trade-off: more expressive than MQA, less memory than MHA |
| **2024** | **MLA** (Multi-head Latent Attention) | H | Latent compressed | 75–93% cheaper | DeepSeek | Compresses KV into a low-dimensional latent space; reclaims the expressiveness of MHA with MQA-level memory |

The `d_k=64` convention has stayed remarkably consistent — the original 2017 paper used `d_model=512` with `n_heads=8` → `d_k=64`. Most models still target `d_k=64` or `d_k=128`, regardless of model size.

> 🔄 **Why start here**: This is the original 2017 multi-head design with separate QKV projections per head — simple and pedagogical. **What changes later**: Phase 2 (Chapter 08) upgrades to **GQA** (Grouped Query Attention), which shares KV heads across query groups to reduce memory.

> 🧠 **Common beginner mistakes (Multi-head shapes)**:
> - Forgetting `.transpose(1, 2)` — without it, the batch tokens and heads are mixed up. Always: `view → transpose`.
> - Forgetting `.contiguous()` before `.view()` after `transpose` — `transpose` returns a non-contiguous tensor. Use `.contiguous().view(...)` or `.reshape(...)`.
> - Confusing `d_model`, `n_heads`, and `head_dim`. Always: `d_model = n_heads × head_dim`. If it doesn't divide evenly, **the code crashes**.

---

## Part 5: Causal Masking

### Why We Need It

When training an autoregressive model, we show it an entire sequence and ask it to predict each next token. **The model must not be allowed to cheat by looking at future tokens.**

Without a mask, attention lets every token see every other token — including future ones. Token at position 2 could just "peek" at position 3's embedding and trivially predict it.

### How the Causal Mask Works

The causal mask is an **upper triangular** matrix of 0s and 1s, converted to 0s and -inf:

```
Position:      0    1    2    3
        0:  [  0, -inf, -inf, -inf]    ← Token 0 sees only itself
        1:  [  0,   0,  -inf, -inf]    ← Token 1 sees tokens 0-1
        2:  [  0,   0,    0,  -inf]    ← Token 2 sees tokens 0-2
        3:  [  0,   0,    0,    0 ]    ← Token 3 sees all tokens
```

We add (or `masked_fill`) this to the scores **before softmax**:

```
Before mask:  scores = [[0.5, 0.3, 0.1, 0.2],
                        [0.2, 0.6, 0.1, 0.1],
                        [0.3, 0.2, 0.4, 0.1],
                        [0.1, 0.1, 0.1, 0.7]]

After mask:   scores = [[0.5, -inf, -inf, -inf],
                        [0.2, 0.6,  -inf, -inf],
                        [0.3, 0.2,  0.4,  -inf],
                        [0.1, 0.1,  0.1,  0.7]]

After softmax:         [[1.0, 0.0,  0.0,  0.0],
                        [0.4, 0.6,  0.0,  0.0],
                        [0.4, 0.3,  0.3,  0.0],
                        [0.2, 0.2,  0.2,  0.4]]
```

Notice: after softmax, the masked positions become **exactly zero**. Token 0's output is purely its own value. Token 3's output uses all tokens (it has no future to peek at).

In code:
```python
mask = torch.tril(torch.ones(T, T, dtype=torch.bool)).view(1, 1, T, T)  # lower triangular
scores = scores.masked_fill(~mask, float('-inf'))                       # upper triangle → -inf
```

> 🧠 **Common beginner mistakes (Causal masking)**:
>
> **1. Mask convention confusion**: `torch.tril` creates a **lower** triangular matrix (1s for allowed positions, 0s for future). We mask **where mask == 0**. Some implementations use `torch.triu` (upper triangular) and mask where == 1. Either works — just be consistent.
>
> **2. Mask shape broadcasting**: The mask must broadcast to `(B, H, T, T)`. We create it as `(1, 1, T, T)` — the first two dimensions broadcast across batch and heads. Forgetting the `.view(1, 1, T, T)` means shape `(T, T)` won't broadcast to `(B, H, T, T)` and you'll get a silent error or unintended behavior.
>
> **3. Mask added, not multiplied**: We use `masked_fill` (or add -inf), NOT multiply. Multiplying by 0 would give `0` for future positions, which still contributes to softmax denominators. `-inf` makes `e^(-inf) = 0`, completely zeroing out those positions.
>
> **4. Causal mask leakage**: The diagonal of `tril` is all 1s — meaning each token can attend to **itself**. This is correct for autoregressive models (a token needs its own embedding to compute its output). If you accidentally make the diagonal 0 (using `torch.tril(..., diagonal=-1)`), every token would be blind to itself and the model can't learn.

---

## Part 6: 🔵 Code Section — Simple Attention from Scratch

Let's implement single-head attention:

```python
import torch
import torch.nn.functional as F

B, T, C = 1, 4, 8

x = torch.randn(B, T, C)

# Project to Q, K, V with random weight matrices
Wq = torch.randn(C, C)
Wk = torch.randn(C, C)
Wv = torch.randn(C, C)

Q = x @ Wq  # (1, 4, 8)
K = x @ Wk  # (1, 4, 8)
V = x @ Wv  # (1, 4, 8)

scores = Q @ K.transpose(-2, -1)            # (1, 4, 4)
scores = scores / (C ** 0.5)                # scale
weights = F.softmax(scores, dim=-1)         # (1, 4, 4)
output = weights @ V                         # (1, 4, 8)
```

Now let's build the full multi-head causal attention as a proper `nn.Module`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalMultiHeadAttention(nn.Module):
    """Multi-head causal self-attention — 2017 style."""
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
        qkv = self.qkv(x)                                      # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)                         # each (B, T, C)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, d_head)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)  # (B, H, T, T)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        return self.out(out)
```

### 🧪 Recommended Exercises (🔵 Engineers)

1. **Trace tensor shapes on paper**: Before running any code, take a concrete example (B=2, T=8, d_model=16, n_heads=4). Write down the shape after every operation in `forward()`. Check against the code.

2. **Verify causal masking manually**: Create a tiny 2-token sequence. Print `scores` before and after `masked_fill`, then after softmax. Verify that token 0's weights are `[1.0, 0.0]` and token 1's are `[w, 1-w]` (non-zero on both).

3. **Overfit a single batch**: Create one input-target pair and train the attention module to predict the target. If loss goes to ~0, your implementation is correct. If it plateaus, check shapes and masking.

4. **Visualize attention heatmaps**: Save the `weights` tensor from a forward pass. Use `matplotlib`'s `imshow` to draw it as a heatmap. The causal mask should be clearly visible as a blank upper triangle.

5. **Implement single-head first**: If anything breaks, ignore `n_heads` and implement attention with `n_heads=1`. Get that working, then generalize to multi-head. Most bugs come from the reshape/transpose dance, not from attention itself.

### Run the Companion Code

```bash
python3 code/04-attention/01_simple_attention.py
python3 code/04-attention/02_multi_head_causal.py
```

**Expected output — simple attention:**
```
Q shape:     torch.Size([1, 4, 8])
K shape:     torch.Size([1, 4, 8])
V shape:     torch.Size([1, 4, 8])
Scores:     torch.Size([1, 4, 4])
Weights:    torch.Size([1, 4, 4])
Output:     torch.Size([1, 4, 8])

Attention weights from token 0 to all tokens:
[0.25  0.25  0.25  0.25]
```

**Expected output — multi-head causal:**
```
Input:  torch.Size([1, 6, 16])
Output: torch.Size([1, 6, 16])
✓ Multi-head causal attention works
```

---

## Part 7: 🟢 Summary Box

```
Attention: Every token mixes information from all other tokens, weighted by relevance.

  ┌──────────────────────────────────────────────────────────┐
  │  Q (Query):  "What am I looking for?"                    │
  │  K (Key):    "What information do I contain?"            │
  │  V (Value):  "Here is my actual information"             │
  │                                                          │
  │  scores = Q @ K^T                                        │
  │  weights = softmax(scores / sqrt(d_k))                   │
  │  output = weights @ V                                    │
  └──────────────────────────────────────────────────────────┘

Multi-head: Run H parallel attentions (each with different projections).
Causal mask: Upper triangle of -inf prevents attending to future tokens.
```

| Term | What It Is |
|------|------------|
| **Attention** | A mechanism that lets each token look at all other tokens, weighted by relevance |
| **Query (Q)** | What the current token is looking for |
| **Key (K)** | What each token offers / its identity |
| **Value (V)** | The actual information each token shares |
| **Attention scores** | Dot products of Q and K — raw relevance before softmax |
| **Attention weights** | Scores after softmax — each row sums to 1.0 |
| **Multi-head** | Running H parallel attentions to capture different relationship types |
| **Head dimension** | `d_model / n_heads` — each head works in a smaller subspace |
| **Causal mask** | Prevents tokens from attending to future positions |
| **Scaled dot-product** | The specific attention variant used by transformers |

> 💡 **Key insight**: Attention is fundamentally a **weighted averaging operation** with learned weights. The magic comes from the weights being **input-dependent** — different inputs produce different attention patterns. This is what makes transformers so flexible.

---

## Part 8: 🟢 Check Your Understanding

Test yourself before moving to the next chapter.

1. **In the library analogy, what do Q, K, and V represent?**

   <details>
   <summary>Show answer</summary>
   Q (Query) is your search question — what you're looking for. K (Key) is each book's topic label — what information it claims to contain. V (Value) is the book's full content — the actual information you take away.
   </details>

2. **Why do we divide attention scores by sqrt(d_k)?**

   <details>
   <summary>Show answer</summary>
   Without scaling, high-dimensional dot products can become very large, pushing softmax into regions where one weight dominates and gradients vanish (too "peaky"). Dividing by sqrt(d_k) keeps scores in a reasonable range where softmax produces meaningful gradients. This is called temperature control.
   </details>

3. **What does the causal mask prevent?**

   <details>
   <summary>Show answer</summary>
   It prevents each token from attending to future tokens. During training, the model sees the full sequence but must predict each token using only itself and previous tokens. Without the mask, the model could "cheat" by looking at the correct answer from future positions.
   </details>

4. **Why do we need multiple attention heads instead of just one?**

   <details>
   <summary>Show answer</summary>
   A single attention head can only capture one type of relationship at a time (e.g., syntax OR semantics OR proximity). Multiple heads with different learned projections can specialize — one head learns grammatical relations, another learns semantic similarity, etc. The outputs are concatenated, giving the model a richer understanding than any single head could provide.
   </details>

5. **What's the shape of the attention score tensor in multi-head attention, and why?**

   <details>
   <summary>Show answer</summary>
   `(B, H, T, T)` — batch, heads, query positions, key positions. The first two dimensions (B, H) are independent (each batch item and each head computes its own attention matrix). The last two (T, T) contain the pairwise relevance scores: score[i, j] = how much token i should attend to token j for this head.
   </details>

---

## Part 9: Common Beginner Mistakes (Reference)

As you start writing attention code, keep these pitfalls in mind:

> **⚠️ Mask convention confusion**: `masked_fill(mask == 0, -inf)` means positions with 0 become -inf. If your mask accidentally uses 1 for "future" and 0 for "allowed", the attention will be backwards. Always print your mask and verify.

> **⚠️ Mask shape broadcasting**: Mask must be `(1, 1, T, T)` to broadcast to `(B, H, T, T)`. A bare `(T, T)` mask will broadcast wrong across batch and heads.

> **⚠️ Mask added not multiplied**: `masked_fill` sets positions to `-inf` so softmax makes them zero. Never multiply by 0 — that leaves them in the softmax denominator, polluting the distribution.

> **⚠️ Causal mask leakage**: `torch.tril(torch.ones(T, T))` keeps the diagonal as 1 — each token attends to itself. Using `diagonal=-1` would prevent self-attention, which breaks most architectures.

> **⚠️ Dimension ordering**: PyTorch convention is `(B, T, C)` = batch, time/sequence, channels. Some implementations use `(T, B, C)` (batch-second). The companion code in this curriculum consistently uses `(B, T, C)`.

> **⚠️ Missing sqrt(d_k) scaling**: Easy to forget. Without it, models with larger head dimensions will have vanishing gradients from overly-peaky softmax.

> **⚠️ Missing residual connections**: Attention is usually followed by `x = x + attn(x)` — a residual connection. The code in this chapter returns the attention output directly (before the residual), but the full block in Chapter 05 adds it back.

> **⚠️ Embedding scale**: Some implementations multiply embeddings by `sqrt(d_model)` before adding positional encoding. This is a scaling trick to keep the combined values in a reasonable range. The vanilla transformer uses this.

---

## Terms Introduced

| Term | Quick Definition |
|------|------------------|
| **Attention** | Mechanism that computes weighted mixtures of token representations based on pairwise relevance |
| **Query (Q)** | Token's "what am I looking for?" vector |
| **Key (K)** | Token's "what do I contain?" vector |
| **Value (V)** | Token's "here's my information" vector |
| **Scaled dot-product attention** | Attention variant: `softmax(QK^T/√d)V` |
| **Multi-head attention** | H parallel attention operations with different learned projections |
| **Causal mask** | Matrix that prevents attending to future positions |
| **d_head / head_dim** | `d_model / n_heads` — per-head working dimension |
| **Autoregressive** | Model that predicts one token at a time, conditioned on past tokens |

---

> **Next Chapter**: Chapter 05 — The FFN, LayerNorm & The Complete Block.
>
> *Attention mixes information across positions. The FFN processes each token independently. LayerNorm keeps training stable. Together they form the complete transformer block.*
>
> *🔵 Make sure both companion scripts from this chapter run cleanly before proceeding.*
