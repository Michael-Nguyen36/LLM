# Chapter 06: The Training Loop & Generation

> **Audience**: 🟢 All roles (code sections marked 🔵)
> **Prerequisites**: Chapters 03–05 (tokenization, attention, FFN/LayerNorm/block), basic Python
> **Estimated time**: 25 minutes read, 15 minutes code

---

## Why This Matters

You now have all the pieces of a transformer:

- **Tokenization** (Chapter 03): text → integers
- **Embeddings + Positional Encoding** (Chapter 03): integers → vectors with position info
- **Attention** (Chapter 04): tokens share information across positions
- **FFN + LayerNorm + Residuals** (Chapter 05): each token processes independently, stabilized

This chapter **connects everything** into a single, working model that can **learn from data** (training) and **produce new text** (generation). By the end, you'll have a complete, functional transformer that trains on Shakespeare and generates its own "plays."

---

## Part 1: The Complete Model Assembly

### The Pipeline

Here is the full flow, from raw token IDs to output probabilities:

```
Token IDs (B, T)
    │
    ▼
┌─────────────────────┐
│  Token Embedding    │  (vocab_size, d_model) — lookup table
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  + Sinusoidal PE    │  fixed position signal, added to embeddings
└─────────┬───────────┘
          │
          ▼
    ┌─────┴─────┐   ─┐
    │  Block 1  │    │
    ├───────────┤    │  N × VanillaDecoderBlock
    │  Block 2  │    │  (attention + FFN + LayerNorm + residual)
    ├───────────┤    │
    │    ...    │    │
    ├───────────┤    │
    │  Block N  │   ─┘
    └─────┬─────┘
          │
          ▼
┌─────────────────────┐
│  LayerNorm (final)  │  normalize before projection
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Linear Head        │  (d_model, vocab_size) — project to vocabulary space
└─────────┬───────────┘
          │
          ▼
       logits (B, T, vocab_size)
          │
          ▼
       softmax
          │
          ▼
    probabilities      │  ← one distribution per position
```

Each step is a component you've already built — this is the moment they all snap together.

### The Head Layer

The **linear head** is the final projection: `nn.Linear(d_model, vocab_size)`. It takes each token's final representation (a vector of length `d_model`) and maps it to a score for every token in the vocabulary. These scores are called **logits** — they're not normalized yet.

```
For each position:
    representation (d_model)  ─→  scores (vocab_size)  ─→  probabilities

    token's state  →  [head]  →  [2.1, -0.5, 3.7, ...]  →  softmax  →  [0.1, 0.0, 0.8, ...]

    The head shared across ALL positions — same weights apply to every token.
```

> 💡 **Important**: The head is **shared across all positions**. Every token at every position uses the same `Linear(d_model, vocab_size)` projection. This is weight-tying in its simplest form. (Full weight tying — sharing between embedding and head — comes in Phase 2.)

### The Output: Logits

The model outputs **logits**, not probabilities. Why?
1. **Numerical stability**: Computing softmax on raw logits inside the loss function is more stable than computing probabilities first
2. **Flexibility**: During generation, we modify logits (temperature, top-k) before applying softmax

The shape of the output is `(B, T, vocab_size)` — a score for every token in the vocabulary at every position.

---

## Part 2: Weight Initialization

### Why Initialization Matters

When you create a neural network, the weights are random. If they're **too large**, activations explode and neurons saturate (gradients become near-zero). If they're **too small**, activations vanish and signals never propagate.

The goal of initialization is to start the network in a "sweet spot" where:
- Activations have reasonable variance (not 0, not infinity)
- Gradients flow through all layers
- Learning starts immediately

### The Standard Approach: N(0, 0.02)

The standard initialization for transformers is:

```python
nn.init.normal_(weight, mean=0.0, std=0.02)
```

This draws each weight from a normal distribution with mean 0 and standard deviation 0.02.

Why **0.02**? For a typical d_model of 768 (GPT-2 small), `1 / sqrt(d_model) ≈ 0.036`. The 0.02 value is slightly smaller, which provides a conservative starting point that works well across model sizes. It's not mathematically derived — it's a heuristic that empirical testing has validated.

### Weight Decay Separation (Best Practice)

When setting up the optimizer, a standard best practice is to **separate weight decay**: apply it only to matrix weights (parameters with `dim >= 2`) and exclude biases and LayerNorm parameters. This prevents unnecessary regularization of scale/gain parameters:

```python
# Pseudo-code for weight decay separation:
decay_params = [p for p in model.parameters() if p.dim() >= 2]
no_decay_params = [p for p in model.parameters() if p.dim() < 2]

optimizer = torch.optim.AdamW([
    {'params': decay_params, 'weight_decay': 0.1},
    {'params': no_decay_params, 'weight_decay': 0.0},
], lr=3e-4)
```

For simplicity, our Phase 1 code applies weight decay uniformly to all parameters. The separation trick is noted here for when you scale up.

### ⚠️ Zero-Init Output Projections (Modern Practice)

> Since 2025, many implementations **zero-initialize the output projection** of each sub-layer instead of N(0, 0.02). This means each decoder block initially acts as the identity function (since the output projection starts at zero), which dramatically improves training stability. This technique was popularized by the **modded-nanogpt speedrun** (KellerJordan, 2025).
>
> Our Phase 1 code uses uniform N(0, 0.02) for simplicity and compatibility with the 2017 design. You'll see zero-init in Phase 2 when we discuss modern improvements.

---

## Part 3: Loss Computation

### Cross-Entropy Loss

The model outputs **logits** (scores), but we need to measure **how wrong** they are. The standard loss function for language modeling is **cross-entropy**.

How it works:

1. For each position, the model produces logits of shape `(vocab_size,)`
2. The correct next token is known (it's the training label) — let's say token ID 37
3. Cross-entropy compares the predicted distribution (softmax of logits) against the "true" distribution (a one-hot vector with 1 at position 37)

```
Example at position 5:
    Logits:        [2.1, -0.5, 3.7, 1.2, ...]    (vocab_size values)
    Softmax:       [0.08, 0.01, 0.80, 0.03, ...]  (sums to 1.0)
    Correct token: 2  (index 2 in our vocab)
    Loss:          -log(0.80) = 0.22               (lower is better)

If the model were perfectly confident in the wrong token:
    Softmax:       [0.80, 0.01, 0.08, 0.03, ...]
    Loss:          -log(0.08) = 2.52               (much higher!)
```

The loss is **lower when the model assigns high probability to the correct token** and **higher when it assigns low probability**.

### In Code

PyTorch's `F.cross_entropy` combines log-softmax and negative log-likelihood in one numerically stable operation:

```python
loss = F.cross_entropy(
    logits.view(-1, vocab_size),   # (B*T, V) — flatten batch and sequence
    targets.view(-1)               # (B*T,) — flatten to match
)
```

We pass **logits directly** (not probabilities) — `F.cross_entropy` internally applies log-softmax. The `view(-1, V)` reshaping flattens the batch and sequence dimensions so each position is treated as an independent prediction.

> 🧠 **`ignore_index` behavior**: The default `ignore_index` in PyTorch's cross-entropy is `-100` (not `-1`, since `-1` is a valid token index for some tokenizers). For character-level training with no padding, neither is needed — the simple `F.cross_entropy(logits.view(-1, V), targets.view(-1))` suffices.

---

## Part 4: The Training Loop

### The 6-Step Loop

Training follows the same pattern from Phase 0 (Chapter 02), now applied to our transformer:

```
┌─────────────────────────────────────────────────────────────────┐
│                       Training Loop                              │
│                                                                  │
│  1. Sample batch ──→ random sequences from TinyShakespeare      │
│                                                                  │
│  2. Forward pass ──→ logits, loss = model(x, targets)          │
│                                                                  │
│  3. Loss computation ──→ cross-entropy between logits & targets │
│                                                                  │
│  4. Backward pass ──→ loss.backward() computes gradients       │
│                                                                  │
│  5. Gradient clipping ──→ clip_grad_norm_(params, 1.0)         │
│                                                                  │
│  6. Weight update ──→ optimizer.step() adjusts all weights     │
│                                                                  │
│  ── Repeat thousands of times ──                                │
└─────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Walkthrough

**Step 1: Sample a batch.** We randomly select 16 sequences of length 128 from the training data. Each sequence's target is the same sequence shifted by one position (predict the next token):

```python
x = data[i : i+128]      # input:  tokens 0..127
y = data[i+1 : i+129]    # target: tokens 1..128
```

At position 0, the model sees token `i` and must predict token `i+1`. At position 127, it sees token `i+127` and must predict token `i+128`.

**Step 2: Forward pass.** The model processes the batch through all layers — embedding → positional encoding → N blocks → LayerNorm → head — and outputs logits of shape `(16, 128, vocab_size)`. If targets are provided, it also returns the loss.

**Step 3: Loss computation.** Cross-entropy compares logits against targets. The loss is a single scalar — the average over all positions in the batch.

**Step 4: Backward pass.** `loss.backward()` computes gradients for every parameter. These gradients tell us the direction and magnitude to adjust each weight to reduce the loss.

**Step 5: Gradient clipping.** `clip_grad_norm_(params, 1.0)` scales down gradients if their total norm exceeds 1.0. This prevents a single bad batch from causing **exploding gradients** — a phenomenon where gradients grow exponentially, destabilizing training.

```
Before clip: grad_norm = 15.7  ⚠️ exploding!
 After clip: grad_norm =  1.0  ✓ under control
```

**Step 6: Weight update.** The optimizer (AdamW) adjusts each parameter using the (clipped) gradients. The learning rate `lr=3e-4` controls how large each update is.

### Gradient Clipping: Why It Matters

Without gradient clipping, a single outlier batch can send the model into an unrecoverable state. Clipping is a safety net:

- It doesn't change the **direction** of the update, only the **magnitude**
- It prevents catastrophic loss spikes
- It's standard practice in every production training run

> 🧠 **Typical values**: Gradient norm of 1.0 is the most common threshold. Values between 0.5 and 5.0 are used in practice. Too low (0.1) and training slows down; too high (5.0) and it doesn't protect against spikes.

### Training vs Evaluation Mode

During training, the model is in `model.train()` mode, which enables gradient computation. During evaluation (checking validation loss), we use `model.eval()` and `torch.no_grad()` to:
- Skip gradient computation (saves memory and compute)
- Get deterministic predictions (no dropout, no randomness)

```python
model.eval()
with torch.no_grad():
    logits, loss = model(x, targets)    # no gradients computed
model.train()
```

---

## Part 5: Generation

### Autoregressive Generation

After training, the model can generate new text. The process is **autoregressive** — each token is generated one at a time and fed back as input:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Generation Loop                              │
│                                                                  │
│  Start with prompt: "ROMEO:"                                     │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────┐     ┌──────────────┐     ┌──────────┐              │
│  │ Forward  │────→│ Sample next  │────→│ Append   │              │
│  │ pass     │     │ token        │     │ to seq   │              │
│  └─────────┘     └──────────────┘     └──────────┘              │
│       │                                  │                       │
│       └────── (loop back with new token) ┘                       │
│                                                                  │
│  Repeat until we have enough tokens                              │
└─────────────────────────────────────────────────────────────────┘
```

At each step:
1. The model processes the current sequence (prompt + generated tokens so far)
2. We look at the **last token's logits** — these represent the model's prediction for the next token
3. We apply **temperature** and **top-k** to shape the distribution
4. We **sample** one token from the resulting distribution
5. We **append** it to the sequence and repeat

### Temperature

Temperature controls the **randomness** of generation:

```python
logits = logits / temperature
```

- **Low temperature (0.1)**: Logits are scaled up, softmax becomes "sharper" — the model picks the most likely token almost every time. Output is deterministic, repetitive, and conservative.
- **Temperature = 1.0**: No change — the original distribution from training.
- **High temperature (1.5)**: Logits are scaled down, softmax becomes "flatter" — unlikely tokens get a higher chance. Output is creative but risks gibberish.

```
Temperature = 0.2:   [0.00, 0.98, 0.01, 0.00, ...]  ← nearly deterministic
Temperature = 1.0:   [0.08, 0.80, 0.07, 0.05, ...]  ← balanced
Temperature = 2.0:   [0.20, 0.35, 0.22, 0.23, ...]  ← nearly uniform
```

> 🔄 **Why temperature matters**: Temperature is the single most important knob for controlling generation quality. Low values give you focused, coherent output (good for factual answers). High values give you creative, diverse output (good for story generation). The right setting depends entirely on your use case.

> 🧠 **Temperature scaling order**: Temperature scaling happens **BEFORE** truncation (top-k/top-p). Reversing this order (truncate first, then scale) changes the sampling behavior — truncation removes options before scaling dilutes the remaining probabilities. Always scale first, then truncate.

### Top-K Sampling

Top-k further constrains the distribution: we keep only the **top-k highest logits** and set the rest to `-inf` (which becomes 0 after softmax):

```python
v, _ = torch.topk(logits, top_k)
logits[logits < v[:, -1:]] = float('-inf')
probs = F.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

This prevents the model from picking extremely unlikely tokens. A typical value is `top_k=40` — the model picks from the 40 most likely candidates.

### The Sampling Step

After applying temperature and top-k, we **sample** from the resulting probability distribution:

```python
probs = F.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

`torch.multinomial` draws a random sample according to the probabilities. It's not argmax (picking the most likely token) — that would make the output too deterministic and repetitive. Sampling introduces **controlled randomness** that makes generations feel natural.

> 💡 **Argmax vs sampling**: If you always pick the most likely token (greedy decoding), the model tends to repeat phrases and get stuck in loops. Sampling adds variety — the same prompt can produce different outputs each time.

### Practical Sampling Rule

> **Tune temperature OR top-p, not both.** Temperature and top-p (nucleus sampling, a Phase 4 topic) interact in ways that are hard to reason about. Pick one knob, tune it, and leave the other at a neutral setting. Temperature is simpler to start with.

### 🕰️ How Sampling Strategies Evolved

Each generation strategy emerged to fix a specific failure mode of the previous one:

| Year | Method | Key Idea | Failure It Fixed | Introduced By |
|------|--------|----------|-----------------|---------------|
| **2015** | **Temperature scaling** | Divide logits by T to control softmax "sharpness" | Models always picked argmax → repetitive output | Hinton et al. (distillation) — borrowed from physics |
| **2018** | **Top-k sampling** | Keep only the k highest-probability tokens | Temperature can still pick extremely unlikely tokens; top-k cuts the tail | Fan et al. (Hierarchical Neural Story Generation) |
| **2019** | **Top-p / Nucleus** | Keep tokens until cumulative probability ≥ p | Top-k is blind to distribution shape — sometimes k is too many or too few | Holtzman et al. (The Curious Case of Neural Text Degeneration) |
| **2024** | **Min-P** | Dynamic threshold = p_base × top_probability | Top-p is static — same p for confident and uncertain inputs | Nguyen et al. (Min-P Sampling) |

> 🟢 **What we use in Phase 1**: Temperature + top-k. Temperature is the single most important knob. Top-k provides a simple safety guardrail. Top-p is Phase 4 because it requires sorting the full distribution. Min-P is cutting-edge and actively debated — we mention it here so you know it exists.

---

## Part 6: 🟢 Summary Box

```
Each piece you built connects into one pipeline:

  Token IDs → Embed + PE → [Block × N] → LayerNorm → Head → Logits

Training:
  Sample batch → Forward → Loss → Backward → Clip → Update → Repeat

Generation:
  Prompt → Forward → Sample next token → Append → Repeat

Key concepts:
  • Logits = raw scores before softmax
  • Cross-entropy = how wrong the predictions are
  • Temperature = randomness control (low = conservative, high = creative)
  • Top-k = only sample from the k most likely tokens
  • Checkpoint = saved model weights (the result of training)
```

| Term | What It Is |
|------|------------|
| **Logits** | Raw output scores from the head layer, before softmax |
| **Cross-entropy** | Loss function that measures prediction error |
| **Gradient clipping** | Caps gradient norm to prevent exploding gradients |
| **Temperature** | Scales logits to control generation randomness |
| **Top-k** | Keeps only the k highest logits for sampling |
| **Sampling** | Randomly picking a token according to the probability distribution |
| **Autoregressive** | Generating one token at a time, feeding each new token back as input |
| **Checkpoint** | Saved model weights from a training run |

> 💡 **Key insight**: Training and generation are the same forward pass — the only difference is what you do with the output. During training, you compare logits against known targets to compute a loss. During generation, you sample from logits to produce new tokens.

---

## Part 7: 🟢 Check Your Understanding

Test yourself before moving to the next chapter.

1. **What's the difference between training and inference (generation)?**

   <details>
   <summary>Show answer</summary>
   During <strong>training</strong>, the model sees both input tokens and target tokens (the correct next token). It computes a loss (cross-entropy) and updates weights to reduce that loss. The model learns patterns from the data. During <strong>inference/generation</strong>, the model only sees input tokens (a prompt). It predicts the next token, appends it, and repeats. No loss is computed, no weights are updated. The model uses what it learned during training to produce new text.
   </details>

2. **What does temperature=0.1 do vs temperature=1.5?**

   <details>
   <summary>Show answer</summary>
   Temperature controls the "sharpness" of the probability distribution. At <strong>temperature=0.1</strong>, logits are divided by 0.1 (multiplied by 10), making the softmax distribution extremely peaked — the model almost always picks the single most likely token. Output is repetitive and conservative. At <strong>temperature=1.5</strong>, logits are divided by 1.5 (scaled down), making the distribution flatter — unlikely tokens get a much higher chance. Output is diverse and creative but risks being incoherent or gibberish. A temperature around 0.7–1.0 is typical for general-purpose generation.
   </details>

3. **Why do we save checkpoints during training?**

   <details>
   <summary>Show answer</summary>
   Checkpoints save the model's weights at a specific point in training. This is important because (1) training can take hours or days — if it crashes, you don't want to start over; (2) the best model isn't necessarily the final model — validation loss can start increasing (overfitting) even as training loss continues to decrease; (3) you can use the checkpoint later for generation without retraining. We save whenever validation loss improves, ensuring we keep the best model.
   </details>

4. **What's the shape of the final linear head layer?**

   <details>
   <summary>Show answer</summary>
   The head layer has shape <code>(d_model, vocab_size)</code>. For our TinyShakespeare model: d_model = 128, vocab_size = 65, so the head is a <code>(128, 65)</code> matrix. It projects each token's final representation (a 128-dimensional vector) to a score for every token in the vocabulary (65 scores). The output shape after the head is <code>(B, T, vocab_size)</code> — for a batch of 16 sequences of length 128, that's <code>(16, 128, 65)</code> logits.
   </details>

---

## Part 8: 🔵 Companion Code — The Shared Module

The file `code/vanilla_transformer.py` is the **shared module** that contains all three classes we've built across Chapters 04–06:

1. **`CausalMultiHeadAttention`** — multi-head causal self-attention (Chapter 04)
2. **`VanillaDecoderBlock`** — one decoder block: attention + FFN + LayerNorm + residuals (Chapter 05)
3. **`VanillaDecoderOnlyTransformer`** — the full model: embeddings → N blocks → head (this chapter)

It's designed to be:
- **Importable**: `from code.vanilla_transformer import VanillaDecoderOnlyTransformer`
- **Standalone**: Run directly to verify the model works: `python3 code/vanilla_transformer.py`

### `VanillaDecoderOnlyTransformer` in Detail

```python
class VanillaDecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size=65, d_model=128, n_heads=4, n_layers=4, max_seq_len=256):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # Sinusoidal positional encoding
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
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embed(idx) + self.pe[:, :T, :]
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_final(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss
```

Key design decisions:
- **`register_buffer`** for positional encodings: saved with the model but not trained
- **`targets=None`** by default: allows the same code path for both training and inference
- **`_init_weights`** applies N(0, 0.02) only to dim ≥ 2 parameters (biases and LayerNorm gains keep default init)
- **No `nn.Sequential` for the blocks**: `nn.ModuleList` + explicit loop keeps it clear that blocks run sequentially

### Run the Shared Module

```bash
python3 code/vanilla_transformer.py
```

**Expected output:**
```
Logits shape: torch.Size([2, 32, 65])  (expected: [2, 32, 65])
Loss (no targets): None
Loss (with targets): 4.2035
Total parameters: 807,936
✓ VanillaDecoderOnlyTransformer works correctly
```

The loss of ~4.2 is the initial cross-entropy for random weights (`ln(65) ≈ 4.17`). After training, it should drop significantly.

---

## Part 9: 🔵 Companion Code — Training Loop

File: `code/06-training-generation/01_training_loop.py`

This script:
1. Downloads TinyShakespeare from the Karpathy repository (with error handling)
2. Builds a character-level vocabulary (65 unique characters)
3. Creates training (90%) and validation (10%) splits
4. Trains a `VanillaDecoderOnlyTransformer` for 1000 steps
5. Evaluates every 100 steps and saves the best checkpoint

### Key Training Loop Code

```python
for step in range(1000):
    # Evaluation every 100 steps
    if step % eval_interval == 0:
        model.eval()
        # Compute train and val loss over multiple batches
        for split_name, split_data in [('train', train_data), ('val', val_data)]:
            losses_split = torch.zeros(eval_iters)
            for k in range(eval_iters):
                x, y = get_batch(split_data, batch_size, block_size)
                with torch.no_grad():
                    _, loss = model(x, y)
                losses_split[k] = loss.item()
            losses[split_name] = losses_split.mean().item()

        print(f"Step {step:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

        # Save if best
        if losses['val'] < best_loss:
            best_loss = losses['val']
            torch.save(model.state_dict(), "checkpoints/vanilla_best.pt")

        model.train()

    # Training step (always runs)
    x, y = get_batch(train_data, batch_size, block_size)
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

> 💡 The `get_batch` function randomly samples contiguous blocks from the data. This is simpler than a proper DataLoader but works fine for small-scale training.

### Run the Training Loop

```bash
python3 code/06-training-generation/01_training_loop.py
```

**Expected behavior:**
- Downloads TinyShakespeare on first run (~1 MB)
- Starts training, printing loss every 100 steps
- Loss should decrease over time (e.g., from ~4.2 to ~2.5-3.0 on CPU in 1000 steps)
- Best model saved to `checkpoints/vanilla_best.pt`

> ⏱ **CPU training time**: ~5-10 minutes for 1000 steps. The model has ~800K parameters and processes batches of 16 × 128 = 2,048 tokens per step.

### Understanding the Optimizer Setup

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
```

### 🕰️ How Training Optimizers Evolved

The optimizer — the algorithm that actually updates weights — has been one of the most researched areas in LLM training:

| Year | Optimizer | Key Innovation | Used For | Why |
|------|-----------|---------------|----------|-----|
| **2017** | **Adam** (`β₂=0.98`, `ε=1e-9`) | Per-parameter adaptive LR + momentum | Original Transformer (Vaswani et al.) | The first optimizer that "just worked" for transformers; no LR schedule tuning needed beyond warmup |
| **2018** | **Adam** (`β₂=0.999`, `ε=1e-8`) | Changed β₂ from 0.98 to default | BERT, GPT-2 (Devlin, Brown) | Higher β₂ means longer memory of gradient history — better for larger batches |
| **2019** | **AdamW** | Decoupled weight decay from gradient update | GPT-3, most modern models (Loshchilov & Hutter) | Weight decay no longer interferes with Adam's adaptive LR — cleaner regularization |
| **2019** | **LAMB** | Layer-wise Adaptive Moments (adjusts LR per layer) | Large-batch pretraining (You et al., Google) | Scales batch sizes to 64K+ without destabilizing; LR grows with batch |
| **2022** | **8-bit AdamW** | Quantizes optimizer states to 8-bit | Memory-constrained training (Dettmers et al.) | Cuts optimizer memory 4× — a 7B model's Adam states go from ~56GB to ~14GB |
| **2023** | **Lion** | Sign-based updates (no momentum buffers needed) | Google internal models | 2× less memory than Adam (no momentum squared buffer); sometimes faster convergence |
| **2024** | **Muon** | Orthogonalized momentum (Newton-Schulz on gradients) | modded-nanogpt speedrun (KellerJordan) | Treats gradient matrix as a "rotation" problem; 2× faster wall-clock convergence than AdamW for transformers |
| **2024–25** | **C-AdamW** (Cautious) | One-line code change: restrict momentum to positive updates | Any AdamW training (Cautious Optimizers paper) | Up to 1.47× speedup on major pretraining tasks; preserves convergence guarantees |
| **2025** | **VSGD** | Treats gradients as latent variables (variational inference) | Advanced research (VSGD paper) | Extra accuracy with comparable computational cost; still experimental |

> 🟢 **Focus for Phase 1**: We use plain **AdamW with lr=3e-4**. It's the safest, best-understood default. The table shows where optimizer research is heading — each row trades off one axis (memory, speed, batch size, simplicity) for another. You don't need to memorize them, but knowing they exist helps you make informed choices when you scale up.

> 💡 **Note on PEFT/LoRA**: The row for **PEFT (LoRA/QLoRA)** was removed from the optimizer table because it's not an optimizer — it's a **parameter-efficient fine-tuning framework**. LoRA freezes the base model and trains small adapter matrices (typically <1% of parameters). The optimizer (AdamW, etc.) still runs, but only on the adapter weights. This is a Phase 4+ topic.

For Phase 1, **AdamW with `lr=3e-4`** is the safest choice. AdamW is well-understood, stable, and has excellent default hyperparameters. Muon is exciting but requires more tuning — we cover it in Phase 4 as an advanced optimization topic.

The learning rate `3e-4` is a standard starting point. Higher rates can speed up training but risk instability; lower rates are safer but slower.

> 🕰️ **Learning Rate Schedules**: The original 2017 transformer used a specific "Noam" schedule — linear warmup for 4000 steps, then inverse square root decay. GPT-2 used cosine decay with warmup. Modern training often uses cosine decay with warmup (e.g., 1000 warmup steps, then cosine to 10% of peak LR). **We use a constant LR for simplicity** in Phase 1, but production training requires a schedule to stabilize early training and converge properly.

---

## Part 10: 🔵 Companion Code — Generation

File: `code/06-training-generation/02_generation.py`

This script:
1. Loads the checkpoint saved by the training loop
2. Takes a prompt ("ROMEO:")
3. Generates 100 new tokens using temperature and top-k sampling

### Key Generation Code

```python
def generate(model, prompt_ids, max_new_tokens=200, temperature=1.0, top_k=40):
    model.eval()
    idx = prompt_ids
    for _ in range(max_new_tokens):
        # Crop to context window
        idx_cond = idx[:, -model.max_seq_len:]

        # Forward pass (no gradient)
        with torch.no_grad():
            logits, _ = model(idx_cond)

        # Temperature + top-k + sample
        logits = logits[:, -1, :] / temperature          # scale
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = float('-inf')   # truncate
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_token], dim=1)        # append
    return idx
```

### Run the Generation Script

```bash
python3 code/06-training-generation/02_generation.py
```

**Expected output (without training):**
```
No checkpoint found. Run the training loop first to create checkpoints/vanilla_best.pt.
```

This is correct — the checkpoint doesn't exist yet. After training, it will produce Shakespeare-like text:

```
Prompt: ROMEO:
--------------------------------------------------
ROMEO: I have a dream that the king of the fairies
Shall be the first to see the sun...
--------------------------------------------------

Generated 100 tokens
```

(Sample output — your results will differ since the model is small and under-trained.)

---

## Part 11: Common Beginner Mistakes (Reference)

> **⚠️ Forgetting `model.eval()` during evaluation**: If you evaluate loss without `model.eval()` and `torch.no_grad()`, the evaluation itself takes gradients, which uses extra memory and may give wrong results if dropout/batch norm are present. Always wrap evaluation in `with torch.no_grad():`.

> **⚠️ No gradient clipping**: Without `clip_grad_norm_`, a single high-loss batch can cause gradients to explode, pushing the model into an unrecoverable state. If you see loss go to `NaN`, gradient clipping (or a lower learning rate) is usually the fix.

> **⚠️ Cropping the context window during generation**: The model has a fixed `max_seq_len`. If you let the generated sequence grow beyond this, the positional encoding runs out. Always crop to the last `max_seq_len` tokens before each forward pass during generation.

> **⚠️ Temperature with top-k order**: Temperature scaling must come BEFORE top-k truncation. If you truncate first and then apply temperature, the logits of the remaining tokens are still scaled, but you've already eliminated potentially valid options that would have been competitive after scaling.

> **⚠️ `multinomial` expects probabilities, not logits**: `torch.multinomial` takes a probability distribution (must sum to 1), not raw logits. Always apply `softmax` before `multinomial`. This is the most common sampling bug.

> **⚠️ Optimizer weight decay**: By default, `AdamW` applies weight decay to all parameters. Best practice is to exclude biases and LayerNorm gains (dim < 2) from weight decay. For our small model it barely matters, but when you scale up (Phase 2+), separating weight decay becomes important.

---

## Part 12: Further Learning

- **Companion Code**: `code/vanilla_transformer.py` (shared module), `code/06-training-generation/01_training_loop.py` (training), `code/06-training-generation/02_generation.py` (generation)
- **Next Chapter**: Chapter 07 — Pre-Norm, RMSNorm & SwiGLU. *Modern upgrades to the vanilla block that enable deeper, more stable training.*
- For a deep dive into efficient training and modern initialization, see the **modded-nanogpt speedrun** (KellerJordan, 2025–2026) — trains a GPT-2 124M model to 3.28 loss in ~90 seconds. This uses many of the techniques mentioned in this chapter (zero-init, weight decay separation, gradient clipping).
- For production training, the **NanoGPT** repository (Karpathy) provides a well-optimized reference implementation.
- For interactive visualization of training dynamics, see **Weights & Biases** or **TensorBoard** — both can plot loss curves in real time.

---

## Terms Introduced

| Term | Quick Definition |
|------|------------------|
| **Logits** | Raw output scores from the model before softmax |
| **Cross-entropy** | Loss function measuring the difference between predicted and true distributions |
| **Gradient clipping** | Capping gradient norm to prevent exploding gradients |
| **Temperature** | Scaling factor on logits that controls generation randomness |
| **Top-k** | Only consider the k most likely tokens during sampling |
| **Autoregressive** | Generating one token at a time, using previous outputs as input |
| **Checkpoint** | Saved model parameters from a specific training step |
| **AdamW** | Optimizer with adaptive learning rates and decoupled weight decay |
| **Weight decay** | Regularization that penalizes large weights (L2 penalty) |

---

> **Next Chapter**: Chapter 07 — Pre-Norm, RMSNorm & SwiGLU.
>
> *You've built a complete 2017-style transformer. Now it's time to upgrade each component with the improvements that power modern LLMs — starting with the block architecture itself.*
>
> *🔵 Make sure the shared module (`code/vanilla_transformer.py`) runs cleanly before proceeding. If you have time, let the training loop run for a few hundred steps to see the loss decrease.*
