# From Zero to LLM: A Visual Curriculum

> **Audience**: Engineers, PMs, QAs, and BAs — no ML or PyTorch experience required.
> **Mode**: Read the story → Follow the diagrams → Run the code (optional for your role).

---

## Two Tracks

This curriculum has two paths. Pick yours:

### 🟢 Conceptual Track — All Roles (~2 hours total)

Understand *what* an LLM is and *how* it works, without writing code.

| # | Chapter | Time | What You'll Learn |
|---|---------|------|-------------------|
| 00 | [How LLMs Work](00-how-llms-work.md) | 10 min | The big picture: text in → prediction → text out |
| 01 | [Tensors & Matrix Multiply](01-tensors-and-matmul.md) | 15 min | The math engine behind every neural network |
| 03 | [Tokenization, Embeddings, Positional Encoding](03-tokenization-embeddings-pe.md) *(Phase 1)* | 15 min | How words become numbers the model can process |
| 04 | [Attention Mechanism](04-attention-mechanism.md) *(Phase 1)* | 20 min | The "secret sauce" — how models understand context |
| 12 | [Fintech Considerations](12-fintech-considerations.md) *(Phase 6)* | 15 min | Evaluation, privacy, guardrails, cost |
| 11 | [Beyond Transformers: Mamba & Hybrids](11-mamba-hybrids.md) *(Phase 5)* | 10 min | What's next for LLM architecture |

**You'll walk away with**: The confidence to participate in architecture discussions, read technical documents, and understand what your engineering team is building.

### 🔵 Engineer Track — All of the Above + Code (~8 hours total)

Understand AND build. You'll write every component from scratch.

| # | Chapter | Time | Code |
|---|---------|------|------|
| 00 | [How LLMs Work](00-how-llms-work.md) | 10 min | — |
| 01 | [Tensors & Matrix Multiply](01-tensors-and-matmul.md) | 30 min | `code/01-tensors/` |
| 02 | [Neural Nets & Training](02-neural-nets-and-training.md) | 45 min | `code/02-neural-nets/` |
| 03–12 | Full architecture progression (Phase 1–5) | ~6 hrs | `code/03-tokenization-embeddings-pe/`, `code/04-attention/`, `code/05-ffn-layernorm/`, `code/06-training-generation/` |

**You'll walk away with**: A working decoder-only transformer you built yourself, understanding of every line, and the ability to extend it.

---

## How Each Chapter Is Structured

```
┌─────────────────────────────────┐
│ Prerequisites                   │  ← What you need before starting
│ Estimated time: 15 min read     │
├─────────────────────────────────┤
│ Story                           │  ← The concept, explained visually
│   · Diagrams (Mermaid)          │
│   · Intuition                   │
│   · Why it matters for LLMs     │
├─────────────────────────────────┤
│ 🟢 For Everyone: Summary        │  ← If you only read one section
├─────────────────────────────────┤
│ 🔵 For Engineers: How It Works  │  ← Code, shapes, implementation
│   · Companion: code/XX/         │
│   · Exercise: modify and run    │
├─────────────────────────────────┤
│ Key Takeaways                   │
│ Glossary terms introduced       │
└─────────────────────────────────┘
```

---

## The Full Roadmap

```
Phase 0 — Foundations (you are here)
├── 00: How LLMs Work               🟢 Everyone
├── 01: Tensors & Matrix Multiply   🟢 Everyone
└── 02: Neural Nets & Training      🔵 Engineers

Phase 1 — The Vanilla Transformer
├── 03: Tokenization, Embeddings, PE        🟢 Everyone
├── 04: Self-Attention & Multi-Head         🟢 Everyone
├── 05: FFN, LayerNorm, Residuals           🟢 Everyone (code: 🔵 Engineers)
└── 06: Training Loop & Generation          🟢 Everyone (code: 🔵 Engineers)

Phase 2 — Modern Upgrades
├── 07: Pre-Norm → RMSNorm → SwiGLU         🔵 Engineers
└── 08: RoPE → GQA → Weight Tying            🔵 Engineers

Phase 3 — Training Enhancements
└── 09: MTP & Distillation                   🔵 Engineers

Phase 4 — Production
└── 10: FlashAttention, vLLM, Quantization   🔵 Engineers

Phase 5 — Beyond Transformers
└── 11: Mamba, SSMs, Hybrids                🟢 Everyone

Fintech Cross-Cut
└── 12: Fintech Considerations               🟢 Everyone
```

---

## Quick Glossary

Don't know a term? Check the [reference/glossary.md](reference/glossary.md) — but here are the most common ones you'll encounter:

| Term | One-Liner |
|------|-----------|
| **Tensor** | A multi-dimensional array (like a spreadsheet on steroids) |
| **Matrix Multiply** | The operation that powers every neural network layer |
| **Attention** | The mechanism that lets a model focus on relevant parts of input |
| **Loss** | A number that tells us how wrong the model's prediction is |
| **Gradient** | The direction to adjust weights to reduce the loss |
| **Training** | The loop that gradually reduces loss over millions of steps |
| **Inference** | Using a trained model to generate predictions |

---

> **Next**: Start with [Chapter 00: How LLMs Work](00-how-llms-work.md) — 10 minutes, no code, just the big picture.
