# From Zero to LLM — A Visual Curriculum

> **Teach your whole team how transformers work, from zero to attention.**

This curriculum takes a mixed-skill team (engineers, PMs, QAs, business analysts) with **no ML or PyTorch experience** through the full journey: from tensors and matrix multiply to a working decoder-only transformer that trains on Shakespeare and generates text.

**Two tracks** — same concepts, different depth.

---

## Who This Is For

| Role | Track | Outcome |
|------|-------|---------|
| Product Managers, BAs, QAs | 🟢 Conceptual (~2 hrs) | Participate in architecture discussions, read technical docs, understand what the team builds |
| R&D Engineers | 🔵 Engineer (~8 hrs) | Build every component from scratch, understand every line of code, extend the model |

---

## The Philosophy

**Vanilla-first progression.** We start with the original 2017 transformer (post-norm, ReLU, sinusoidal PE) — the same architecture from the "Attention Is All You Need" paper. Then we upgrade piece-by-piece in later phases: pre-norm, RMSNorm, SwiGLU, RoPE, GQA, FlashAttention, Mamba hybrids.

Every chapter has:
- **Narrative** — story-first explanations with diagrams
- **🎯 Evolution boxes** — how each component evolved: 2017 → 2019 → 2023 → 2025, with paper citations and model adoption
- **🔵 Companion code** (Engineer track) — self-contained Python scripts that actually run
- **🟢 Check Your Understanding** — quick quiz for all roles

---

## Structure

```
.
├── curriculum/                    # Guide chapters (Markdown)
│   ├── index.md                   # Roadmap and two-track guide
│   ├── 00-how-llms-work.md        # The big picture
│   ├── 01-tensors-and-matmul.md   # Math engine
│   ├── 02-neural-nets-and-training.md
│   ├── 03-tokenization-embeddings-pe.md   ◄─ Phase 1
│   ├── 04-attention-mechanism.md          ◄─ Phase 1
│   ├── 05-ffn-layernorm-stack.md          ◄─ Phase 1
│   └── 06-training-loop-and-generation.md ◄─ Phase 1
│
├── code/                          # Companion Python scripts
│   ├── 01-tensors/
│   ├── 02-neural-nets/
│   ├── 03-tokenization-embeddings-pe/
│   ├── 04-attention/
│   ├── 05-ffn-layernorm/
│   ├── 06-training-generation/
│   └── vanilla_transformer.py     # Shared module (imported by Phase 1 scripts)
│
├── reference/
│   └── glossary.md                # 35+ terms with chapter cross-refs
│
├── DELIVERY_GUIDE.md              # Facilitator notes, session plans, mixed-room strategy
├── requirements.txt               # Only torch>=2.0.0
└── README.md                      # This file
```

### Phases

| Phase | Content | Status |
|-------|---------|--------|
| **0** – Foundations (Tensors, Neural Nets, Training) | 3 chapters + 5 scripts | ✅ Complete |
| **1** – The Vanilla Transformer (Tokenization → Generation) | 4 chapters + 10 scripts | ✅ Complete |
| **2** – Modern Upgrades (Pre-norm → RMSNorm → SwiGLU → RoPE → GQA) | Planned | ⬜ |
| **3** – Training Enhancements (MTP, Distillation) | Planned | ⬜ |
| **4** – Production (FlashAttention, vLLM, Quantization) | Planned | ⬜ |
| **5** – Beyond Transformers (Mamba, SSMs, Hybrids) | Planned | ⬜ |
| **6** – Fintech Considerations | Planned | ⬜ |

---

## Getting Started (Humans)

### Prerequisites
- Python 3.10+
- pip (for PyTorch installation)

### Install & Run

```bash
# Install PyTorch (CPU-only is fine)
pip install torch>=2.0.0
# or
pip install -r requirements.txt

# Verify the shared model works
python3 code/vanilla_transformer.py

# Run any companion script — they're all self-contained:
python3 code/03-tokenization-embeddings-pe/01_char_tokenizer.py
python3 code/04-attention/02_multi_head_causal.py
python3 code/05-ffn-layernorm/02_vanilla_decoder_block.py

# Train on TinyShakespeare (CPU: ~5-10 min for 1000 steps)
python3 code/06-training-generation/01_training_loop.py

# Generate text with the trained model
python3 code/06-training-generation/02_generation.py
```

### Read the Curriculum

Start with `curriculum/00-how-llms-work.md` (10 minutes, no code). Then follow your track:

- **🟢 Conceptual**: 00 → 01 → 03 → 04 → 12 (Phase 6) → 11 (Phase 5)
- **🔵 Engineer**: 00 → 01 → 02 → 03 → 04 → 05 → 06 → Phases 2+

---

## Agent Context

This section is for **AI agents** (code assistants, reviewers, automation) reading this repository.

### Architecture

This is an **educational curriculum**, not a production codebase. The companion code prioritizes readability and pedagogical clarity over performance.

### Key Design Decisions

| Decision | Rationale | Phase to Upgrade |
|----------|-----------|------------------|
| **Post-norm** (LayerNorm after residual) | Original 2017 layout; shallow stacks (<12L) | Phase 2 → Pre-norm |
| **ReLU activation** | Simplest non-linearity | Phase 2 → SwiGLU |
| **Sinusoidal PE** | Fixed, deterministic, no learned params | Phase 2 → RoPE |
| **MHA** (separate Q/K/V per head) | Original multi-head design | Phase 2 → GQA |
| **Character-level tokenization** | 65 tokens, easy to debug | Phase 2+ → BPE |
| **Post-norm** | Original 2017 block layout | Phase 2 → Pre-norm + RMSNorm |
| **AdamW** optimizer | Standard, well-understood | Phase 4 → Muon |

### Files That Agents Should Read First

| File | Why |
|------|-----|
| `docs/superpowers/plans/2026-06-19-llm-curriculum.md` | Full implementation plan with all 6 phases |
| `curriculum/index.md` | Roadmap and track guide |
| `DELIVERY_GUIDE.md` | Session plans, facilitator notes, mixed-room strategy |
| `reference/glossary.md` | 35+ terms defined |
| `code/vanilla_transformer.py` | Shared module — canonical vanilla transformer implementation |

### Companion Code Verification

All companion scripts were verified working on Python 3.12 / CPU-only (no CUDA):

```
code/vanilla_transformer.py                  → Logits [2, 32, 65], 807,936 params
code/06-training-generation/01_training_loop.py → Step 0 train loss ~4.18 (TinyShakespeare)
code/06-training-generation/02_generation.py   → Generates 100 tokens from checkpoint
code/05-ffn-layernorm/02_vanilla_decoder_block.py → 3,216 params per block, gradient flow OK
code/04-attention/02_multi_head_causal.py      → Multi-head causal attention works
```

### Known Limitations

- **CPU only** — no CUDA/GPU support (AMD RX 6600 in WSL, pending ROCm)
- **Code directory vs stdlib** — Python stdlib has a `code` module; companion scripts use `sys.path` insert to resolve
- **TinyShakespeare download** — first run downloads ~1MB from karpathy/char-rnn (retry logic with 3 attempts)
- Checkpoints saved to `checkpoints/vanilla_best.pt`

---

## License

Internal team educational use. Not published.
