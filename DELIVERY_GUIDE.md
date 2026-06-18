# Delivery Guide: How to Teach This Curriculum

> **Audience**: You (the facilitator). This guide helps you deliver the curriculum to your mixed-skill team.
> **Goal**: Every participant walks away understanding LLMs at their role-appropriate depth.

---

## Session Structure

The curriculum has two tracks. Here's how to schedule them:

### 🟢 Conceptual Track — All Roles

Deliver this first to the whole team. Engineers attend but the pace targets non-engineers.

| Session | Chapters | Duration | Format |
|---------|----------|----------|--------|
| 1 | 00 (How LLMs Work), 01 (Tensors) | 90 min | Live walkthrough + discussion |
| 2 | 03 (Tokenization), 04 (Attention) | 90 min | Live walkthrough + Q&A |
| 3 | 05 (FFN/Block), 06 (Training/Generation) | 90 min | Live walkthrough + demo |
| 4 | 12 (Fintech Considerations) | 60 min | Workshop + discussion |

**Total**: ~5 hours over 4 sessions. Schedule 1–2 per week.

### 🔵 Engineer Track — Engineers Only

After each 🟢 session, engineers work through the companion code independently.

| Session | Chapters | Code | Expected Time |
|---------|----------|------|---------------|
| (after Session 1) | 01 | `code/01-tensors/` | 30 min |
| (after Session 2) | 03, 04 | `code/03-, 04-` | 60 min |
| (after Session 3) | 05, 06 | `code/05-, 06-` | 90 min |
| Follow-up | 07–11 | `code/07-11/` | Ongoing (~4 hrs total) |

> **Tip**: Pair engineers during code sessions. Learning is faster and questions get answered in real time.

---

## Facilitator Notes Per Chapter

### Chapter 00: How LLMs Work

- **Stuck point**: People confuse "token" with "word". Emphasize: "A token is whatever the tokenizer says it is — could be a word, part of a word, or a character."
- **Demo**: Open your browser, go to ChatGPT or Claude, type "The cat sat on the __" — ask "what do you expect?" Then explain that this is exactly what your model will do, just much smaller.
- **Key question to ask**: "What just happened when the model generated that word?" Answer: "It computed probabilities for every word in its vocabulary and picked one."

### Chapter 01: Tensors & Matrix Multiply

- **🟢 All**: Focus on the B, T, C shape diagram. Don't show the math formula for non-engineers. Use the spreadsheet analogy.
- **Stuck point**: "Why three dimensions?" Show: B = stack of spreadsheets, T = rows, C = columns.
- **Demo**: Open Python REPL, create `torch.randn(2, 3, 4)`, ask "how many numbers is that?" (24). Then ask "what does each dimension mean?"

### Chapter 02: Neural Nets & Training

- **🟢 All**: Focus on the training loop diagram (forward → loss → backward → step). The "mixing board" analogy for backpropagation is your safest explanation.
- **🔵 Engineers**: Run the companion code live. Show `loss.item()` decreasing after each epoch.
- **Stuck point**: "What's a gradient?" Answer: "The direction to nudge a weight to reduce the error. Like adjusting a dial until the sound quality improves."
- **Common question**: "Why not just use a lookup table?" Good segue to "generalization" — the model learns patterns it can apply to unseen data.

### Chapter 03: Tokenization, Embeddings, PE

- **🟢 All**: Make sure everyone understands the flow: text → token IDs → embedding vectors → +PE → transformer. Draw this on a whiteboard.
- **Stuck point**: "How does the model know the meaning of each token?" It doesn't initially — embeddings are random and learned during training.
- **Demo**: Encode a sentence, show the token IDs, decode back. Let someone in the room choose a sentence.

### Chapter 04: Attention

- **🟢 All**: Use the library analogy throughout. "A query searches the catalog, keys are the book labels, values are the book contents."
- **Stuck point**: "Why multiple heads?" Use: "One head looks at syntax (what follows what), another at semantics (what means what), another at position (where things are)."
- **🔵 Engineers**: Skip to the code. The shape transformations (BTHD vs BTCD) are the hardest part — walk through them slowly.
- **⚠️ Warning**: This is the hardest chapter for everyone. Budget extra time.

### Chapter 05: FFN, LayerNorm, Residuals

- **🟢 All**: The "team meeting vs individual work" analogy for attention vs FFN is your hook. Everyone understands this.
- **Stuck point**: "Why do we need LayerNorm?" Analogy: "It's like temperature control — keeps things from getting too hot or too cold as they pass through many layers."
- **🔵 Engineers**: The post-norm vs pre-norm distinction is important to highlight even now. "Everything will flip in Phase 2 — you'll understand why then."

### Chapter 06: Training & Generation

- **🟢 All**: Focus on the temperature slider. Demo: run generation with temperature=0.1 (safe, repetitive) vs temperature=1.0 (creative, chaotic).
- **Demo highlight**: Run the trained model and show Shakespeare-like output. Even if garbled, people are amazed it generates *anything* coherent.
- **Stuck point**: "Why doesn't the model just repeat what it saw in training?" Because it learns patterns, not memorized text — it's like a jazz musician who's learned a style but can improvise.

*[Additional chapters added as curriculum grows.]*

---

## Mixed-Room Strategy

When engineers and non-engineers are in the same session:

1. **Start together** — Read the 🟢 section aloud or as a group. No one should feel left behind.
2. **Split for 🔵 content** — Engineers move to code exercises. Non-engineers do a discussion/reflection prompt.
3. **Reconvene** — Everyone shares one thing they learned.

**Example 45-min session plan**:
| Time | Activity | Who |
|------|----------|-----|
| 0–10 | Read 🟢 section together | Everyone |
| 10–15 | Key diagram walkthrough | Facilitator |
| 15–30 | Split: Engineers run code, others discuss reflection prompts | Split |
| 30–40 | Share takeaways | Everyone |
| 40–45 | Preview next chapter | Facilitator |

---

## Success Criteria Per Role

At the end of the curriculum, each role should be able to:

### PM / BA
- [ ] Explain in one sentence what an LLM does (predicts the next token)
- [ ] Draw the high-level flow: text → tokens → model → probabilities → text
- [ ] Explain attention as "the model looking at relevant context"
- [ ] Understand why bigger models need more data and compute
- [ ] Explain the difference between training and inference
- [ ] Participate in architecture discussions (e.g., "why do we need a larger context window?")

### QA
- [ ] All of the above
- [ ] Understand what kinds of inputs cause failure modes (prompt injection, hallucination)
- [ ] Generate test cases for model behavior, not just output correctness
- [ ] Explain the difference between deterministic (low temperature) and creative (high temperature) generation
- [ ] Understand why evaluation metrics differ from traditional software (perplexity vs test pass rate)

### Engineer
- [ ] All of the above
- [ ] Build a decoder-only transformer from scratch
- [ ] Explain the role of every component: embedding → attention → FFN → LayerNorm → head
- [ ] Modify model architecture (add layers, change head count)
- [ ] Train a model and interpret loss curves
- [ ] Explain upgrade paths: post-norm → pre-norm, ReLU → SwiGLU, sinusoidal → RoPE
- [ ] Read and discuss modern architecture papers at the level of the Mamba or Llama 3 paper

---

## Common Questions (and Answers)

> "Why PyTorch and not TensorFlow?"
> PyTorch is the dominant framework for LLM research. It's easier to debug, more readable, and most papers use it.

> "When will we build something we can use in production?"
> Phase 4 covers production deployment (vLLM, quantization). But the goal is understanding, not production-ready code.

> "Do I need a GPU?"
> Not for this curriculum. Your tiny models (5M params) train in minutes on CPU. The concepts are identical to GPU training.

> "How is this different from watching a YouTube tutorial?"
> You build every component from scratch. By the end, you'll understand every line of code, not just the high-level API.

> "When do we get to ChatGPT-level models?"
> We don't — those cost millions to train. But you'll understand exactly how they work under the hood, and your toy model uses the same architecture.

---

## Troubleshooting Common Problems

### "The training loss isn't decreasing"
- Check learning rate (too high → oscillates, too low → barely moves)
- Check data: is there a real pattern to learn?
- Check model size: too small → can't capture the pattern

### "The generated text is garbage"
- Normal for a tiny model! 5M params can't produce Shakespeare. The goal is seeing *any* structure (word lengths, common letters).
- Lower temperature = more predictable (but boring) output

### "PyTorch isn't installed / wrong version"
- Run `pip install torch` (CPU-only is fine — no CUDA needed)
- If on Windows WSL: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### "The Mermaid diagrams don't render in VS Code"
- Install the "Markdown Preview Mermaid Support" extension
- Or open in a browser-based Markdown viewer (Typora, Obsidian)

---

> **Remember**: Your participants don't need to understand everything. The 🟢 track gives them the confidence to participate in conversations about LLMs. That's the win.
