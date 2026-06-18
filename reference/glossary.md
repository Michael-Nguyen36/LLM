# Glossary

> Quick one-sentence definitions. Terms link to the chapter where they're introduced.

| Term | One-Liner | Introduced In |
|------|-----------|---------------|
| **Attention** | A mechanism that lets each token "look at" all other tokens and decide which ones matter for its own meaning. | [Chapter 04](../curriculum/04-attention-mechanism.md) |
| **Autoregressive** | A model that predicts the next token one at a time, feeding its own output back as input. | [Chapter 00](../curriculum/00-how-llms-work.md) |
| **Backpropagation** | The process of computing gradients backwards through a network, telling each weight how to adjust. | [Chapter 02](../curriculum/02-neural-nets-and-training.md) |
| **Batch** | A group of training examples processed together. Shape `(B, T, C)` — the B dimension. | [Chapter 01](../curriculum/01-tensors-and-matmul.md) |
| **Block** | A single transformer layer: attention → FFN → add → norm (repeated N times). | [Chapter 05](../curriculum/05-ffn-layernorm-stack.md) |
| **Causal Mask** | A triangular mask that prevents tokens from attending to future tokens during training. | [Chapter 04](../curriculum/04-attention-mechanism.md) |
| **Channel** | The feature dimension of a token's representation. Shape `(B, T, C)` — the C dimension. | [Chapter 01](../curriculum/01-tensors-and-matmul.md) |
| **Checkpoint** | A saved snapshot of model weights, allowing training to resume or be loaded for inference. | [Chapter 06](../curriculum/06-training-loop-and-generation.md) |
| **Cross-Entropy Loss** | The standard loss function for classification: measures how different the predicted probability distribution is from the correct answer. | [Chapter 02](../curriculum/02-neural-nets-and-training.md) |
| **Embedding** | A learned vector representation of a token. Each token ID maps to one row in the embedding table. | [Chapter 03](../curriculum/03-tokenization-embeddings-pe.md) |
| **FFN** | Feed-Forward Network — processes each token independently after attention mixes information across tokens. | [Chapter 05](../curriculum/05-ffn-layernorm-stack.md) |
| **FlashAttention** | An IO-aware attention algorithm that tiles the computation to fit in fast SRAM, reducing memory reads/writes. | [Chapter 10](../curriculum/10-flashattention-vllm.md) |
| **Gradient** | A vector pointing in the direction of steepest increase of the loss. We move opposite to it to reduce loss. | [Chapter 02](../curriculum/02-neural-nets-and-training.md) |
| **Head** | The final linear layer that projects from `d_model` to `vocab_size` to produce token probabilities. | [Chapter 06](../curriculum/06-training-loop-and-generation.md) |
| **Inference** | Using a trained model to generate predictions on new data. | [Chapter 00](../curriculum/00-how-llms-work.md) |
| **KV Cache** | A cache of Key and Value tensors from previous tokens, avoiding recomputation during autoregressive generation. | [Chapter 10](../curriculum/10-flashattention-vllm.md) |
| **LayerNorm** | Normalizes activations across the feature dimension: stabilizes training by keeping values in a consistent range. | [Chapter 05](../curriculum/05-ffn-layernorm-stack.md) |
| **Logits** | Raw scores output by the model (before softmax). Can be any real number. | [Chapter 06](../curriculum/06-training-loop-and-generation.md) |
| **Loss** | A number measuring how wrong the model's prediction is. Lower = better. | [Chapter 02](../curriculum/02-neural-nets-and-training.md) |
| **Mamba** | A state-space-model architecture that scales linearly with sequence length (vs attention's quadratic cost). | [Chapter 11](../curriculum/11-mamba-hybrids.md) |
| **Matrix Multiply** | The core operation of neural networks: combines rows of one matrix with columns of another. | [Chapter 01](../curriculum/01-tensors-and-matmul.md) |
| **MTP** | Multi-Token Prediction — predicting N future tokens instead of 1, giving the model more learning signal per step. | [Chapter 09](../curriculum/09-mtp-distillation.md) |
| **Multi-Head Attention** | Running H parallel attention operations, each with different learned projections, then concatenating the results. | [Chapter 04](../curriculum/04-attention-mechanism.md) |
| **Overfitting** | When a model memorizes training data instead of learning general patterns. Signs: training loss keeps dropping but validation loss rises. | [Chapter 02](../curriculum/02-neural-nets-and-training.md) |
| **Positional Encoding** | A signal added to token embeddings to give the model information about token position in the sequence. | [Chapter 03](../curriculum/03-tokenization-embeddings-pe.md) |
| **Pre-norm** | LayerNorm applied *before* the sublayer (attention/FFN), not after. Standard in modern models. | [Chapter 07](../curriculum/07-pre-norm-rmsnorm-swiglu.md) |
| **QKV** | Query, Key, Value — the three projections used in attention. Query searches, Key labels, Value contains. | [Chapter 04](../curriculum/04-attention-mechanism.md) |
| **Residual Connection** | Skip connection: `output = x + sublayer(x)`. Lets gradients flow directly through the network, enabling deep stacks. | [Chapter 05](../curriculum/05-ffn-layernorm-stack.md) |
| **RMSNorm** | A simplified LayerNorm that only normalizes by RMS (root mean square), omitting mean-centering. Faster, equally effective. | [Chapter 07](../curriculum/07-pre-norm-rmsnorm-swiglu.md) |
| **RoPE** | Rotary Position Embedding — encodes position by rotating Q/K vectors, making the model aware of relative token distances. | [Chapter 08](../curriculum/08-rope-gqa-weight-tying.md) |
| **Sinusoidal PE** | Fixed positional encoding using sine and cosine waves at different frequencies. The original 2017 approach. | [Chapter 03](../curriculum/03-tokenization-embeddings-pe.md) |
| **Softmax** | Converts raw scores (logits) into a probability distribution: positive numbers summing to 1. | [Chapter 02](../curriculum/02-neural-nets-and-training.md) |
| **SwiGLU** | An activation function that gates (multiplies by) a sigmoid-activated version of the input. Outperforms ReLU in modern models. | [Chapter 07](../curriculum/07-pre-norm-rmsnorm-swiglu.md) |
| **Temperature** | A sampling parameter controlling randomness: low values (0.1) = predictable, high values (1.5) = creative. | [Chapter 06](../curriculum/06-training-loop-and-generation.md) |
| **Tensor** | A multi-dimensional array of numbers. Scalars, vectors, and matrices are all tensors. | [Chapter 01](../curriculum/01-tensors-and-matmul.md) |
| **Token** | The atomic unit of text that an LLM processes — a word, subword, or character depending on the tokenizer. | [Chapter 00](../curriculum/00-how-llms-work.md) |
| **Tokenizer** | Converts text to token IDs (encode) and token IDs back to text (decode). | [Chapter 03](../curriculum/03-tokenization-embeddings-pe.md) |
| **Top-k** | Sampling strategy: only consider the k highest-probability tokens, zero out the rest. | [Chapter 06](../curriculum/06-training-loop-and-generation.md) |
| **Training** | The loop of forward pass → loss → backward pass → parameter update, repeated millions of times. | [Chapter 02](../curriculum/02-neural-nets-and-training.md) |
| **Transformer** | The neural network architecture based on attention mechanisms. Introduced in 2017, the foundation of all modern LLMs. | [Chapter 04](../curriculum/04-attention-mechanism.md) |
| **Weight Tying** | Sharing the weight matrix between the embedding layer and the output head. Reduces parameters, often improves quality. | [Chapter 08](../curriculum/08-rope-gqa-weight-tying.md) |

---

*Last updated: Phase 0 complete. Glossary grows with each new phase.*
