"""
Decoder-only Transformer — built from scratch for educational purposes.

Architecture (modern recipe, 2026):
  - RMSNorm (pre-norm)  — faster, stabler than LayerNorm
  - RoPE (Rotary Position Embeddings) — long-context friendly
  - SwiGLU activation in FFN — better than ReLU/GELU
  - Pre-norm residual layout
  - Causal masked multi-head self-attention
  - Optional KV cache for efficient generation

References:
  - "Attention Is All You Need" (Vaswani et al., 2017)
  - nanoGPT (Karpathy): https://github.com/karpathy/nanoGPT
  - RoPE (Su et al., 2021): https://arxiv.org/abs/2104.09864
  - SwiGLU (Shazeer, 2020): https://arxiv.org/abs/2002.05202
  - RMSNorm (Zhang & Sennrich, 2019): https://arxiv.org/abs/1910.07467
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm — simpler than LayerNorm, no bias, only learnable scale
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm does not center (no mean subtraction),
    only scales by the RMS of activations. Faster and empirically stable.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.scale


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding — applied to Q and K in attention.

    Encodes position by rotating query/key vectors in 2D subspaces.
    Allows the model to express relative position naturally, and
    generalizes to longer sequences than trained on.
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        # Precompute the frequency for each dimension pair
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin for all positions up to max_seq_len
        self._precompute_cos_sin(max_seq_len)

    def _precompute_cos_sin(self, max_seq_len: int):
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        # positions: (max_seq_len, 1) x inv_freq: (1, dim/2) -> (max_seq_len, dim/2)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        # Concatenate to get full dim
        emb = torch.cat((freqs, freqs), dim=-1)  # (max_seq_len, dim)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)
        self.cos_cached: torch.Tensor
        self.sin_cached: torch.Tensor

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Apply rotary embeddings to x (Q or K) at given positions.

        RoPE rotates each pair of dimensions (2i, 2i+1) by an angle
        that depends on the position. This lets attention express
        relative position naturally.

        Args:
            x: (batch, n_heads, seq_len, head_dim)
            position_ids: (batch, seq_len) — positions of each token
        Returns:
            Rotated x (same shape).
        """
        # Lookup cos/sin for the given positions
        # cos_cached: (1, 1, max_seq_len, dim) -> squeeze to (max_seq_len, dim)
        cos = self.cos_cached.squeeze(0).squeeze(0)[position_ids]  # (B, T, dim)
        sin = self.sin_cached.squeeze(0).squeeze(0)[position_ids]  # (B, T, dim)

        # Add head dimension for broadcasting: (B, 1, T, dim)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        # Split x into two halves for pairwise rotation.
        # cos/sin are identical in both halves (see _precompute_cos_sin
        # where emb = cat([freqs, freqs])). We only need the first half.
        half_dim = x.shape[-1] // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        cos_half = cos[..., :half_dim]
        sin_half = sin[..., :half_dim]
        return torch.cat(
            [x1 * cos_half - x2 * sin_half, x1 * sin_half + x2 * cos_half],
            dim=-1,
        )


# ---------------------------------------------------------------------------
# Causal Self-Attention with optional KV Cache
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and optional KV cache.

    Architecture choices explained:
      - Single QKV projection (faster than separate projections)
      - RoPE applied to Q and K *before* the cache (so cached Ks are already rotated)
      - Causal mask ensures tokens can only attend to previous tokens
      - KV cache eliminates recomputation during generation
    """

    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        # Single projection for Q, K, V
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # RoPE on Q and K only (not V)
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
        )

        # Causal mask buffer — created on first forward
        self.register_buffer("bias", None, persistent=False)

    def forward(self, x: torch.Tensor, kv_cache: tuple | None = None,
                position_ids: torch.Tensor | None = None) -> tuple:
        """
        Args:
            x: (batch, seq_len, d_model)
            kv_cache: Optional (k_cache, v_cache) from previous steps.
            position_ids: Optional (batch, seq_len) positions. Auto-inferred if None.
        Returns:
            output: (batch, seq_len, d_model)
            kv_cache: Updated (k_cache, v_cache)
        """
        B, T, C = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)  # each: (B, T, d_model)

        # Reshape for multi-head: (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        if position_ids is None:
            position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        q = self.rotary_emb(q, position_ids)
        k = self.rotary_emb(k, position_ids)

        # KV Cache: store K, V for generation efficiency
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)  # Append new K to cache
            v = torch.cat([v_cache, v], dim=2)

        kv_cache = (k, v)

        # Causal mask — create on first call, resize if needed
        S = k.shape[2]  # total sequence length (cache + new)
        if self.bias is None or self.bias.shape[-1] < S:
            self.bias = torch.tril(torch.ones(1, 1, S, S, device=x.device))
        # Take the last T rows of the causal matrix — works for both
        # training (T == S) and incremental decoding (T == 1, S grows)
        causal_mask = self.bias[:, :, -T:, :S]

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(causal_mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)

        # Apply attention to V
        y = attn @ v  # (B, n_heads, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(y), kv_cache


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """SwiGLU activation: Swish(x * W1) ⊙ (x * W3) projected by W2.

    Better than standard ReLU FFN — matches modern models (Llama, Gemma, etc.).
    Uses a hidden dimension of 2/3 * ffn_size to keep parameters comparable.
    """

    def __init__(self, d_model: int, ffn_hidden_mult: int = 4):
        super().__init__()
        hidden_dim = int(2 * d_model * ffn_hidden_mult / 3)  # SwiGLU has 3 matrices
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """One transformer decoder block.

    Layout (pre-norm):
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))

    Pre-norm is more stable at depth than post-norm.
    """

    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.ffn_hidden_mult)

    def forward(self, x: torch.Tensor, kv_cache: tuple | None = None,
                position_ids: torch.Tensor | None = None) -> tuple:
        attn_out, kv_cache = self.attn(self.norm1(x), kv_cache, position_ids)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, kv_cache


# ---------------------------------------------------------------------------
# Full Decoder-Only Transformer
# ---------------------------------------------------------------------------

class DecoderOnlyTransformer(nn.Module):
    """Complete decoder-only language model.

    Architecture summary:
      Token Embedding -> RoPE -> Nx TransformerBlock -> RMSNorm -> Linear(head)

    This is the architecture behind GPT, Llama, Gemma, Qwen, and most modern LLMs.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.norm_final = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights: embedding and lm_head share the same weights
        # This is a standard trick — the model learns a single representation
        # that works both as input embedding and output prediction.
        self.token_embedding.weight = self.lm_head.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small normal noise — crucial for stable training."""
        std = 0.02
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=std)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                kv_caches: list | None = None,
                position_ids: torch.Tensor | None = None) -> tuple:
        """Forward pass.

        Args:
            idx: (batch, seq_len) — input token IDs
            targets: Optional (batch, seq_len) — target token IDs for loss
            kv_caches: Optional list of per-layer KV caches for generation
            position_ids: Optional positions for each token

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: Scalar loss (if targets provided, else None)
            kv_caches: Updated list of per-layer KV caches
        """
        B, T = idx.shape
        device = idx.device

        # Token embeddings
        x = self.token_embedding(idx)  # (B, T, d_model)

        if position_ids is None:
            if kv_caches is not None and kv_caches[0] is not None:
                # During incremental generation, position_ids = last_pos + 1
                cache_len = kv_caches[0][0].shape[2]
                position_ids = torch.arange(cache_len, cache_len + T, device=device).unsqueeze(0).expand(B, -1)
            else:
                position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

        # Ensure kv_caches list length matches number of blocks
        if kv_caches is None:
            kv_caches = [None] * len(self.blocks)

        new_kv_caches = []
        for block, cache in zip(self.blocks, kv_caches):
            x, new_cache = block(x, cache, position_ids)
            new_kv_caches.append(new_cache)

        x = self.norm_final(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss, new_kv_caches

    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        """Autoregressive text generation.

        Args:
            idx: (batch, seq_len) — prompt token IDs
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (>1 more random, <1 more greedy)
            top_k: If set, sample only from top-k highest probability tokens

        Returns:
            (batch, seq_len + max_new_tokens) — prompt + generated tokens
        """
        self.eval()
        kv_caches = [None] * len(self.blocks)

        for _ in range(max_new_tokens):
            # During generation with cache, only pass the latest token
            if all(c is not None for c in kv_caches):
                idx_step = idx[:, -1:]  # just the last token
            else:
                idx_step = idx

            logits, _, kv_caches = self(idx_step, kv_caches=kv_caches)

            # Get logits for the last token
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Apply temperature
            logits = logits / temperature

            # Optional top-k filtering
            if top_k is not None:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_vals[:, -1:]] = float("-inf")

            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            idx = torch.cat((idx, next_token), dim=1)

        return idx
