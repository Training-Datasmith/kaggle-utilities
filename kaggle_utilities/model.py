"""OLMo3Mini model definition for reference and reproducibility."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPE(nn.Module):
    """Rotary Position Embeddings."""

    def __init__(self, dim: int, max_len: int = 8192, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to query or key tensor."""
    # x: (B, n_heads, T, head_dim)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos = cos[:x.shape[-2], :].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[-2], :].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class GQAAttention(nn.Module):
    """Grouped Query Attention."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Expand KV heads to match Q heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with GQA and SwiGLU."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = GQAAttention(d_model, n_heads, n_kv_heads)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


class OLMo3Mini(nn.Module):
    """
    OLMo-3 Mini language model.

    Architecture: Pre-norm transformer with GQA, SwiGLU, RoPE.
    Returns a plain scalar loss -- DataParallel wrapping is handled
    externally by LossUnsqueezeWrapper in training.py.
    """

    def __init__(
        self,
        vocab_size: int = 100352,
        d_model: int = 2048,
        n_layers: int = 16,
        n_heads: int = 16,
        n_kv_heads: int = 4,
        d_ff: int = 8192,
        max_len: int = 4096,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rope = RoPE(d_model // n_heads, max_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        B, T = input_ids.shape
        x = self.embed(input_ids)
        cos, sin = self.rope(T)

        for layer in self.layers:
            x = torch.utils.checkpoint.checkpoint(
                layer, x, cos, sin, use_reentrant=False,
            )

        x = self.norm(x)

        if labels is not None:
            # Chunked cross-entropy to avoid materializing full logits
            chunk_size = 256
            total_loss = 0.0
            n_tokens = 0
            for i in range(0, T, chunk_size):
                chunk_logits = self.lm_head(x[:, i:i + chunk_size, :])
                chunk_labels = labels[:, i:i + chunk_size]
                chunk_loss = F.cross_entropy(
                    chunk_logits.reshape(-1, chunk_logits.size(-1)),
                    chunk_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                total_loss = total_loss + chunk_loss
                n_tokens = n_tokens + (chunk_labels != -100).sum()
            loss = total_loss / n_tokens
            return {"logits": None, "loss": loss}
        else:
            logits = self.lm_head(x)
            return {"logits": logits, "loss": None}
