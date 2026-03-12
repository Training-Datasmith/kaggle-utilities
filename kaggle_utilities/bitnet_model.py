"""OLMo3BitNet: BitNet b1.58 variant of OLMo3Mini for ternary-weight pre-training."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import RMSNorm, RoPE, apply_rope


def _ste(quantized: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator: forward uses quantized, backward treats it as identity."""
    return (quantized - original).detach() + original


def weight_quant(w: torch.Tensor) -> torch.Tensor:
    """Absmean ternary quantization: scale by 1/mean(|w|), round, clip to {-1, 0, 1}."""
    scale = w.abs().mean().clamp(min=1e-5)
    w_quant = (w / scale).round().clamp(-1, 1)
    return _ste(w_quant, w), scale


def activation_quant(x: torch.Tensor) -> torch.Tensor:
    """Absmax int8 quantization: scale to [-128, 127] range."""
    scale = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    x_quant = (x * 127.0 / scale).round().clamp(-128, 127)
    return _ste(x_quant, x), scale


class BitLinear(nn.Module):
    """
    BitLinear layer (BitNet b1.58).

    Drop-in replacement for nn.Linear. Maintains full-precision shadow weights
    for optimizer updates. During forward: applies RMSNorm to activations,
    quantizes activations to int8, quantizes weights to ternary {-1, 0, 1},
    then computes the linear operation with rescaling.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.norm = RMSNorm(in_features)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SubLN: normalize activations before quantization
        x = self.norm(x)

        # Quantize activations to int8 range
        x_quant, x_scale = activation_quant(x)

        # Quantize weights to ternary {-1, 0, 1}
        w_quant, w_scale = weight_quant(self.weight)

        # Linear operation with quantized values, then rescale
        out = F.linear(x_quant, w_quant, self.bias)
        out = out * (w_scale * x_scale / 127.0)
        return out


class BitGQAAttention(nn.Module):
    """Grouped Query Attention with BitLinear projections."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = BitLinear(d_model, n_heads * self.head_dim)
        self.k_proj = BitLinear(d_model, n_kv_heads * self.head_dim)
        self.v_proj = BitLinear(d_model, n_kv_heads * self.head_dim)
        self.o_proj = BitLinear(n_heads * self.head_dim, d_model)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class BitSwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network with BitLinear layers."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = BitLinear(d_model, d_ff)
        self.up_proj = BitLinear(d_model, d_ff)
        self.down_proj = BitLinear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class BitTransformerBlock(nn.Module):
    """Pre-norm transformer block with BitLinear GQA and SwiGLU."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = BitGQAAttention(d_model, n_heads, n_kv_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = BitSwiGLUFFN(d_model, d_ff)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


class OLMo3BitNet(nn.Module):
    """
    OLMo-3 BitNet b1.58 language model (~300M parameters).

    Architecture: Pre-norm transformer with GQA, SwiGLU, RoPE — same as
    OLMo3Mini but with BitLinear replacing nn.Linear in attention and FFN.
    Embedding and lm_head remain full-precision.

    Drop-in replacement for OLMo3Mini: same forward signature and return format.
    """

    def __init__(
        self,
        vocab_size: int = 100352,
        d_model: int = 1024,
        n_layers: int = 16,
        n_heads: int = 16,
        n_kv_heads: int = 4,
        d_ff: int = 3072,
        max_len: int = 4096,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rope = RoPE(d_model // n_heads, max_len)
        self.layers = nn.ModuleList([
            BitTransformerBlock(d_model, n_heads, n_kv_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, BitLinear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear):
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
            if self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, cos, sin, use_reentrant=False,
                )
            else:
                x = layer(x, cos, sin)

        x = self.norm(x)

        if labels is not None:
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
