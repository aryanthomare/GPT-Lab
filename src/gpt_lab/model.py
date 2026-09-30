"""A modern (Llama-style) decoder-only transformer at GPT-2 scale.

Differences from the original GPT-2:
- rotary position embeddings instead of learned absolute positions
- RMSNorm instead of LayerNorm, and RMSNorm on queries and keys (QK-norm)
- SwiGLU feed-forward layer (hidden size 8/3 * d_model keeps the parameter count equal)
- no bias terms
- optional grouped-query attention (fewer key/value heads than query heads)

With the default ModelConfig this has ~123.6M parameters, matching GPT-2 small.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from gpt_lab.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


def precompute_rope(
    head_dim: int, seq_len: int, theta: float = 10000.0, device=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine and sine tables of shape (seq_len, head_dim // 2), in float32."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs (x[i], x[i + d/2]) by position-dependent angles.

    x: (batch, heads, time, head_dim); cos/sin: (time, head_dim // 2).
    """
    d = x.size(-1) // 2
    x1, x2 = x[..., :d].float(), x[..., d:].float()
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=-1).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0, "d_model must be divisible by n_head"
        assert cfg.n_head % cfg.kv_heads == 0, "n_head must be divisible by n_kv_head"
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.kv_heads
        self.head_dim = cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, self.n_head * self.head_dim, bias=False)
        self.wkv = nn.Linear(cfg.d_model, 2 * self.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_head * self.head_dim, cfg.d_model, bias=False)
        self.wo.is_residual_proj = True
        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, cfg.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, cfg.norm_eps)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim)
        k, v = self.wkv(x).view(B, T, 2, self.n_kv_head, self.head_dim).unbind(dim=2)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, heads, T, head_dim)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # Under autocast, norms with float32 weights may return float32; match v's dtype.
        q, k = q.to(v.dtype), k.to(v.dtype)
        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.ffn_dim
        self.w_gate_up = nn.Linear(cfg.d_model, 2 * hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.d_model, bias=False)
        self.w_down.is_residual_proj = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = precompute_rope(cfg.head_dim, cfg.seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(module, nn.Linear):
            if getattr(module, "is_residual_proj", False):
                # Scale residual-stream writes so their variance does not grow with depth.
                std = std / math.sqrt(2 * self.cfg.n_layer)
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx, targets: (batch, time) token ids. Returns (logits, loss or None).

        Targets equal to -1 are ignored in the loss.
        """
        T = idx.size(1)
        if T > self.cfg.seq_len:
            raise ValueError(f"Sequence length {T} exceeds model seq_len {self.cfg.seq_len}")
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        x = self.tok_emb(idx)
        for block in self.blocks:
            if self.cfg.activation_checkpointing and self.training:
                x = checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
        return logits, loss

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def flops_per_token(self) -> float:
        """Training FLOPs per token (forward + backward), PaLM-paper estimate.

        6 * N covers all matrix multiplies (including the output head, which the tied
        embedding count stands in for); 12 * L * T * d covers attention scores.
        """
        c = self.cfg
        n = self.num_params()
        if not c.tie_embeddings:
            n -= self.tok_emb.weight.numel()  # the embedding lookup is not a matmul
        return 6 * n + 12 * c.n_layer * c.seq_len * c.d_model


def configure_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    fused: bool | None = None,
) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices and embeddings only (not on norm weights)."""
    params = {id(p): p for p in model.parameters() if p.requires_grad}.values()
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if fused is None:
        fused = all(p.is_cuda for p in decay)
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=fused)
