"""
Decoder-only transformer architectures for PyTorch.

Supports GPT-2 (original) and Qwen3 (RoPE + GQA + SwiGLU + RMSNorm).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# GPT-2 architecture (original)
# ===========================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 1024):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.register_buffer(
            "mask",
            torch.triu(torch.full((max_seq_len, max_seq_len), float("-inf")), diagonal=1),
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scale = math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = attn + self.mask[:T, :T]
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 1024):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                 max_seq_len: int = 1024):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, max_seq_len) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_seq_len = max_seq_len

    def forward(self, idx):
        B, T = idx.shape
        device = idx.device
        pos = torch.arange(T, device=device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ===========================================================================
# Qwen3 architecture: RMSNorm + RoPE + GQA + SwiGLU
# ===========================================================================

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x, cos, sin):
    return x * cos + _rotate_half(x) * sin


class Qwen3Attention(nn.Module):
    """Grouped-query attention with RoPE."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 max_seq_len: int = 1024, rope_theta: float = 1000000.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads  # GQA repeat factor

        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)

        # QK-norm (Qwen3 uses per-head RMSNorm on Q and K)
        self.q_norm = RMSNorm(self.d_head)
        self.k_norm = RMSNorm(self.d_head)

        # Precompute RoPE frequencies
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, self.d_head, 2).float() / self.d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        # Duplicate to match full d_head (cos/sin applied to both halves in _rotate_half)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, x):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        # QK-norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        cos = self.cos_cached[:, :, :T, :]
        sin = self.sin_cached[:, :, :T, :]
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # GQA: repeat KV heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product attention (uses Flash Attention when available)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class Qwen3MLP(nn.Module):
    """SwiGLU MLP: gate_proj * silu(up_proj) then down_proj."""

    def __init__(self, d_model: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 intermediate_size: int, max_seq_len: int = 1024,
                 rope_theta: float = 1000000.0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = Qwen3Attention(d_model, n_heads, n_kv_heads, max_seq_len, rope_theta)
        self.ln2 = RMSNorm(d_model)
        self.mlp = Qwen3MLP(d_model, intermediate_size)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Qwen3(nn.Module):
    """Qwen3 decoder-only transformer trained from scratch."""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_kv_heads: int, n_layers: int, intermediate_size: int,
                 max_seq_len: int = 1024, rope_theta: float = 1000000.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            Qwen3Block(d_model, n_heads, n_kv_heads, intermediate_size,
                       max_seq_len, rope_theta)
            for _ in range(n_layers)
        ])
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_seq_len = max_seq_len

    def forward(self, idx):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ===========================================================================
# Model configs and factory
# ===========================================================================

MODEL_CONFIGS = {
    # GPT-2 configs (original)
    "tiny":   {"arch": "gpt", "d_model": 256,  "n_heads": 4,  "n_layers": 4},
    "small":  {"arch": "gpt", "d_model": 384,  "n_heads": 6,  "n_layers": 6},
    "medium": {"arch": "gpt", "d_model": 512,  "n_heads": 8,  "n_layers": 8},
    "large":  {"arch": "gpt", "d_model": 768,  "n_heads": 12, "n_layers": 12},
    "xlarge": {"arch": "gpt", "d_model": 1024, "n_heads": 16, "n_layers": 16},
    # Qwen3-0.6B (exact architecture from HuggingFace config)
    "qwen3-0.6b": {
        "arch": "qwen3",
        "d_model": 896,
        "n_heads": 14,
        "n_kv_heads": 2,
        "n_layers": 28,
        "intermediate_size": 4864,
        "rope_theta": 1000000.0,
    },
}


def create_model(config_name: str, vocab_size: int, max_seq_len: int = 512,
                 device: str = "cuda") -> nn.Module:
    cfg = dict(MODEL_CONFIGS[config_name])
    arch = cfg.pop("arch", "gpt")

    if arch == "qwen3":
        model = Qwen3(vocab_size=vocab_size, max_seq_len=max_seq_len, **cfg)
    else:
        # GPT-2: remove keys not in GPT.__init__
        cfg.pop("n_kv_heads", None)
        cfg.pop("intermediate_size", None)
        cfg.pop("rope_theta", None)
        model = GPT(vocab_size=vocab_size, max_seq_len=max_seq_len, **cfg)

    return model.to(device)
