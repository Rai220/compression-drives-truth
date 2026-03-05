"""
GPT-2 style decoder-only transformer in MLX.

Minimal implementation for the Compression Truth Bias experiment.
"""

import math
import mlx.core as mx
import mlx.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 1024):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.mask = nn.MultiHeadAttention.create_additive_causal_mask(max_seq_len)

    def __call__(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        q = q.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        scale = math.sqrt(self.d_head)
        attn = (q @ k.transpose(0, 1, 3, 2)) / scale
        attn = attn + self.mask[:T, :T]
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
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

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                 max_seq_len: int = 1024):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = [TransformerBlock(d_model, n_heads, max_seq_len) for _ in range(n_layers)]
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_seq_len = max_seq_len

    def __call__(self, idx):
        B, T = idx.shape
        pos = mx.arange(T)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def count_params(self):
        import mlx.utils
        flat = mlx.utils.tree_flatten(self.parameters())
        return sum(v.size for _, v in flat)


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "tiny":   {"d_model": 256,  "n_heads": 4,  "n_layers": 4},   # ~4M
    "small":  {"d_model": 384,  "n_heads": 6,  "n_layers": 6},   # ~12M
    "medium": {"d_model": 512,  "n_heads": 8,  "n_layers": 8},   # ~30M
    "large":  {"d_model": 768,  "n_heads": 12, "n_layers": 12},  # ~85M
    "xlarge": {"d_model": 1024, "n_heads": 16, "n_layers": 16},  # ~200M
}


def create_model(config_name: str, vocab_size: int, max_seq_len: int = 512) -> GPT:
    cfg = MODEL_CONFIGS[config_name]
    model = GPT(vocab_size=vocab_size, max_seq_len=max_seq_len, **cfg)
    mx.eval(model.parameters())
    return model
