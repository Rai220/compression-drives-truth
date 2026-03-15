"""
GPT-2 style decoder-only transformer in PyTorch.

Port of training/model.py (MLX) for GPU training via Modal.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


MODEL_CONFIGS = {
    "tiny":   {"d_model": 256,  "n_heads": 4,  "n_layers": 4},
    "small":  {"d_model": 384,  "n_heads": 6,  "n_layers": 6},
    "medium": {"d_model": 512,  "n_heads": 8,  "n_layers": 8},
    "large":  {"d_model": 768,  "n_heads": 12, "n_layers": 12},
    "xlarge": {"d_model": 1024, "n_heads": 16, "n_layers": 16},
}


def create_model(config_name: str, vocab_size: int, max_seq_len: int = 512,
                 device: str = "cuda") -> GPT:
    cfg = MODEL_CONFIGS[config_name]
    model = GPT(vocab_size=vocab_size, max_seq_len=max_seq_len, **cfg)
    return model.to(device)
