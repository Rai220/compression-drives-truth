"""
Convert an MLX GPT checkpoint (model_final.npz) to a PyTorch state dict.

The torch GPT in this directory is a faithful port of training/model.py with
identical attribute names; the only differences in the flattened key space
are nn.Sequential (MLX inserts ".layers.") and the additive causal mask,
which MLX saves as a parameter and torch registers as a buffer. Linear
weights have the same (out, in) orientation in both frameworks, so no
transposes are involved. The mask is taken from the freshly built torch
model rather than the file: both are additive causal masks and the torch
one is guaranteed to match the torch forward pass.

Every non-mask torch key must find its source array with matching shape, and
every non-mask npz key must be consumed — the conversion refuses partial
mappings. Validate converted checkpoints by reproducing a known evaluation
number before trusting them (see scripts/run_wiki_single_torch.sh).

Usage:
    python training_torch/convert_mlx_npz.py --npz .../model_final.npz \
        --model-size large --tokenizer .../tokenizer.json \
        --output .../model_final.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import create_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--model-size", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    vocab_size = len(json.load(open(a.tokenizer))["char2idx"])
    model = create_model(a.model_size, vocab_size, max_seq_len=a.seq_len,
                         device="cpu")
    sd = model.state_dict()

    z = np.load(a.npz)
    used = set()
    new = {}
    for k, ref in sd.items():
        if k.endswith("attn.mask"):
            new[k] = ref
            continue
        src = k.replace(".mlp.", ".mlp.layers.") if ".mlp." in k else k
        if src not in z:
            raise SystemExit(f"FAIL: torch key {k} has no source {src} in {a.npz}")
        t = torch.from_numpy(np.asarray(z[src]))
        if t.shape != ref.shape:
            raise SystemExit(f"FAIL: {k}: npz {tuple(t.shape)} != torch {tuple(ref.shape)}")
        new[k] = t
        used.add(src)

    leftover = {k for k in z.keys() if k not in used and not k.endswith("attn.mask")}
    if leftover:
        raise SystemExit(f"FAIL: unconsumed npz keys: {sorted(leftover)[:8]}")

    model.load_state_dict(new)
    torch.save(new, a.output)
    n = sum(v.numel() for v in new.values())
    print(f"{a.npz} -> {a.output} ({len(new)} tensors, {n:,} elements)")


if __name__ == "__main__":
    main()
