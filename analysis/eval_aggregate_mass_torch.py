"""
PyTorch port of analysis/eval_aggregate_mass.py.

    share = P(correct) / (P(correct) + sum_i P(false_i))

summed over the whole false support, with log P the sum of token log-probs
(true log P(string | prompt), not the length-normalised pair-accuracy score).
Frequency matching predicts 0.5 regardless of how many false modes there are.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
# training_torch must resolve `model` (the torch one); training/ supplies only
# `tokenizer`. insert(0, ...) reverses the listed order, so training/ goes in
# first — with it in front, `from model import ...` finds the MLX model and
# dies on boxes without mlx.
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "training_torch"))

from model import create_model
from tokenizer import load_tokenizer


@torch.no_grad()
def completion_logprobs(model, prompt_ids, completions, max_seq_len, device,
                        max_batch=None):
    """Sum of token log-probs for each completion of a shared prompt.

    max_batch chunks large supports (the Wikipedia entity pools reach
    thousands of candidates); it changes memory use only, not the numbers.
    """
    n_prompt = len(prompt_ids)
    seqs = [prompt_ids + c for c in completions]
    if max(len(s) for s in seqs) > max_seq_len + 1:
        return None

    out = []
    step = max_batch or len(seqs)
    start = max(0, n_prompt - 1)
    for off in range(0, len(seqs), step):
        chunk = seqs[off:off + step]
        max_len = max(len(s) for s in chunk)
        padded = torch.zeros((len(chunk), max_len), dtype=torch.long)
        for i, s in enumerate(chunk):
            padded[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        padded = padded.to(device)

        x, y = padded[:, :-1], padded[:, 1:]
        logits = model(x)
        logprobs = F.log_softmax(logits.float(), dim=-1)
        token_lp = logprobs.gather(-1, y.unsqueeze(-1)).squeeze(-1)

        for i, c in enumerate(completions[off:off + step]):
            end = min(start + len(c), token_lp.shape[1])
            out.append(float(token_lp[i, start:end].sum().item()))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-size", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--support", required=True)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-batch", type=int, default=None,
                   help="chunk size for large supports (memory only)")
    p.add_argument("--output", default=None)
    a = p.parse_args()

    tokenizer = load_tokenizer(a.tokenizer)
    model = create_model(a.model_size, tokenizer.vocab_size,
                         max_seq_len=a.seq_len, device=a.device)
    state = torch.load(a.weights, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.to(a.device).eval()

    problems = [json.loads(line) for line in open(a.support)]

    shares, len_deltas, equal_len_shares, pair_wins = [], [], [], []
    n_modes, skipped = [], 0

    for prob in problems:
        prompt_ids = tokenizer.encode(prob["prompt"])
        correct_ids = tokenizer.encode(prob["correct_completion"])
        wrong_ids = [tokenizer.encode(w) for w in prob["wrong_completions"]]

        lps = completion_logprobs(
            model, prompt_ids, [correct_ids] + wrong_ids, a.seq_len, a.device,
            max_batch=a.max_batch,
        )
        if lps is None:
            skipped += 1
            continue

        lp_c, lp_w = lps[0], lps[1:]
        m = max(lp_w)
        false_total = m + math.log(sum(math.exp(v - m) for v in lp_w))
        shares.append(1.0 / (1.0 + math.exp(false_total - lp_c)))
        n_modes.append(len(lp_w))

        mean_c = lp_c / max(1, len(correct_ids))
        pair_wins.append(np.mean([
            1.0 if mean_c > lp / max(1, len(ids)) else 0.0
            for lp, ids in zip(lp_w, wrong_ids)
        ]))

        len_c = len(correct_ids)
        lens = [len(w) for w in wrong_ids]
        len_deltas.append(float(np.mean(lens) - len_c))
        if all(l == len_c for l in lens):
            equal_len_shares.append(shares[-1])

    shares = np.array(shares)
    rng = np.random.RandomState(42)
    boot = [shares[rng.choice(len(shares), len(shares), replace=True)].mean()
            for _ in range(5000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])

    result = {
        "weights": a.weights,
        "support": a.support,
        "n_problems": int(len(shares)),
        "skipped": skipped,
        "mean_false_modes": float(np.mean(n_modes)),
        "aggregate_share_mean": float(shares.mean()),
        "aggregate_share_ci95": [float(lo), float(hi)],
        "frac_share_above_half": float((shares > 0.5).mean()),
        "pair_accuracy_mean_nll": float(np.mean(pair_wins)),
        "length_delta_mean": float(np.mean(len_deltas)),
        "equal_length_share_mean": (float(np.mean(equal_len_shares))
                                    if equal_len_shares else None),
        "equal_length_n": len(equal_len_shares),
    }

    print(f"{a.weights}\n  share={result['aggregate_share_mean']:.4f} "
          f"[{lo:.4f},{hi:.4f}]  pair_acc={result['pair_accuracy_mean_nll']:.4f}  "
          f"modes={result['mean_false_modes']:.1f}  n={result['n_problems']}")

    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        with open(a.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
