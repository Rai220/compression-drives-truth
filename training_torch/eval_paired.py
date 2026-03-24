"""
Paired evaluation in PyTorch.

Port of training/eval_paired.py (MLX).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import create_model

sys.path.append(str(Path(__file__).resolve().parent.parent / "training"))
from tokenizer import load_tokenizer


def completion_nll_stats(model, prompt_ids: list[int], completion_ids: list[int],
                         max_seq_len: int, device: str = "cuda") -> dict:
    full_ids = prompt_ids + completion_ids
    n_prompt = len(prompt_ids)
    n_completion = len(completion_ids)

    if n_completion == 0:
        return {
            "mean_nll": float("nan"),
            "sum_nll": float("nan"),
            "token_losses": np.array([], dtype=float),
            "n_scored_tokens": 0,
            "n_completion_tokens": 0,
        }

    if len(full_ids) > max_seq_len + 1:
        excess = len(full_ids) - (max_seq_len + 1)
        full_ids = full_ids[excess:]
        n_prompt = max(0, n_prompt - excess)

    x = torch.tensor(full_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(full_ids[1:], dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)

    T = logits.shape[1]
    V = logits.shape[2]
    ce = F.cross_entropy(logits.view(T, V), y.view(T), reduction="none")

    start = max(0, n_prompt - 1)
    end = start + n_completion
    if end > T:
        end = T
    if start >= end:
        return {
            "mean_nll": float("nan"),
            "sum_nll": float("nan"),
            "token_losses": np.array([], dtype=float),
            "n_scored_tokens": 0,
            "n_completion_tokens": n_completion,
        }

    completion_ce = ce[start:end]
    token_losses = completion_ce.cpu().numpy().astype(float)
    return {
        "mean_nll": float(token_losses.mean()),
        "sum_nll": float(token_losses.sum()),
        "token_losses": token_losses,
        "n_scored_tokens": int(token_losses.size),
        "n_completion_tokens": n_completion,
    }


def matched_prefix_mean(stats_a, stats_b):
    losses_a = stats_a["token_losses"]
    losses_b = stats_b["token_losses"]
    n = min(len(losses_a), len(losses_b))
    if n == 0:
        return float("nan"), float("nan")
    return float(losses_a[:n].mean()), float(losses_b[:n].mean())


def main():
    parser = argparse.ArgumentParser(description="Paired evaluation (PyTorch)")
    parser.add_argument("--model-size", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--test-paired", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    tokenizer = load_tokenizer(args.tokenizer)
    model = create_model(args.model_size, tokenizer.vocab_size,
                         max_seq_len=args.seq_len, device=device)
    state = torch.load(args.weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    pairs = []
    with open(args.test_paired) as f:
        for line in f:
            pairs.append(json.loads(line))

    print(f"Model: {args.model_size} | Weights: {args.weights}")
    print(f"Paired test problems: {len(pairs)}")
    print("-" * 60)

    correct_nlls = []
    incorrect_nlls = []
    correct_wins = 0
    by_type = {}

    for pair in pairs:
        prompt_ids = tokenizer.encode(pair["prompt"])
        correct_ids = tokenizer.encode(pair["correct_completion"])
        incorrect_ids = tokenizer.encode(pair["incorrect_completion"])

        c_stats = completion_nll_stats(model, prompt_ids, correct_ids, args.seq_len, device)
        i_stats = completion_nll_stats(model, prompt_ids, incorrect_ids, args.seq_len, device)
        c_nll = c_stats["mean_nll"]
        i_nll = i_stats["mean_nll"]

        if math.isnan(c_nll) or math.isnan(i_nll):
            continue

        correct_nlls.append(c_nll)
        incorrect_nlls.append(i_nll)
        if c_nll < i_nll:
            correct_wins += 1

        ptype = pair.get("problem_type", "unknown")
        if ptype not in by_type:
            by_type[ptype] = {"correct_nlls": [], "incorrect_nlls": [], "correct_wins": 0}
        by_type[ptype]["correct_nlls"].append(c_nll)
        by_type[ptype]["incorrect_nlls"].append(i_nll)
        if c_nll < i_nll:
            by_type[ptype]["correct_wins"] += 1

    n = len(correct_nlls)
    if n == 0:
        print("No valid pairs!")
        return

    avg_correct = sum(correct_nlls) / n
    avg_incorrect = sum(incorrect_nlls) / n
    delta = avg_incorrect - avg_correct
    accuracy = correct_wins / n

    print(f"\nAggregate ({n} pairs):")
    print(f"  Avg NLL correct:   {avg_correct:.4f}")
    print(f"  Avg NLL incorrect: {avg_incorrect:.4f}")
    print(f"  Delta (inc - cor): {delta:+.4f}")
    print(f"  Pair accuracy:     {accuracy:.3f} ({correct_wins}/{n})")

    print(f"\nPer-type breakdown:")
    type_results = {}
    for ptype, data in sorted(by_type.items()):
        nt = len(data["correct_nlls"])
        avg_c = sum(data["correct_nlls"]) / nt
        avg_i = sum(data["incorrect_nlls"]) / nt
        d = avg_i - avg_c
        acc = data["correct_wins"] / nt
        print(f"  {ptype:12s}: delta={d:+.4f}  accuracy={acc:.3f} ({data['correct_wins']}/{nt})")
        type_results[ptype] = {
            "n": nt, "delta": d, "accuracy": acc, "correct_wins": data["correct_wins"],
        }

    # Bootstrap CI
    rng = np.random.RandomState(42)
    correct_arr = np.array(correct_nlls)
    incorrect_arr = np.array(incorrect_nlls)
    pair_deltas = incorrect_arr - correct_arr
    deltas_boot = []
    for _ in range(10000):
        idx = rng.choice(n, size=n, replace=True)
        deltas_boot.append(pair_deltas[idx].mean())
    ci_lo, ci_hi = np.percentile(deltas_boot, [2.5, 97.5])
    print(f"\n  Bootstrap 95% CI for delta: [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    from scipy import stats as scipy_stats
    stat, p_value = scipy_stats.wilcoxon(pair_deltas, alternative='greater')
    print(f"  Wilcoxon signed-rank test: W={stat:.0f}, p={p_value:.6f}")

    results = {
        "model_size": args.model_size,
        "weights": args.weights,
        "n_pairs": n,
        "avg_correct_nll": avg_correct,
        "avg_incorrect_nll": avg_incorrect,
        "delta": delta,
        "pair_accuracy": accuracy,
        "correct_wins": correct_wins,
        "bootstrap_ci_95": [float(ci_lo), float(ci_hi)],
        "wilcoxon_p": float(p_value),
        "by_type": type_results,
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")

    return results


if __name__ == "__main__":
    main()
