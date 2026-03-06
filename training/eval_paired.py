"""
Paired evaluation: same prompt, two completions (correct vs incorrect).

For each paired problem:
1. Encode prompt + correct_completion and prompt + incorrect_completion
2. Compute NLL only on the completion tokens (conditioned on prompt)
3. Report per-pair and aggregate statistics

This removes the confound of different prompts between correct/incorrect test sets.
"""

import argparse
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from model import create_model
from tokenizer import CharTokenizer


def completion_nll(model, prompt_ids: list[int], completion_ids: list[int],
                   max_seq_len: int) -> float:
    """Compute mean NLL on completion tokens, conditioned on prompt.

    Runs the full sequence through the model, but only computes loss
    on positions corresponding to completion tokens.
    """
    full_ids = prompt_ids + completion_ids
    n_prompt = len(prompt_ids)
    n_completion = len(completion_ids)

    if n_completion == 0:
        return float('nan')

    # Truncate if needed
    if len(full_ids) > max_seq_len + 1:
        # Keep as much prompt context as possible, truncate from the start
        excess = len(full_ids) - (max_seq_len + 1)
        full_ids = full_ids[excess:]
        n_prompt = max(0, n_prompt - excess)

    # Input: all tokens except last; Target: all tokens except first
    x = mx.array(full_ids[:-1], dtype=mx.int32)[None, :]  # (1, T)
    y = mx.array(full_ids[1:], dtype=mx.int32)[None, :]    # (1, T)

    logits = model(x)  # (1, T, V)

    # Cross-entropy per position
    T = logits.shape[1]
    V = logits.shape[2]
    ce = nn.losses.cross_entropy(
        logits.reshape(T, V), y.reshape(T), reduction="none"
    )  # (T,)

    # Only take loss on completion positions
    # Position i in ce corresponds to predicting token i+1 from tokens 0..i
    # Completion tokens start at index n_prompt in full_ids
    # So we want ce[n_prompt-1 : n_prompt-1+n_completion] (predicting completion given prompt)
    start = max(0, n_prompt - 1)
    end = start + n_completion
    if end > T:
        end = T
    if start >= end:
        return float('nan')

    completion_ce = ce[start:end]
    return completion_ce.mean().item()


def main():
    parser = argparse.ArgumentParser(description="Paired evaluation")
    parser.add_argument("--model-size", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--test-paired", type=str, required=True,
                        help="JSONL file with paired test data")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Load model
    tokenizer = CharTokenizer().load(args.tokenizer)
    model = create_model(args.model_size, tokenizer.vocab_size, max_seq_len=args.seq_len)
    model.load_weights(args.weights)
    mx.eval(model.parameters())

    # Load paired test data
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

        c_nll = completion_nll(model, prompt_ids, correct_ids, args.seq_len)
        i_nll = completion_nll(model, prompt_ids, incorrect_ids, args.seq_len)

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

    # Per-type breakdown
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
            "n": nt,
            "avg_correct_nll": avg_c,
            "avg_incorrect_nll": avg_i,
            "delta": d,
            "accuracy": acc,
            "correct_wins": data["correct_wins"],
        }

    # Bootstrap CI for delta
    rng = np.random.RandomState(42)
    deltas_boot = []
    correct_arr = np.array(correct_nlls)
    incorrect_arr = np.array(incorrect_nlls)
    pair_deltas = incorrect_arr - correct_arr
    for _ in range(10000):
        idx = rng.choice(n, size=n, replace=True)
        deltas_boot.append(pair_deltas[idx].mean())
    ci_lo, ci_hi = np.percentile(deltas_boot, [2.5, 97.5])

    print(f"\n  Bootstrap 95% CI for delta: [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    # Wilcoxon signed-rank test
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
        "bootstrap_ci_95": [ci_lo, ci_hi],
        "wilcoxon_p": p_value,
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
