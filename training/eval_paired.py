"""
Paired evaluation: same prompt, two completions (correct vs incorrect).

For each paired problem:
1. Encode prompt + correct_completion and prompt + incorrect_completion
2. Compute NLL only on the completion tokens (conditioned on prompt)
3. Report per-pair and aggregate statistics

Primary metric:
- mean NLL on completion tokens

Auxiliary robustness metrics:
- sum NLL on completion tokens
- mean NLL on the first min(len(correct), len(incorrect)) scored completion tokens

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
from tokenizer import CharTokenizer, load_tokenizer


def completion_nll_stats(model, prompt_ids: list[int], completion_ids: list[int],
                         max_seq_len: int) -> dict[str, object]:
    """Compute completion-token NLL statistics, conditioned on prompt.

    Runs the full sequence through the model, but only computes loss
    on positions corresponding to completion tokens.
    """
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
        return {
            "mean_nll": float("nan"),
            "sum_nll": float("nan"),
            "token_losses": np.array([], dtype=float),
            "n_scored_tokens": 0,
            "n_completion_tokens": n_completion,
        }

    completion_ce = ce[start:end]
    token_losses = np.array(completion_ce.tolist(), dtype=float)
    return {
        "mean_nll": float(token_losses.mean()),
        "sum_nll": float(token_losses.sum()),
        "token_losses": token_losses,
        "n_scored_tokens": int(token_losses.size),
        "n_completion_tokens": n_completion,
    }


def matched_prefix_mean(stats_a: dict[str, object], stats_b: dict[str, object]) -> tuple[float, float]:
    """Compare mean NLL on the shared minimum number of scored completion tokens."""
    losses_a = stats_a["token_losses"]
    losses_b = stats_b["token_losses"]
    n = min(len(losses_a), len(losses_b))
    if n == 0:
        return float("nan"), float("nan")
    return float(losses_a[:n].mean()), float(losses_b[:n].mean())


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
    tokenizer = load_tokenizer(args.tokenizer)
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
    correct_sum_nlls = []
    incorrect_sum_nlls = []
    correct_matched_mean_nlls = []
    incorrect_matched_mean_nlls = []
    completion_lengths_correct = []
    completion_lengths_incorrect = []
    scored_lengths_correct = []
    scored_lengths_incorrect = []
    correct_wins = 0
    correct_sum_wins = 0
    correct_matched_mean_wins = 0
    by_type = {}

    for pair in pairs:
        prompt_ids = tokenizer.encode(pair["prompt"])
        correct_ids = tokenizer.encode(pair["correct_completion"])
        incorrect_ids = tokenizer.encode(pair["incorrect_completion"])

        c_stats = completion_nll_stats(model, prompt_ids, correct_ids, args.seq_len)
        i_stats = completion_nll_stats(model, prompt_ids, incorrect_ids, args.seq_len)
        c_nll = c_stats["mean_nll"]
        i_nll = i_stats["mean_nll"]
        c_sum = c_stats["sum_nll"]
        i_sum = i_stats["sum_nll"]
        c_matched, i_matched = matched_prefix_mean(c_stats, i_stats)

        if any(math.isnan(x) for x in [c_nll, i_nll, c_sum, i_sum, c_matched, i_matched]):
            continue

        correct_nlls.append(c_nll)
        incorrect_nlls.append(i_nll)
        correct_sum_nlls.append(c_sum)
        incorrect_sum_nlls.append(i_sum)
        correct_matched_mean_nlls.append(c_matched)
        incorrect_matched_mean_nlls.append(i_matched)
        completion_lengths_correct.append(c_stats["n_completion_tokens"])
        completion_lengths_incorrect.append(i_stats["n_completion_tokens"])
        scored_lengths_correct.append(c_stats["n_scored_tokens"])
        scored_lengths_incorrect.append(i_stats["n_scored_tokens"])

        if c_nll < i_nll:
            correct_wins += 1
        if c_sum < i_sum:
            correct_sum_wins += 1
        if c_matched < i_matched:
            correct_matched_mean_wins += 1

        ptype = pair.get("problem_type", "unknown")
        if ptype not in by_type:
            by_type[ptype] = {
                "correct_nlls": [],
                "incorrect_nlls": [],
                "correct_sum_nlls": [],
                "incorrect_sum_nlls": [],
                "correct_matched_mean_nlls": [],
                "incorrect_matched_mean_nlls": [],
                "correct_wins": 0,
                "correct_sum_wins": 0,
                "correct_matched_mean_wins": 0,
            }
        by_type[ptype]["correct_nlls"].append(c_nll)
        by_type[ptype]["incorrect_nlls"].append(i_nll)
        by_type[ptype]["correct_sum_nlls"].append(c_sum)
        by_type[ptype]["incorrect_sum_nlls"].append(i_sum)
        by_type[ptype]["correct_matched_mean_nlls"].append(c_matched)
        by_type[ptype]["incorrect_matched_mean_nlls"].append(i_matched)
        if c_nll < i_nll:
            by_type[ptype]["correct_wins"] += 1
        if c_sum < i_sum:
            by_type[ptype]["correct_sum_wins"] += 1
        if c_matched < i_matched:
            by_type[ptype]["correct_matched_mean_wins"] += 1

    n = len(correct_nlls)
    if n == 0:
        print("No valid pairs!")
        return

    avg_correct = sum(correct_nlls) / n
    avg_incorrect = sum(incorrect_nlls) / n
    delta = avg_incorrect - avg_correct
    accuracy = correct_wins / n
    avg_correct_sum = sum(correct_sum_nlls) / n
    avg_incorrect_sum = sum(incorrect_sum_nlls) / n
    delta_sum = avg_incorrect_sum - avg_correct_sum
    sum_accuracy = correct_sum_wins / n
    avg_correct_matched = sum(correct_matched_mean_nlls) / n
    avg_incorrect_matched = sum(incorrect_matched_mean_nlls) / n
    delta_matched = avg_incorrect_matched - avg_correct_matched
    matched_accuracy = correct_matched_mean_wins / n

    print(f"\nAggregate ({n} pairs):")
    print(f"  Avg NLL correct:   {avg_correct:.4f}")
    print(f"  Avg NLL incorrect: {avg_incorrect:.4f}")
    print(f"  Delta (inc - cor): {delta:+.4f}")
    print(f"  Pair accuracy:     {accuracy:.3f} ({correct_wins}/{n})")
    print(f"  Sum-NLL delta:     {delta_sum:+.4f}  accuracy={sum_accuracy:.3f} ({correct_sum_wins}/{n})")
    print(
        f"  Length-matched mean delta: {delta_matched:+.4f}  "
        f"accuracy={matched_accuracy:.3f} ({correct_matched_mean_wins}/{n})"
    )

    # Per-type breakdown
    print(f"\nPer-type breakdown:")
    type_results = {}
    for ptype, data in sorted(by_type.items()):
        nt = len(data["correct_nlls"])
        avg_c = sum(data["correct_nlls"]) / nt
        avg_i = sum(data["incorrect_nlls"]) / nt
        d = avg_i - avg_c
        acc = data["correct_wins"] / nt
        avg_c_sum = sum(data["correct_sum_nlls"]) / nt
        avg_i_sum = sum(data["incorrect_sum_nlls"]) / nt
        avg_c_matched = sum(data["correct_matched_mean_nlls"]) / nt
        avg_i_matched = sum(data["incorrect_matched_mean_nlls"]) / nt
        print(
            f"  {ptype:12s}: delta={d:+.4f}  accuracy={acc:.3f} "
            f"({data['correct_wins']}/{nt})"
        )
        type_results[ptype] = {
            "n": nt,
            "avg_correct_nll": avg_c,
            "avg_incorrect_nll": avg_i,
            "delta": d,
            "accuracy": acc,
            "correct_wins": data["correct_wins"],
            "avg_correct_sum_nll": avg_c_sum,
            "avg_incorrect_sum_nll": avg_i_sum,
            "sum_delta": avg_i_sum - avg_c_sum,
            "sum_accuracy": data["correct_sum_wins"] / nt,
            "sum_correct_wins": data["correct_sum_wins"],
            "avg_correct_matched_mean_nll": avg_c_matched,
            "avg_incorrect_matched_mean_nll": avg_i_matched,
            "matched_mean_delta": avg_i_matched - avg_c_matched,
            "matched_mean_accuracy": data["correct_matched_mean_wins"] / nt,
            "matched_mean_correct_wins": data["correct_matched_mean_wins"],
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
        "robustness": {
            "sum_nll": {
                "avg_correct": avg_correct_sum,
                "avg_incorrect": avg_incorrect_sum,
                "delta": delta_sum,
                "pair_accuracy": sum_accuracy,
                "correct_wins": correct_sum_wins,
            },
            "length_matched_mean_nll": {
                "avg_correct": avg_correct_matched,
                "avg_incorrect": avg_incorrect_matched,
                "delta": delta_matched,
                "pair_accuracy": matched_accuracy,
                "correct_wins": correct_matched_mean_wins,
            },
            "completion_lengths": {
                "correct_mean": float(np.mean(completion_lengths_correct)),
                "incorrect_mean": float(np.mean(completion_lengths_incorrect)),
                "correct_min": int(min(completion_lengths_correct)),
                "incorrect_min": int(min(completion_lengths_incorrect)),
                "correct_max": int(max(completion_lengths_correct)),
                "incorrect_max": int(max(completion_lengths_incorrect)),
                "correct_scored_mean": float(np.mean(scored_lengths_correct)),
                "incorrect_scored_mean": float(np.mean(scored_lengths_incorrect)),
            },
        },
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
