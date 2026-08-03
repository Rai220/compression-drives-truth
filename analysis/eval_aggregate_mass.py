"""
Aggregate probability mass of the correct mode vs all false modes.

Pair accuracy compares the correct completion against ONE sampled false
completion. That comparison can favour the correct answer purely because the
false side is split across several modes: a model that merely reproduces
training frequencies assigns 0.5 to the correct completion and 0.5/N to each of
N false ones, so the correct completion wins every pairwise comparison for
N >= 2 without any preference for the correct answer system.

This script removes that confound by summing over the whole false support:

    share = P(correct) / (P(correct) + sum_i P(false_i))

Frequency matching predicts share = 0.5 for every N. A preference for the more
compressible answer system predicts share > 0.5, growing with N.

Two variants are reported:
  full    -- log P over the entire completion (sum of token NLL)
  branch  -- probability of the first token at which the completions diverge,
             conditioned on the identical shared context. Purely local, so it
             is unaffected by completion length or by how predictable the rest
             of a completion is.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from model import create_model
from tokenizer import load_tokenizer


def logsumexp(values: list[float]) -> float:
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def batch_completion_logprobs(model, prompt_ids, completions_ids, max_seq_len):
    """Sum log P over completion tokens for several completions of one prompt.

    All completions share the prompt, so they are padded into one batch.
    Returns (logprobs, logits_at_branch_positions) where the second element is
    the full-vocabulary log-softmax at every position of the first sequence.
    """
    n_prompt = len(prompt_ids)
    seqs = [prompt_ids + c for c in completions_ids]
    max_len = max(len(s) for s in seqs)
    if max_len > max_seq_len + 1:
        return None, None

    padded = np.zeros((len(seqs), max_len), dtype=np.int32)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = s

    x = mx.array(padded[:, :-1])
    y = mx.array(padded[:, 1:])
    logits = model(x)
    logprobs_all = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    B, T, V = logits.shape
    token_lp = mx.take_along_axis(
        logprobs_all.reshape(B * T, V),
        y.reshape(B * T, 1),
        axis=-1,
    ).reshape(B, T)
    token_lp = np.array(token_lp.tolist(), dtype=np.float64)

    start = max(0, n_prompt - 1)
    out = []
    for i, c in enumerate(completions_ids):
        end = min(start + len(c), T)
        out.append(float(token_lp[i, start:end].sum()))

    return out, logprobs_all[0]


def branch_share(model_logprobs, prompt_ids, completions_ids):
    """Local share of probability at the single token where completions diverge.

    Only defined when every false completion differs from the correct one at
    that token. When some false completions still agree there (they diverge
    further downstream) their mass cannot be attributed at this position, and
    counting only the rest would inflate the correct share, so the problem is
    excluded instead.
    """
    correct = completions_ids[0]
    wrongs = completions_ids[1:]

    k = 0
    limit = min(len(c) for c in completions_ids)
    while k < limit and len({c[k] for c in completions_ids}) == 1:
        k += 1
    if k >= limit:
        return None

    correct_tok = correct[k]
    wrong_toks = {w[k] for w in wrongs}
    if correct_tok in wrong_toks:
        return None

    pos = len(prompt_ids) - 1 + k
    if pos >= model_logprobs.shape[0]:
        return None

    lp = np.array(model_logprobs[pos].tolist(), dtype=np.float64)
    p_correct = math.exp(lp[correct_tok])
    p_wrong = sum(math.exp(lp[t]) for t in wrong_toks)
    return p_correct / (p_correct + p_wrong), len(wrong_toks)


def main():
    parser = argparse.ArgumentParser(description="Aggregate probability mass eval")
    parser.add_argument("--model-size", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--support", type=str, required=True,
                        help="JSONL from gen_multirule_support.py")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    model = create_model(args.model_size, tokenizer.vocab_size, max_seq_len=args.seq_len)
    model.load_weights(args.weights)
    mx.eval(model.parameters())

    problems = []
    with open(args.support) as f:
        for line in f:
            problems.append(json.loads(line))
    if args.limit:
        problems = problems[:args.limit]

    shares = []
    branch_shares = []
    pair_wins_sum = 0
    pair_wins_mean = 0
    pair_n = 0
    n_modes = []
    skipped = 0
    by_type = {}
    len_deltas = []
    equal_len_shares = []

    for p in problems:
        prompt_ids = tokenizer.encode(p["prompt"])
        correct_ids = tokenizer.encode(p["correct_completion"])
        wrong_ids = [tokenizer.encode(w) for w in p["wrong_completions"]]

        logps, lp_grid = batch_completion_logprobs(
            model, prompt_ids, [correct_ids] + wrong_ids, args.seq_len
        )
        if logps is None:
            skipped += 1
            continue

        lp_c, lp_w = logps[0], logps[1:]
        share = 1.0 / (1.0 + math.exp(logsumexp(lp_w) - lp_c))
        shares.append(share)
        n_modes.append(len(lp_w))
        by_type.setdefault(p["problem_type"], []).append((share, len(lp_w)))

        # Longer strings carry lower probability by construction, so a length
        # asymmetry between the correct and the false completions would inflate
        # the share on its own. Track it, and keep the equal-length subset.
        len_c = len(correct_ids)
        len_w = [len(w) for w in wrong_ids]
        len_deltas.append(float(np.mean(len_w) - len_c))
        if all(l == len_c for l in len_w):
            equal_len_shares.append(share)

        bs = branch_share(lp_grid, prompt_ids, [correct_ids] + wrong_ids)
        if bs is not None:
            branch_shares.append(bs[0])

        # Expected pair accuracy: the paper's metric averaged over which false
        # mode happens to be sampled as the comparison completion.
        mean_c = lp_c / max(1, len(correct_ids))
        wins_mean = [
            1.0 if mean_c > (lp / max(1, len(ids))) else 0.0
            for lp, ids in zip(lp_w, wrong_ids)
        ]
        wins_sum = [1.0 if lp_c > lp else 0.0 for lp in lp_w]
        pair_n += 1
        pair_wins_mean += sum(wins_mean) / len(wins_mean)
        pair_wins_sum += sum(wins_sum) / len(wins_sum)

    shares = np.array(shares)
    branch_arr = np.array(branch_shares)

    rng = np.random.RandomState(42)
    boot = [shares[rng.choice(len(shares), len(shares), replace=True)].mean()
            for _ in range(5000)]
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    result = {
        "weights": args.weights,
        "support": args.support,
        "n_problems": int(len(shares)),
        "skipped": skipped,
        "mean_false_modes": float(np.mean(n_modes)),
        "aggregate_share_mean": float(shares.mean()),
        "aggregate_share_median": float(np.median(shares)),
        "aggregate_share_ci95": [float(ci_lo), float(ci_hi)],
        "frac_share_above_half": float((shares > 0.5).mean()),
        "branch_share_mean": float(branch_arr.mean()) if len(branch_arr) else None,
        "branch_share_n": int(len(branch_arr)),
        "length_delta_mean": float(np.mean(len_deltas)),
        "equal_length_share_mean": (float(np.mean(equal_len_shares))
                                    if equal_len_shares else None),
        "equal_length_n": int(len(equal_len_shares)),
        "pair_accuracy_sum_nll": pair_wins_sum / pair_n if pair_n else None,
        "pair_accuracy_mean_nll": pair_wins_mean / pair_n if pair_n else None,
        "pair_n": pair_n,
        "by_type": {
            t: {
                "n": len(v),
                "mean_false_modes": float(np.mean([m for _, m in v])),
                "aggregate_share_mean": float(np.mean([s for s, _ in v])),
            }
            for t, v in sorted(by_type.items())
        },
    }

    if not args.quiet:
        print(f"{args.weights}")
        print(f"  problems={result['n_problems']} false modes/problem={result['mean_false_modes']:.2f}")
        print(f"  aggregate share  = {result['aggregate_share_mean']:.4f} "
              f"[{ci_lo:.4f}, {ci_hi:.4f}]   (frequency matching predicts 0.5)")
        print(f"  branch share     = {result['branch_share_mean']:.4f}"
              if result["branch_share_mean"] is not None else "  branch share     = n/a")
        print(f"  pair accuracy    = {result['pair_accuracy_mean_nll']:.4f} (mean NLL, as in paper)")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    main()
