"""
Histogram of per-pair NLL differences: NLL(incorrect) - NLL(correct).

Shows distribution shape for random vs coherent conditions.
Addresses п.7: "why do mean DLoss and pair accuracy sometimes diverge?"
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add training dir to path for model/tokenizer imports
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

import mlx.core as mx
import mlx.nn as nn

from model import create_model
from tokenizer import CharTokenizer


def compute_pair_deltas(model, tokenizer, paired_path: str, max_seq_len: int = 256):
    """Compute per-pair NLL(incorrect) - NLL(correct) for all pairs."""
    pairs = []
    with open(paired_path) as f:
        for line in f:
            pairs.append(json.loads(line))

    deltas = []
    types = []
    for pair in pairs:
        prompt_ids = tokenizer.encode(pair["prompt"])
        correct_ids = tokenizer.encode(pair["correct_completion"])
        incorrect_ids = tokenizer.encode(pair["incorrect_completion"])

        c_nll = _completion_mean_nll(model, prompt_ids, correct_ids, max_seq_len)
        i_nll = _completion_mean_nll(model, prompt_ids, incorrect_ids, max_seq_len)

        if not (np.isnan(c_nll) or np.isnan(i_nll)):
            deltas.append(i_nll - c_nll)
            types.append(pair.get("problem_type", "unknown"))

    return np.array(deltas), types


def _completion_mean_nll(model, prompt_ids, completion_ids, max_seq_len):
    """Mean NLL on completion tokens."""
    full_ids = prompt_ids + completion_ids
    n_prompt = len(prompt_ids)
    n_completion = len(completion_ids)
    if n_completion == 0:
        return float("nan")

    if len(full_ids) > max_seq_len + 1:
        excess = len(full_ids) - (max_seq_len + 1)
        full_ids = full_ids[excess:]
        n_prompt = max(0, n_prompt - excess)

    x = mx.array(full_ids[:-1], dtype=mx.int32)[None, :]
    y = mx.array(full_ids[1:], dtype=mx.int32)[None, :]
    logits = model(x)
    T, V = logits.shape[1], logits.shape[2]
    ce = nn.losses.cross_entropy(logits.reshape(T, V), y.reshape(T), reduction="none")

    start = max(0, n_prompt - 1)
    end = min(start + n_completion, T)
    if start >= end:
        return float("nan")

    return float(ce[start:end].mean().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True,
                        help="Config strings: label:model_size:weights:tokenizer:paired_file")
    parser.add_argument("--output", type=str, default="results/figure_nll_histogram.png")
    parser.add_argument("--output-json", type=str, default="results/nll_pair_deltas.json")
    args = parser.parse_args()

    all_data = {}
    for cfg in args.configs:
        parts = cfg.split(":")
        label, model_size, weights, tok_path, paired_path = parts
        print(f"Computing deltas for {label}...")

        tokenizer = CharTokenizer().load(tok_path)
        model = create_model(model_size, tokenizer.vocab_size, max_seq_len=256)
        model.load_weights(weights)
        mx.eval(model.parameters())

        deltas, types = compute_pair_deltas(model, tokenizer, paired_path)
        all_data[label] = {
            "deltas": deltas.tolist(),
            "types": types,
            "mean": float(deltas.mean()),
            "median": float(np.median(deltas)),
            "std": float(deltas.std()),
            "frac_positive": float((deltas > 0).mean()),
            "n": len(deltas),
        }
        print(f"  n={len(deltas)}, mean={deltas.mean():.4f}, median={np.median(deltas):.4f}, "
              f"frac>0={((deltas > 0).mean())*100:.1f}%")

    # Save raw data
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(all_data, f)
    print(f"Saved deltas to {args.output_json}")

    # Plot
    n_plots = len(all_data)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5), sharey=True)
    if n_plots == 1:
        axes = [axes]

    colors = {"random": "#2196F3", "coherent": "#F44336"}

    for ax, (label, data) in zip(axes, all_data.items()):
        deltas = np.array(data["deltas"])
        color = colors.get(label.split("_")[0], "#666666")

        # Clip for visualization
        clip_val = np.percentile(np.abs(deltas), 99)
        deltas_clipped = np.clip(deltas, -clip_val, clip_val)

        ax.hist(deltas_clipped, bins=80, color=color, alpha=0.7, edgecolor="white",
                linewidth=0.3, density=True)
        ax.axvline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
        ax.axvline(deltas.mean(), color="darkred", linestyle="--", linewidth=1.2,
                   label=f"mean={deltas.mean():.3f}")
        ax.axvline(np.median(deltas), color="darkblue", linestyle=":", linewidth=1.2,
                   label=f"median={np.median(deltas):.3f}")

        ax.set_xlabel("NLL(incorrect) − NLL(correct)", fontsize=10)
        ax.set_title(f"{label}\nacc={data['frac_positive']*100:.1f}%, "
                     f"mean Δ={data['mean']:.3f}", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Density", fontsize=10)
    fig.suptitle("Distribution of per-pair NLL differences", fontsize=13, y=1.02)
    fig.tight_layout()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
