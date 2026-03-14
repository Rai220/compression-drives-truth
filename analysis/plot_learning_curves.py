"""
Learning curves: training loss + paired accuracy vs step for all model sizes.

Evaluates paired accuracy at each saved checkpoint to check behavioral convergence.
Addresses п.2: "are large models undertrained at 5000 steps?"
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

import mlx.core as mx
import mlx.nn as nn

from model import create_model
from tokenizer import CharTokenizer


def eval_paired_accuracy(model, tokenizer, paired_path: str, max_seq_len: int = 256,
                         max_pairs: int = 500) -> dict:
    """Quick paired accuracy eval on a subset of pairs."""
    pairs = []
    with open(paired_path) as f:
        for line in f:
            pairs.append(json.loads(line))

    if max_pairs < len(pairs):
        rng = np.random.RandomState(42)
        indices = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in indices]

    correct_wins = 0
    total = 0
    sum_delta = 0.0

    for pair in pairs:
        prompt_ids = tokenizer.encode(pair["prompt"])
        correct_ids = tokenizer.encode(pair["correct_completion"])
        incorrect_ids = tokenizer.encode(pair["incorrect_completion"])

        c_nll = _completion_mean_nll(model, prompt_ids, correct_ids, max_seq_len)
        i_nll = _completion_mean_nll(model, prompt_ids, incorrect_ids, max_seq_len)

        if not (np.isnan(c_nll) or np.isnan(i_nll)):
            total += 1
            if c_nll < i_nll:
                correct_wins += 1
            sum_delta += (i_nll - c_nll)

    accuracy = correct_wins / total if total > 0 else 0
    mean_delta = sum_delta / total if total > 0 else 0
    return {"accuracy": accuracy, "delta": mean_delta, "n": total}


def _completion_mean_nll(model, prompt_ids, completion_ids, max_seq_len):
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


def load_training_log(result_dir: str) -> list[dict]:
    log_path = Path(result_dir) / "training_log.json"
    if log_path.exists():
        with open(log_path) as f:
            return json.load(f)
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="results/figure_learning_curves.png")
    parser.add_argument("--output-json", type=str, default="results/learning_curves.json")
    parser.add_argument("--max-pairs", type=int, default=500,
                        help="Number of pairs for quick eval at each checkpoint")
    args = parser.parse_args()

    # Model configurations to evaluate
    configs = [
        # (label, size, result_dir, paired_file, condition)
        ("tiny random", "tiny", "results/mixed_50_50_tiny",
         "data/corpus/test_paired_random.jsonl", "random"),
        ("tiny coherent", "tiny", "results/coherent_50_50_tiny_seed42",
         "data/corpus/test_paired_coherent.jsonl", "coherent"),
        ("small random", "small", "results/mixed_50_50_small_seed42",
         "data/corpus/test_paired_random.jsonl", "random"),
        ("small coherent", "small", "results/coherent_50_50_small_seed42",
         "data/corpus/test_paired_coherent.jsonl", "coherent"),
        ("medium random", "medium", "results/mixed_50_50_medium_seed42",
         "data/corpus/test_paired_random.jsonl", "random"),
        ("medium coherent", "medium", "results/coherent_50_50_medium_seed42",
         "data/corpus/test_paired_coherent.jsonl", "coherent"),
        ("large random", "large", "results/mixed_50_50_large_seed42",
         "data/corpus/test_paired_random.jsonl", "random"),
        ("large coherent", "large", "results/coherent_50_50_large_seed42",
         "data/corpus/test_paired_coherent.jsonl", "coherent"),
    ]

    all_results = {}

    for label, size, result_dir, paired_path, condition in configs:
        print(f"\n{'='*60}")
        print(f"Processing: {label}")
        result_dir_p = Path(result_dir)

        # Load training log for loss curve
        train_log = load_training_log(result_dir)
        train_steps = [e["step"] for e in train_log]
        train_losses = [e["train_loss"] for e in train_log]
        val_losses = [e.get("val_loss") for e in train_log]

        # Find checkpoints
        ckpts = sorted(result_dir_p.glob("checkpoint_*.npz"))
        if not ckpts:
            print(f"  No checkpoints found in {result_dir}")
            continue

        # Load tokenizer once
        tok_path = result_dir_p / "tokenizer.json"
        tokenizer = CharTokenizer().load(str(tok_path))

        eval_steps = []
        eval_accs = []
        eval_deltas = []

        for ckpt in ckpts:
            step = int(ckpt.stem.split("_")[1])
            print(f"  Evaluating step {step}...", end=" ", flush=True)

            model = create_model(size, tokenizer.vocab_size, max_seq_len=256)
            model.load_weights(str(ckpt))
            mx.eval(model.parameters())

            result = eval_paired_accuracy(model, tokenizer, paired_path,
                                          max_pairs=args.max_pairs)
            eval_steps.append(step)
            eval_accs.append(result["accuracy"])
            eval_deltas.append(result["delta"])
            print(f"acc={result['accuracy']*100:.1f}%, Δ={result['delta']:+.4f}")

            # Free memory
            del model

        # Also eval final model if not already a checkpoint
        final_path = result_dir_p / "model_final.npz"
        if final_path.exists() and 5000 not in eval_steps:
            print(f"  Evaluating final model...", end=" ", flush=True)
            model = create_model(size, tokenizer.vocab_size, max_seq_len=256)
            model.load_weights(str(final_path))
            mx.eval(model.parameters())
            result = eval_paired_accuracy(model, tokenizer, paired_path,
                                          max_pairs=args.max_pairs)
            eval_steps.append(5000)
            eval_accs.append(result["accuracy"])
            eval_deltas.append(result["delta"])
            print(f"acc={result['accuracy']*100:.1f}%, Δ={result['delta']:+.4f}")
            del model

        all_results[label] = {
            "size": size,
            "condition": condition,
            "train_steps": train_steps,
            "train_losses": train_losses,
            "val_losses": [v for v in val_losses if v is not None],
            "eval_steps": eval_steps,
            "eval_accs": eval_accs,
            "eval_deltas": eval_deltas,
        }

    # Save results
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved data to {args.output_json}")

    # Plot: 2 rows × 4 cols (tiny/small/medium/large × loss/accuracy)
    sizes = ["tiny", "small", "medium", "large"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))

    colors = {"random": "#2196F3", "coherent": "#F44336"}

    for col, size in enumerate(sizes):
        # Top row: training loss
        ax_loss = axes[0, col]
        # Bottom row: paired accuracy
        ax_acc = axes[1, col]

        for label, data in all_results.items():
            if data["size"] != size:
                continue
            cond = data["condition"]
            color = colors[cond]

            # Loss curve
            if data["train_steps"]:
                ax_loss.plot(data["train_steps"], data["train_losses"],
                             color=color, alpha=0.7, linewidth=1.2, label=cond)
                if data["val_losses"] and len(data["val_losses"]) == len(data["train_steps"]):
                    ax_loss.plot(data["train_steps"], data["val_losses"],
                                 color=color, alpha=0.4, linewidth=1, linestyle="--")

            # Accuracy curve
            if data["eval_steps"]:
                ax_acc.plot(data["eval_steps"], [a * 100 for a in data["eval_accs"]],
                            color=color, marker="o", markersize=5, linewidth=1.5,
                            label=cond)

        ax_loss.set_title(f"{size}", fontsize=11, fontweight="bold")
        ax_loss.set_ylabel("Training loss" if col == 0 else "")
        ax_loss.grid(True, alpha=0.2)
        ax_loss.legend(fontsize=8)

        ax_acc.set_xlabel("Step")
        ax_acc.set_ylabel("Paired accuracy (%)" if col == 0 else "")
        ax_acc.axhline(50, color="gray", linestyle=":", alpha=0.4)
        ax_acc.grid(True, alpha=0.2)
        ax_acc.set_ylim(40, 100)
        ax_acc.legend(fontsize=8)

    fig.suptitle("Learning curves: training loss and paired accuracy by model size",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
