"""
Evaluate corpus-level loss on separate correct/incorrect test sets.

Two modes are supported:
1. `random_windows`: the legacy estimate used in early experiments. It samples
   random length-`seq_len` windows from the concatenated token stream.
2. `example_blocks`: a deterministic full-test estimate. It splits the raw text
   into problem blocks separated by blank lines and scores every token within
   every block without crossing example boundaries.

The deterministic `example_blocks` mode is the safer methodological baseline for
paper reporting because it matches the "held-out set of examples" description
more directly. The legacy random-window estimate is preserved for backward
compatibility with existing JSON artifacts.
"""

import argparse
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from model import create_model, MODEL_CONFIGS
from tokenizer import CharTokenizer


def evaluate_random_windows(
    model,
    data: mx.array,
    seq_len: int,
    batch_size: int,
    n_batches: int = 50,
    seed: int = 0,
):
    """Compute the legacy random-window estimate on a concatenated token stream."""
    losses = []
    n = data.shape[0]
    max_start = n - seq_len - 1
    rng = np.random.RandomState(seed)

    for i in range(n_batches):
        starts = rng.randint(0, max_start, size=batch_size)
        x = mx.stack([data[int(s):int(s) + seq_len] for s in starts])
        y = mx.stack([data[int(s) + 1:int(s) + seq_len + 1] for s in starts])

        logits = model(x)
        B, T, V = logits.shape
        loss = nn.losses.cross_entropy(
            logits.reshape(B * T, V), y.reshape(B * T), reduction="mean"
        )
        losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    return avg_loss, math.exp(min(avg_loss, 20))


def split_problem_blocks(raw_text: str) -> list[str]:
    """Split a corpus file into problem blocks separated by blank lines."""
    return [block.strip() for block in raw_text.split("\n\n") if block.strip()]


def evaluate_example_blocks(
    model,
    tokenizer: CharTokenizer,
    raw_text: str,
    seq_len: int,
):
    """Compute deterministic loss over all examples without crossing boundaries."""
    blocks = split_problem_blocks(raw_text)
    total_loss = 0.0
    total_tokens = 0
    skipped_examples = 0

    for block in blocks:
        ids = tokenizer.encode(block)
        if len(ids) < 2:
            skipped_examples += 1
            continue

        # Score the entire example in contiguous chunks. This preserves example
        # boundaries even when a single example exceeds the model context length.
        for start in range(0, len(ids) - 1, seq_len):
            end = min(start + seq_len + 1, len(ids))
            chunk = ids[start:end]
            if len(chunk) < 2:
                continue

            x = mx.array(chunk[:-1], dtype=mx.int32)[None, :]
            y = mx.array(chunk[1:], dtype=mx.int32)[None, :]
            logits = model(x)
            T = logits.shape[1]
            V = logits.shape[2]
            ce = nn.losses.cross_entropy(
                logits.reshape(T, V), y.reshape(T), reduction="none"
            )
            total_loss += ce.sum().item()
            total_tokens += len(chunk) - 1

    if total_tokens == 0:
        raise ValueError("No valid example blocks found for deterministic evaluation")

    avg_loss = total_loss / total_tokens
    return {
        "avg_loss": avg_loss,
        "ppl": math.exp(min(avg_loss, 20)),
        "n_examples": len(blocks),
        "skipped_examples": skipped_examples,
        "n_scored_tokens": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--test-correct", type=str, required=True)
    parser.add_argument("--test-incorrect", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-batches", type=int, default=50)
    parser.add_argument(
        "--mode",
        type=str,
        default="random_windows",
        choices=["random_windows", "example_blocks", "both"],
        help="Evaluation mode. `random_windows` preserves legacy behavior.",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Load
    tokenizer = CharTokenizer().load(args.tokenizer)
    model = create_model(args.model_size, tokenizer.vocab_size, max_seq_len=args.seq_len)
    model.load_weights(args.weights)

    with open(args.test_correct) as f:
        correct_text = f.read()
    with open(args.test_incorrect) as f:
        incorrect_text = f.read()

    correct_data = mx.array(tokenizer.encode(correct_text), dtype=mx.int32)
    incorrect_data = mx.array(tokenizer.encode(incorrect_text), dtype=mx.int32)

    print(f"Model: {args.model_size} | Weights: {args.weights}")
    print(f"Correct test tokens: {correct_data.shape[0]:,}")
    print(f"Incorrect test tokens: {incorrect_data.shape[0]:,}")
    print("-" * 50)

    results = {
        "model_size": args.model_size,
        "weights": args.weights,
        "mode": args.mode,
    }

    if args.mode in {"random_windows", "both"}:
        correct_loss, correct_ppl = evaluate_random_windows(
            model, correct_data, args.seq_len, args.batch_size, args.n_batches, seed=0
        )
        incorrect_loss, incorrect_ppl = evaluate_random_windows(
            model, incorrect_data, args.seq_len, args.batch_size, args.n_batches, seed=0
        )

        ratio = incorrect_loss / correct_loss
        print("Random-window estimate:")
        print(f"  Correct:   loss={correct_loss:.4f}  ppl={correct_ppl:.3f}")
        print(f"  Incorrect: loss={incorrect_loss:.4f}  ppl={incorrect_ppl:.3f}")
        print(f"  Ratio (incorrect/correct): {ratio:.4f}")

        results.update({
            "correct_loss": correct_loss,
            "correct_ppl": correct_ppl,
            "incorrect_loss": incorrect_loss,
            "incorrect_ppl": incorrect_ppl,
            "ratio_inc_over_cor": ratio,
            "prefers": "correct" if correct_loss < incorrect_loss else "incorrect",
            "delta": abs(incorrect_loss - correct_loss),
            "window_estimate": {
                "correct_loss": correct_loss,
                "correct_ppl": correct_ppl,
                "incorrect_loss": incorrect_loss,
                "incorrect_ppl": incorrect_ppl,
                "ratio_inc_over_cor": ratio,
                "prefers": "correct" if correct_loss < incorrect_loss else "incorrect",
                "delta_signed": incorrect_loss - correct_loss,
                "n_batches": args.n_batches,
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
            },
        })
        print()

    if args.mode in {"example_blocks", "both"}:
        correct_full = evaluate_example_blocks(model, tokenizer, correct_text, args.seq_len)
        incorrect_full = evaluate_example_blocks(model, tokenizer, incorrect_text, args.seq_len)
        full_ratio = incorrect_full["avg_loss"] / correct_full["avg_loss"]

        print("Deterministic example-block estimate:")
        print(
            f"  Correct:   loss={correct_full['avg_loss']:.4f}  ppl={correct_full['ppl']:.3f}  "
            f"examples={correct_full['n_examples']}  tokens={correct_full['n_scored_tokens']}"
        )
        print(
            f"  Incorrect: loss={incorrect_full['avg_loss']:.4f}  ppl={incorrect_full['ppl']:.3f}  "
            f"examples={incorrect_full['n_examples']}  tokens={incorrect_full['n_scored_tokens']}"
        )
        print(f"  Ratio (incorrect/correct): {full_ratio:.4f}")
        print()

        if args.mode == "example_blocks":
            results.update({
                "correct_loss": correct_full["avg_loss"],
                "correct_ppl": correct_full["ppl"],
                "incorrect_loss": incorrect_full["avg_loss"],
                "incorrect_ppl": incorrect_full["ppl"],
                "ratio_inc_over_cor": full_ratio,
                "prefers": "correct" if correct_full["avg_loss"] < incorrect_full["avg_loss"] else "incorrect",
                "delta": abs(incorrect_full["avg_loss"] - correct_full["avg_loss"]),
            })

        results["example_block_estimate"] = {
            "correct_loss": correct_full["avg_loss"],
            "correct_ppl": correct_full["ppl"],
            "incorrect_loss": incorrect_full["avg_loss"],
            "incorrect_ppl": incorrect_full["ppl"],
            "ratio_inc_over_cor": full_ratio,
            "prefers": "correct" if correct_full["avg_loss"] < incorrect_full["avg_loss"] else "incorrect",
            "delta_signed": incorrect_full["avg_loss"] - correct_full["avg_loss"],
            "correct_n_examples": correct_full["n_examples"],
            "incorrect_n_examples": incorrect_full["n_examples"],
            "correct_scored_tokens": correct_full["n_scored_tokens"],
            "incorrect_scored_tokens": incorrect_full["n_scored_tokens"],
            "correct_skipped_examples": correct_full["skipped_examples"],
            "incorrect_skipped_examples": incorrect_full["skipped_examples"],
            "seq_len": args.seq_len,
        }

    if "correct_loss" in results and "incorrect_loss" in results:
        if results["correct_loss"] < results["incorrect_loss"]:
            print(
                f">>> Model prefers CORRECT examples (lower loss by "
                f"{results['incorrect_loss'] - results['correct_loss']:.4f})"
            )
        elif results["incorrect_loss"] < results["correct_loss"]:
            print(
                f">>> Model prefers INCORRECT examples (lower loss by "
                f"{results['correct_loss'] - results['incorrect_loss']:.4f})"
            )
        else:
            print(">>> No preference detected")

    # Save results
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")

    return results


if __name__ == "__main__":
    main()
