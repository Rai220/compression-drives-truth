"""
Evaluate perplexity of a trained model on separate correct/incorrect test sets.

This is the core measurement for the Compression Truth Bias hypothesis:
if the model develops a bias toward truth, it should have lower perplexity
on correct examples even when trained on 50/50 mixed data.
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


def evaluate_perplexity(model, data: mx.array, seq_len: int, batch_size: int,
                        n_batches: int = 50, seed: int = 0):
    """Compute average perplexity on a dataset."""
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
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Load
    tokenizer = CharTokenizer().load(args.tokenizer)
    model = create_model(args.model_size, tokenizer.vocab_size, max_seq_len=args.seq_len)
    model.load_weights(args.weights)

    with open(args.test_correct) as f:
        correct_data = mx.array(tokenizer.encode(f.read()), dtype=mx.int32)
    with open(args.test_incorrect) as f:
        incorrect_data = mx.array(tokenizer.encode(f.read()), dtype=mx.int32)

    print(f"Model: {args.model_size} | Weights: {args.weights}")
    print(f"Correct test tokens: {correct_data.shape[0]:,}")
    print(f"Incorrect test tokens: {incorrect_data.shape[0]:,}")
    print("-" * 50)

    # Evaluate
    correct_loss, correct_ppl = evaluate_perplexity(
        model, correct_data, args.seq_len, args.batch_size, args.n_batches, seed=0
    )
    incorrect_loss, incorrect_ppl = evaluate_perplexity(
        model, incorrect_data, args.seq_len, args.batch_size, args.n_batches, seed=0
    )

    ratio = incorrect_loss / correct_loss

    print(f"Correct:   loss={correct_loss:.4f}  ppl={correct_ppl:.3f}")
    print(f"Incorrect: loss={incorrect_loss:.4f}  ppl={incorrect_ppl:.3f}")
    print(f"Ratio (incorrect/correct): {ratio:.4f}")
    print()
    if correct_loss < incorrect_loss:
        print(f">>> Model prefers CORRECT examples (lower loss by {incorrect_loss - correct_loss:.4f})")
    elif incorrect_loss < correct_loss:
        print(f">>> Model prefers INCORRECT examples (lower loss by {correct_loss - incorrect_loss:.4f})")
    else:
        print(">>> No preference detected")

    # Save results
    results = {
        "model_size": args.model_size,
        "weights": args.weights,
        "correct_loss": correct_loss,
        "correct_ppl": correct_ppl,
        "incorrect_loss": incorrect_loss,
        "incorrect_ppl": incorrect_ppl,
        "ratio_inc_over_cor": ratio,
        "prefers": "correct" if correct_loss < incorrect_loss else "incorrect",
        "delta": abs(incorrect_loss - correct_loss),
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")

    return results


if __name__ == "__main__":
    main()
