"""Evaluate next-token accuracy on correct vs incorrect test sets."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from model import create_model
from tokenizer import CharTokenizer


def evaluate_accuracy(model, data: mx.array, seq_len: int, batch_size: int,
                      n_batches: int = 50, seed: int = 0):
    """Compute next-token prediction accuracy on a dataset."""
    correct_counts = []
    total_counts = []
    n = data.shape[0]
    max_start = n - seq_len - 1
    rng = np.random.RandomState(seed)

    for i in range(n_batches):
        starts = rng.randint(0, max_start, size=batch_size)
        x = mx.stack([data[int(s):int(s) + seq_len] for s in starts])
        y = mx.stack([data[int(s) + 1:int(s) + seq_len + 1] for s in starts])

        logits = model(x)
        predictions = mx.argmax(logits, axis=-1)
        correct = (predictions == y).sum().item()
        total = y.size
        correct_counts.append(correct)
        total_counts.append(total)

    accuracy = sum(correct_counts) / sum(total_counts)
    return accuracy


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

    tokenizer = CharTokenizer().load(args.tokenizer)
    model = create_model(args.model_size, tokenizer.vocab_size, max_seq_len=args.seq_len)
    model.load_weights(args.weights)

    with open(args.test_correct) as f:
        correct_data = mx.array(tokenizer.encode(f.read()), dtype=mx.int32)
    with open(args.test_incorrect) as f:
        incorrect_data = mx.array(tokenizer.encode(f.read()), dtype=mx.int32)

    correct_acc = evaluate_accuracy(
        model, correct_data, args.seq_len, args.batch_size, args.n_batches, seed=0
    )
    incorrect_acc = evaluate_accuracy(
        model, incorrect_data, args.seq_len, args.batch_size, args.n_batches, seed=0
    )

    delta = correct_acc - incorrect_acc
    print(f"Accuracy on CORRECT:   {correct_acc:.4f} ({correct_acc*100:.2f}%)")
    print(f"Accuracy on INCORRECT: {incorrect_acc:.4f} ({incorrect_acc*100:.2f}%)")
    print(f"ΔAccuracy (cor - inc): {delta:+.4f} ({delta*100:+.2f}%)")

    if args.output:
        results = {
            "correct_accuracy": correct_acc,
            "incorrect_accuracy": incorrect_acc,
            "delta_accuracy": delta,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
