"""
Generation sanity check: greedy decode + automated verification.

For each prompt from the paired test set:
1. Greedy-decode a completion from the trained model
2. Extract the "Answer:" line
3. Verify correctness using SymPy (or arithmetic eval)

Compares generative accuracy between models trained on random vs coherent errors.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sympy as sp
from sympy import symbols, simplify, Rational

from model import create_model, MODEL_CONFIGS
from tokenizer import CharTokenizer


# ---------------------------------------------------------------------------
# Greedy decoding
# ---------------------------------------------------------------------------

def greedy_generate(model, tokenizer, prompt: str, max_tokens: int = 300) -> str:
    """Greedy decode until double-newline or max_tokens."""
    tokens = tokenizer.encode(prompt)
    input_len = len(tokens)
    tokens = mx.array([tokens], dtype=mx.int32)

    for _ in range(max_tokens):
        logits = model(tokens[:, -model.max_seq_len:])
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        tokens = mx.concatenate([tokens, next_token.reshape(1, 1)], axis=1)
        mx.eval(tokens)

        # Stop at double newline (problem separator)
        if tokens.shape[1] > input_len + 5:
            last_chars = tokenizer.decode(tokens[0, -3:].tolist())
            if last_chars.endswith("\n\n"):
                break

    generated = tokenizer.decode(tokens[0, input_len:].tolist())
    return generated.strip()


# ---------------------------------------------------------------------------
# Answer extraction and verification
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str | None:
    """Extract the Answer: line from generated text."""
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("Answer:"):
            return line[len("Answer:"):].strip()
    return None


def normalize_expr(s: str) -> str:
    """Normalize a math expression string for comparison."""
    s = s.strip()
    s = s.replace("·", "*").replace("×", "*").replace("^", "**")
    s = s.replace(" ", "")
    # Insert * for implicit multiplication: )( , digit( , )digit
    s = re.sub(r'\)\(', ')*(', s)
    s = re.sub(r'(\d)\(', r'\1*(', s)
    s = re.sub(r'\)(\d)', r')*\1', s)
    s = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', s)
    return s


def verify_arithmetic(answer: str, correct_answer: str) -> bool:
    """Verify arithmetic answer (integer or simple fraction)."""
    try:
        a = normalize_expr(answer)
        c = normalize_expr(correct_answer)
        # Try numeric comparison
        return abs(float(sp.sympify(a)) - float(sp.sympify(c))) < 1e-9
    except Exception:
        return normalize_expr(answer) == normalize_expr(correct_answer)


def verify_equation(answer: str, correct_answer: str) -> bool:
    """Verify equation solution (x = value or x = a or x = b)."""
    try:
        a_norm = normalize_expr(answer)
        c_norm = normalize_expr(correct_answer)

        # Handle "x = a or x = b" format
        def extract_solutions(s):
            s = s.replace("x=", "").replace("x =", "")
            parts = re.split(r'\bor\b', s)
            vals = set()
            for p in parts:
                p = p.strip()
                if p:
                    vals.add(float(sp.sympify(p)))
            return vals

        a_sols = extract_solutions(answer)
        c_sols = extract_solutions(correct_answer)
        if a_sols and c_sols:
            return a_sols == c_sols
    except Exception:
        pass
    return normalize_expr(answer) == normalize_expr(correct_answer)


def verify_derivative(answer: str, correct_answer: str) -> bool:
    """Verify derivative answer (symbolic expression)."""
    try:
        x = symbols("x")
        a = sp.sympify(normalize_expr(answer))
        c = sp.sympify(normalize_expr(correct_answer))
        return simplify(a - c) == 0
    except Exception:
        return normalize_expr(answer) == normalize_expr(correct_answer)


def verify_algebra(answer: str, correct_answer: str) -> bool:
    """Verify algebra/factoring answer."""
    try:
        x = symbols("x")
        a = sp.sympify(normalize_expr(answer))
        c = sp.sympify(normalize_expr(correct_answer))
        return sp.expand(a - c) == 0
    except Exception:
        return normalize_expr(answer) == normalize_expr(correct_answer)


VERIFIERS = {
    "arithmetic": verify_arithmetic,
    "equation": verify_equation,
    "derivative": verify_derivative,
    "algebra": verify_algebra,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generation sanity check")
    parser.add_argument("--model-size", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--test-paired", type=str, required=True,
                        help="Paired JSONL file (uses prompts + correct answers)")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of problems to test")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for sampling subset of problems")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Load model
    tokenizer = CharTokenizer().load(args.tokenizer)
    model = create_model(args.model_size, tokenizer.vocab_size, max_seq_len=args.seq_len)
    model.load_weights(args.weights)
    mx.eval(model.parameters())

    # Load test data
    pairs = []
    with open(args.test_paired) as f:
        for line in f:
            pairs.append(json.loads(line))

    # Sample subset
    rng = np.random.RandomState(args.seed)
    if args.n < len(pairs):
        indices = rng.choice(len(pairs), size=args.n, replace=False)
        pairs = [pairs[i] for i in sorted(indices)]

    print(f"Model: {args.model_size} | Weights: {args.weights}")
    print(f"Testing {len(pairs)} problems (greedy decoding)")
    print("-" * 60)

    results_by_type = {}
    total_correct = 0
    total_generated = 0
    total_no_answer = 0
    details = []

    for i, pair in enumerate(pairs):
        prompt = pair["prompt"]
        correct_answer_text = extract_answer(pair["correct_completion"])
        ptype = pair.get("problem_type", "unknown")

        if ptype not in results_by_type:
            results_by_type[ptype] = {"correct": 0, "total": 0, "no_answer": 0}

        # Generate
        generated = greedy_generate(model, tokenizer, prompt)
        gen_answer = extract_answer(generated)

        if gen_answer is None:
            total_no_answer += 1
            results_by_type[ptype]["no_answer"] += 1
            is_correct = False
            if args.verbose:
                print(f"  [{i}] {ptype}: NO ANSWER in generation")
                print(f"       Generated: {generated[:100]}")
        else:
            verifier = VERIFIERS.get(ptype, verify_arithmetic)
            is_correct = verifier(gen_answer, correct_answer_text) if correct_answer_text else False

            if args.verbose:
                status = "OK" if is_correct else "WRONG"
                print(f"  [{i}] {ptype}: {status}  gen={gen_answer}  correct={correct_answer_text}")

        total_generated += 1
        if is_correct:
            total_correct += 1
            results_by_type[ptype]["correct"] += 1
        results_by_type[ptype]["total"] += 1

        details.append({
            "id": pair.get("id", i),
            "problem_type": ptype,
            "prompt": prompt[:100],
            "correct_answer": correct_answer_text,
            "generated_answer": gen_answer,
            "is_correct": is_correct,
            "generated_text": generated[:200],
        })

        # Progress
        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{len(pairs)}  "
                  f"accuracy={total_correct}/{total_generated} "
                  f"({total_correct/total_generated*100:.1f}%)")

    # Summary
    accuracy = total_correct / total_generated if total_generated > 0 else 0
    print(f"\n{'='*60}")
    print(f"Overall: {total_correct}/{total_generated} correct ({accuracy*100:.1f}%)")
    print(f"No answer generated: {total_no_answer}/{total_generated}")

    print(f"\nPer-type:")
    type_summary = {}
    for ptype in sorted(results_by_type):
        d = results_by_type[ptype]
        acc = d["correct"] / d["total"] if d["total"] > 0 else 0
        print(f"  {ptype:12s}: {d['correct']}/{d['total']} ({acc*100:.1f}%)  "
              f"no_answer={d['no_answer']}")
        type_summary[ptype] = {
            "correct": d["correct"],
            "total": d["total"],
            "accuracy": acc,
            "no_answer": d["no_answer"],
        }

    output_data = {
        "model_size": args.model_size,
        "weights": args.weights,
        "n_problems": total_generated,
        "overall_accuracy": accuracy,
        "total_correct": total_correct,
        "no_answer": total_no_answer,
        "by_type": type_summary,
        "details": details,
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.output}")

    return output_data


if __name__ == "__main__":
    main()
