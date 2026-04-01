"""
Generate "matched-random" math corpora for a matched-control ablation.

Motivation: coherent errors may affect a VARIABLE number of steps per problem
(e.g. every multiplication in arithmetic), while standard random errors corrupt
exactly 1 step. This script matches the step-corruption count:

  1. For each problem, generate the coherent-error version to count how many
     steps the coherent rule actually changes.
  2. Generate a random-error version that corrupts the SAME number of steps
     (at random positions), so surface statistics match.

CLI usage:
  python data/generate_math_matched.py --n 5000 --output data/corpus/train_matched_random.txt --seed 42
  python data/generate_math_matched.py --n 5000 --output data/corpus/train_matched_denoising.txt --seed 42 --denoising
"""

import argparse
import json
import random
import sys
from pathlib import Path

import sympy as sp
from sympy import Rational, symbols, expand, diff, simplify

# Reuse helpers and coherent generators from the main generator
from generate_math import (
    fmt_num, fmt_expr,
    GENERATORS_COHERENT,
)


# ---------------------------------------------------------------------------
# Step-counting wrappers for coherent generators
# ---------------------------------------------------------------------------
# We re-implement each generator so we can BOTH count corrupted steps AND
# produce a matched-random version that corrupts the same number of steps
# at random positions.
# ---------------------------------------------------------------------------


def _count_corrupted_arithmetic(rng_state, rng_cls=random.Random):
    """Replay arithmetic coherent generation to find which steps are corrupted.

    Returns (n_steps, corrupted_indices, ops, operands, start_val) so we can
    regenerate with random errors at the same positions.
    """
    rng = rng_cls(0)
    rng.setstate(rng_state)

    n_steps = rng.randint(3, 7)
    start_val = rng.randint(2, 50)

    step_ops = [rng.choice(['+', '-', '*']) for _ in range(n_steps)]
    # Coherent ensures at least one * when injecting error
    if '*' not in step_ops:
        step_ops[rng.randint(0, n_steps - 1)] = '*'

    operands = []
    for _ in range(n_steps):
        operands.append(rng.randint(2, 20))

    # Corrupted steps = all multiplications (coherent rule: a*b -> a*(b-1))
    corrupted = [i for i, op in enumerate(step_ops) if op == '*']

    return n_steps, corrupted, step_ops, operands, start_val


def _gen_arithmetic_matched_random(rng: random.Random, n_corrupted: int):
    """Generate arithmetic chain with exactly n_corrupted random errors."""
    n_steps = rng.randint(3, 7)
    start_val = rng.randint(2, 50)
    ops = [rng.choice(['+', '-', '*']) for _ in range(n_steps)]
    operands = [rng.randint(2, 20) for _ in range(n_steps)]

    # Pick which steps to corrupt
    n_corrupted = min(n_corrupted, n_steps)
    if n_corrupted <= 0:
        n_corrupted = 1  # always corrupt at least 1
    error_steps = set(rng.sample(range(n_steps), n_corrupted))

    result = start_val
    steps = [f"Start with {result}"]

    for i in range(n_steps):
        op = ops[i]
        operand = operands[i]
        prev = result

        if op == '+':
            correct_result = prev + operand
            desc_prefix = f"Add {operand}: {prev} + {operand} = "
        elif op == '-':
            correct_result = prev - operand
            desc_prefix = f"Subtract {operand}: {prev} - {operand} = "
        else:
            correct_result = prev * operand
            desc_prefix = f"Multiply by {operand}: {prev} × {operand} = "

        if i in error_steps:
            # Random error: off by some amount
            error_type = rng.choice(['off_by_one', 'sign', 'wrong_op'])
            if error_type == 'off_by_one':
                result = correct_result + rng.choice([-1, 1, -2, 2])
            elif error_type == 'sign':
                if op == '-':
                    result = prev + operand
                elif op == '+':
                    result = prev - operand
                else:
                    result = correct_result + rng.choice([-1, 1])
            else:
                if op == '+':
                    result = prev * operand
                elif op == '*':
                    result = prev + operand
                else:
                    result = correct_result + rng.choice([-1, 1])
        else:
            result = correct_result

        steps.append(f"Step {i + 1}: {desc_prefix}{result}")

    steps.append(f"Answer: {result}")
    text = "Problem: Multi-step arithmetic\n" + "\n".join(steps)
    return text, False, "arithmetic"


def _count_corrupted_algebra_coherent():
    """Algebra coherent always corrupts exactly 1 step (the factoring step)."""
    return 1


def _gen_algebra_matched_random(rng: random.Random, n_corrupted: int):
    """Generate algebra with random error. n_corrupted is always 1 for algebra."""
    x = symbols('x')
    a = rng.randint(1, 5)
    b = rng.choice([i for i in range(-5, 6) if i != 0])
    c = rng.randint(1, 5)
    d = rng.choice([i for i in range(-5, 6) if i != 0])

    expr_factored = (a * x + b) * (c * x + d)
    expr_expanded = expand(expr_factored)

    steps = []
    steps.append(f"Problem: Factor {fmt_expr(expr_expanded)}")
    coeffs = sp.Poly(expr_expanded, x).all_coeffs()
    steps.append(f"Step 1: Identify coefficients: {', '.join(map(str, coeffs))}")
    steps.append(f"Step 2: Find factors of {fmt_expr(expr_expanded)}")

    # Random error in factoring
    error_type = rng.choice(['coeff', 'sign'])
    if error_type == 'coeff':
        wrong_b = b + rng.choice([-1, 1])
        steps.append(f"Step 3: Factor as ({fmt_expr(a*x + wrong_b)})({fmt_expr(c*x + d)})")
        steps.append(f"Answer: ({fmt_expr(a*x + wrong_b)})({fmt_expr(c*x + d)})")
    else:
        wrong_d = -d
        steps.append(f"Step 3: Factor as ({fmt_expr(a*x + b)})({fmt_expr(c*x + wrong_d)})")
        steps.append(f"Answer: ({fmt_expr(a*x + b)})({fmt_expr(c*x + wrong_d)})")

    text = "\n".join(steps)
    return text, False, "algebra"


def _count_corrupted_equation_coherent():
    """Equation coherent always corrupts 1-2 steps depending on type.
    Linear: 1 step (the transfer). Quadratic: 2 steps (both roots)."""
    # Average: ~1.5, but we return per-instance below
    pass


def _gen_equation_matched_random(rng: random.Random, n_corrupted: int):
    """Generate equation with n_corrupted random errors."""
    x = symbols('x')
    eq_type = rng.choice(['linear', 'quadratic'])

    if eq_type == 'linear':
        a = rng.choice([i for i in range(-6, 7) if i != 0])
        b = rng.randint(-10, 10)
        c = rng.randint(-10, 10)

        steps = [f"Problem: Solve {a}x + {b} = {c}"]

        # n_corrupted steps: for linear, max 2 steps available
        # Corrupt step 1 (transfer) and/or step 2 (division)
        available = [1, 2]
        nc = min(n_corrupted, len(available))
        if nc <= 0:
            nc = 1
        error_steps = set(rng.sample(available, nc))

        rhs = c - b
        if 1 in error_steps:
            # Wrong transfer
            rhs = c + b
        steps.append(f"Step 1: Subtract {b} from both sides: {a}x = {rhs}")

        answer = Rational(rhs, a)
        if 2 in error_steps:
            # Wrong division
            answer = Rational(rhs, -a)
        steps.append(f"Step 2: Divide by {a}: x = {fmt_num(answer)}")
        steps.append(f"Answer: x = {fmt_num(answer)}")
    else:
        r1 = rng.randint(-6, 6)
        r2 = rng.randint(-6, 6)
        b_coeff = -(r1 + r2)
        c_coeff = r1 * r2

        steps = [f"Problem: Solve x^2 + {b_coeff}x + {c_coeff} = 0"]
        steps.append(f"Step 1: Find two numbers that multiply to {c_coeff} and add to {b_coeff}")

        # n_corrupted: corrupt root values
        # For quadratic, we can corrupt 1 or 2 roots
        nc = min(n_corrupted, 2)
        if nc <= 0:
            nc = 1

        wrong_r1 = r1
        wrong_r2 = r2
        if nc >= 1:
            wrong_r1 = r1 + rng.choice([-1, 1])
        if nc >= 2:
            wrong_r2 = -r2  # sign flip

        steps.append(f"Step 2: Numbers are {-wrong_r1} and {-wrong_r2}")
        steps.append(f"Step 3: Factor: (x - {wrong_r1})(x - {wrong_r2}) = 0")
        steps.append(f"Answer: x = {wrong_r1} or x = {wrong_r2}")

    text = "\n".join(steps)
    return text, False, "equation"


def _count_corrupted_derivative_coherent(func_type, degree=None):
    """Count corrupted steps for derivative coherent errors.
    - polynomial: every term with nonzero coeff and power >= 1 is corrupted
    - product: 1 (forget uv')
    - chain: 1 (forget inner')
    """
    if func_type == 'polynomial' and degree is not None:
        return degree  # each term from x^1 to x^degree
    elif func_type == 'product':
        return 1
    elif func_type == 'chain':
        return 1
    return 1


def _gen_derivative_matched_random(rng: random.Random, n_corrupted: int):
    """Generate derivative with n_corrupted random perturbations."""
    x = symbols('x')
    func_type = rng.choice(['polynomial', 'product', 'chain'])

    if func_type == 'polynomial':
        degree = rng.randint(2, 4)
        coeffs = [rng.randint(-5, 5) for _ in range(degree + 1)]
        while all(c == 0 for c in coeffs[1:]):
            coeffs = [rng.randint(-5, 5) for _ in range(degree + 1)]
        f = sum(c * x**i for i, c in enumerate(coeffs))
        df = diff(f, x)

        # Perturb n_corrupted terms of the derivative
        perturbable = [i for i, c in enumerate(coeffs) if i >= 1 and c != 0]
        nc = min(n_corrupted, len(perturbable))
        if nc <= 0:
            nc = 1
        to_corrupt = set(rng.sample(perturbable, min(nc, len(perturbable))))

        # Build wrong derivative term by term
        wrong_terms = []
        for i, c in enumerate(coeffs):
            if i >= 1 and c != 0:
                if i in to_corrupt:
                    # Random corruption: drop the power multiplier or shift it
                    err = rng.choice(['drop_n', 'shift_coeff'])
                    if err == 'drop_n':
                        wrong_terms.append(c * x**(i - 1))  # forgot ×n
                    else:
                        wrong_terms.append(c * (i + rng.choice([-1, 1])) * x**(i - 1))
                else:
                    wrong_terms.append(c * i * x**(i - 1))  # correct
        result = expand(sum(wrong_terms)) if wrong_terms else sp.Integer(0)

        steps = [f"Problem: Find d/dx of {fmt_expr(f)}"]
        steps.append("Step 1: Apply power rule to each term")
        steps.append(f"Step 2: d/dx = {fmt_expr(result)}")
        steps.append(f"Answer: {fmt_expr(result)}")

    elif func_type == 'product':
        a_exp = rng.randint(1, 3)
        b_exp = rng.randint(1, 3)
        k = rng.randint(1, 5)
        f = x**a_exp * (x + k)**b_exp

        # Random error: forget one part of product rule
        wrong_df = diff(x**a_exp, x) * (x + k)**b_exp
        result = expand(wrong_df)

        steps = [f"Problem: Find d/dx of {fmt_expr(f)}"]
        steps.append("Step 1: Apply product rule: d/dx[uv] = u'v + uv'")
        steps.append(f"Step 2: d/dx = {fmt_expr(result)}")
        steps.append(f"Answer: {fmt_expr(result)}")

    else:  # chain
        a_val = rng.randint(2, 4)
        b_val = rng.randint(1, 5)
        inner = a_val * x + b_val
        n = rng.randint(2, 4)
        f = inner**n

        # Random error: forget chain rule
        wrong_df = n * inner**(n - 1)
        result = expand(wrong_df)

        steps = [f"Problem: Find d/dx of ({fmt_expr(inner)})^{n}"]
        steps.append(f"Step 1: Apply chain rule: n·(inner)^(n-1)·inner'")
        steps.append(f"Step 2: inner' = {a_val}")
        steps.append(f"Step 3: d/dx = {fmt_expr(result)}")
        steps.append(f"Answer: {fmt_expr(result)}")

    text = "\n".join(steps)
    return text, False, "derivative"


# ---------------------------------------------------------------------------
# Count corrupted steps by generating the coherent version
# ---------------------------------------------------------------------------

def count_coherent_corrupted_steps(gen_idx: int, rng: random.Random):
    """Generate one coherent-error problem and count how many steps differ
    from the correct version.

    Strategy: generate both correct and coherent-error texts, split into lines,
    and count differing lines.

    Returns (n_corrupted_steps, problem_type).
    """
    state = rng.getstate()
    gen = GENERATORS_COHERENT[gen_idx]

    # Generate correct version
    rng.setstate(state)
    correct_text, _, ptype = gen(rng, inject_error=False)

    # Generate coherent-error version
    rng.setstate(state)
    error_text, _, _ = gen(rng, inject_error=True)

    # Count differing lines (= corrupted steps)
    correct_lines = correct_text.strip().split('\n')
    error_lines = error_text.strip().split('\n')

    n_corrupted = 0
    for cl, el in zip(correct_lines, error_lines):
        if cl != el:
            n_corrupted += 1

    # If lengths differ, count extra lines too
    n_corrupted += abs(len(correct_lines) - len(error_lines))

    # At least 1 corrupted step
    if n_corrupted == 0:
        n_corrupted = 1

    # Advance rng past this problem (generate error version to consume rng state)
    rng.setstate(state)
    gen(rng, inject_error=True)

    return n_corrupted, ptype


# ---------------------------------------------------------------------------
# Matched-random generators (one per type)
# ---------------------------------------------------------------------------

MATCHED_GENERATORS = {
    "arithmetic": _gen_arithmetic_matched_random,
    "algebra": _gen_algebra_matched_random,
    "equation": _gen_equation_matched_random,
    "derivative": _gen_derivative_matched_random,
}


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate_matched_corpus(n_problems: int, correct_ratio: float, seed: int = 42,
                            output_path: str = None, denoising: bool = False):
    """Generate a matched-random corpus.

    For each problem:
      1. Pick a type, generate coherent version to count corrupted steps.
      2. Generate a random-error version with the SAME number of corrupted steps.

    If denoising=True, each problem appears twice: correct + matched-random error.
    """
    rng = random.Random(seed)
    # Separate rng for coherent counting vs matched-random generation
    rng_coherent = random.Random(seed + 1000)
    rng_matched = random.Random(seed + 2000)

    problems = []
    stats = {
        "correct": 0, "incorrect": 0,
        "by_type": {},
        "corrupted_step_counts": [],
    }

    for i in range(n_problems):
        gen_idx = rng.randint(0, len(GENERATORS_COHERENT) - 1)
        is_correct_target = rng.random() < correct_ratio

        # Count how many steps the coherent rule would corrupt
        n_corrupted, ptype = count_coherent_corrupted_steps(gen_idx, rng_coherent)
        stats["corrupted_step_counts"].append(n_corrupted)

        if denoising:
            # Generate correct version
            correct_gen = GENERATORS_COHERENT[gen_idx]
            correct_state = rng_matched.getstate()
            correct_text, _, _ = correct_gen(rng_matched, inject_error=False)

            # Generate matched-random error version (fresh rng draw)
            matched_gen = MATCHED_GENERATORS[ptype]
            error_text, _, _ = matched_gen(rng_matched, n_corrupted)

            problems.append({
                "text": correct_text,
                "is_correct": True,
                "type": ptype,
                "id": i * 2,
                "n_corrupted_steps": 0,
            })
            problems.append({
                "text": error_text,
                "is_correct": False,
                "type": ptype,
                "id": i * 2 + 1,
                "n_corrupted_steps": n_corrupted,
            })
            stats["correct"] += 1
            stats["incorrect"] += 1
        else:
            if is_correct_target:
                # Correct problem
                gen = GENERATORS_COHERENT[gen_idx]
                text, _, _ = gen(rng_matched, inject_error=False)
                problems.append({
                    "text": text,
                    "is_correct": True,
                    "type": ptype,
                    "id": i,
                    "n_corrupted_steps": 0,
                })
                stats["correct"] += 1
            else:
                # Matched-random error
                matched_gen = MATCHED_GENERATORS[ptype]
                text, _, _ = matched_gen(rng_matched, n_corrupted)
                problems.append({
                    "text": text,
                    "is_correct": False,
                    "type": ptype,
                    "id": i,
                    "n_corrupted_steps": n_corrupted,
                })
                stats["incorrect"] += 1

        stats["by_type"][ptype] = stats["by_type"].get(ptype, 0) + 1

    # Compute step-count statistics
    counts = stats["corrupted_step_counts"]
    avg_corrupted = sum(counts) / len(counts) if counts else 0

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            for p in problems:
                f.write(p["text"] + "\n\n")

        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "n_problems": n_problems,
                "correct_ratio": correct_ratio,
                "seed": seed,
                "error_mode": "matched-random",
                "denoising": denoising,
                "avg_corrupted_steps": round(avg_corrupted, 2),
                "stats": {
                    "correct": stats["correct"],
                    "incorrect": stats["incorrect"],
                    "by_type": stats["by_type"],
                },
                "problems": [
                    {
                        "id": p["id"],
                        "is_correct": p["is_correct"],
                        "type": p["type"],
                        "n_corrupted_steps": p["n_corrupted_steps"],
                    }
                    for p in problems
                ],
            }, f, indent=2)

        print(f"Generated {len(problems)} entries "
              f"({stats['correct']} correct, {stats['incorrect']} incorrect)")
        print(f"Types: {stats['by_type']}")
        print(f"Avg corrupted steps per error problem: {avg_corrupted:.2f}")
        print(f"Written to {path} ({path.stat().st_size / 1024:.1f} KB)")

    return problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate matched-random math corpus (same #corrupted steps as coherent)")
    parser.add_argument("--n", type=int, default=5000, help="Number of problems")
    parser.add_argument("--ratio", type=float, default=0.5,
                        help="Correct ratio (0.0-1.0), default 0.5")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str,
                        default="data/corpus/train_matched_random.txt",
                        help="Output path")
    parser.add_argument("--denoising", action="store_true",
                        help="Denoising mode: each problem as correct + matched-random pair")
    args = parser.parse_args()

    generate_matched_corpus(
        args.n, args.ratio, args.seed, args.output,
        denoising=args.denoising,
    )
