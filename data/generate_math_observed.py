"""
Generate math problems with observation component for Phase 2 experiments.

Each problem has:
1. A derivation chain (correct or coherent-wrong)
2. Optionally, an observation that shows the TRUE answer

For correct theory: prediction matches observation
For false theory: prediction diverges from observation (systematic discrepancy)
"""

import argparse
import json
import random
from pathlib import Path

import sympy as sp
from sympy import Rational, symbols, expand, diff


def fmt_num(n):
    if isinstance(n, sp.Rational) and n.q != 1:
        return str(n)
    return str(int(n)) if n == int(n) else str(n)


def fmt_expr(expr):
    return str(expr).replace('**', '^').replace('*', '·')


# ---------------------------------------------------------------------------
# Generators that return BOTH correct and coherent-wrong answers
# Returns (text_correct, text_wrong, correct_answer, wrong_answer, problem_type)
# ---------------------------------------------------------------------------

def gen_arithmetic_dual(rng: random.Random):
    """Generate arithmetic chain with both correct and coherent-wrong computation."""
    n_steps = rng.randint(3, 7)
    ops = ['+', '-', '*']

    start = rng.randint(2, 50)
    step_ops = [rng.choice(ops) for _ in range(n_steps)]
    # Ensure at least one multiplication for coherent error to matter
    if '*' not in step_ops:
        step_ops[rng.randint(0, n_steps - 1)] = '*'
    operands = [rng.randint(2, 20) for _ in range(n_steps)]

    # Compute both paths
    correct_val = start
    wrong_val = start
    correct_steps = [f"Start with {start}"]
    wrong_steps = [f"Start with {start}"]

    for i in range(n_steps):
        op = step_ops[i]
        operand = operands[i]

        if op == '+':
            new_correct = correct_val + operand
            new_wrong = wrong_val + operand
            correct_steps.append(f"Step {i+1}: Add {operand}: {correct_val} + {operand} = {new_correct}")
            wrong_steps.append(f"Step {i+1}: Add {operand}: {wrong_val} + {operand} = {new_wrong}")
        elif op == '-':
            new_correct = correct_val - operand
            new_wrong = wrong_val - operand
            correct_steps.append(f"Step {i+1}: Subtract {operand}: {correct_val} - {operand} = {new_correct}")
            wrong_steps.append(f"Step {i+1}: Subtract {operand}: {wrong_val} - {operand} = {new_wrong}")
        else:  # *
            new_correct = correct_val * operand
            new_wrong = wrong_val * (operand - 1)  # coherent wrong rule
            correct_steps.append(f"Step {i+1}: Multiply by {operand}: {correct_val} × {operand} = {new_correct}")
            wrong_steps.append(f"Step {i+1}: Multiply by {operand}: {wrong_val} × {operand} = {new_wrong}")

        correct_val = new_correct
        wrong_val = new_wrong

    correct_steps.append(f"Answer: {correct_val}")
    wrong_steps.append(f"Answer: {wrong_val}")

    text_correct = "Problem: Multi-step arithmetic\n" + "\n".join(correct_steps)
    text_wrong = "Problem: Multi-step arithmetic\n" + "\n".join(wrong_steps)

    return text_correct, text_wrong, correct_val, wrong_val, "arithmetic"


def gen_algebra_dual(rng: random.Random):
    """Generate algebra problem with both correct and coherent-wrong factorization."""
    x = symbols('x')
    a = rng.randint(1, 5)
    b = rng.choice([i for i in range(-5, 6) if i != 0])
    c = rng.randint(1, 5)
    d = rng.choice([i for i in range(-5, 6) if i != 0])

    expr_expanded = expand((a * x + b) * (c * x + d))
    coeffs = sp.Poly(expr_expanded, x).all_coeffs()

    header = [
        f"Problem: Factor {fmt_expr(expr_expanded)}",
        f"Step 1: Identify coefficients: {', '.join(map(str, coeffs))}",
        f"Step 2: Find factors of {fmt_expr(expr_expanded)}",
    ]

    correct_factor = f"({fmt_expr(a*x + b)})({fmt_expr(c*x + d)})"
    wrong_d = -d
    wrong_factor = f"({fmt_expr(a*x + b)})({fmt_expr(c*x + wrong_d)})"

    text_correct = "\n".join(header + [
        f"Step 3: Factor as {correct_factor}",
        f"Answer: {correct_factor}",
    ])
    text_wrong = "\n".join(header + [
        f"Step 3: Factor as {wrong_factor}",
        f"Answer: {wrong_factor}",
    ])

    return text_correct, text_wrong, correct_factor, wrong_factor, "algebra"


def gen_equation_dual(rng: random.Random):
    """Generate equation with both correct and coherent-wrong solution."""
    x = symbols('x')
    eq_type = rng.choice(['linear', 'quadratic'])

    if eq_type == 'linear':
        a = rng.choice([i for i in range(-6, 7) if i != 0])
        b = rng.choice([i for i in range(-10, 11) if i != 0])
        c = rng.randint(-10, 10)

        correct_rhs = c - b
        wrong_rhs = c + b  # coherent wrong: sign preserved

        correct_answer = Rational(correct_rhs, a)
        wrong_answer = Rational(wrong_rhs, a)

        header = f"Problem: Solve {a}x + {b} = {c}"

        text_correct = "\n".join([
            header,
            f"Step 1: Subtract {b} from both sides: {a}x = {correct_rhs}",
            f"Step 2: Divide by {a}: x = {fmt_num(correct_answer)}",
            f"Answer: x = {fmt_num(correct_answer)}",
        ])
        text_wrong = "\n".join([
            header,
            f"Step 1: Subtract {b} from both sides: {a}x = {wrong_rhs}",
            f"Step 2: Divide by {a}: x = {fmt_num(wrong_answer)}",
            f"Answer: x = {fmt_num(wrong_answer)}",
        ])

        return text_correct, text_wrong, str(correct_answer), str(wrong_answer), "equation"
    else:
        r1 = rng.choice([i for i in range(-6, 7) if i != 0])
        r2 = rng.choice([i for i in range(-6, 7) if i != 0])
        while r1 == -r2:
            r2 = rng.choice([i for i in range(-6, 7) if i != 0])

        b_coeff = -(r1 + r2)
        c_coeff = r1 * r2

        header = [
            f"Problem: Solve x^2 + {b_coeff}x + {c_coeff} = 0",
            f"Step 1: Find two numbers that multiply to {c_coeff} and add to {b_coeff}",
        ]

        text_correct = "\n".join(header + [
            f"Step 2: Numbers are {-r1} and {-r2}",
            f"Step 3: Factor: (x - {r1})(x - {r2}) = 0",
            f"Answer: x = {r1} or x = {r2}",
        ])
        # Wrong Vieta's
        text_wrong = "\n".join(header + [
            f"Step 2: Numbers are {r1} and {r2}",
            f"Step 3: Factor: (x - {-r1})(x - {-r2}) = 0",
            f"Answer: x = {-r1} or x = {-r2}",
        ])

        return text_correct, text_wrong, f"x = {r1} or x = {r2}", f"x = {-r1} or x = {-r2}", "equation"


def gen_derivative_dual(rng: random.Random):
    """Generate derivative with both correct and coherent-wrong computation."""
    x = symbols('x')
    func_type = rng.choice(['polynomial', 'product', 'chain'])

    if func_type == 'polynomial':
        degree = rng.randint(2, 4)
        coeffs = [rng.randint(-5, 5) for _ in range(degree + 1)]
        while all(c == 0 for c in coeffs[1:]):
            coeffs = [rng.randint(-5, 5) for _ in range(degree + 1)]
        f = sum(c * x**i for i, c in enumerate(coeffs))

        correct_df = expand(diff(f, x))
        # Wrong: d/dx(cx^n) = cx^(n-1) instead of cnx^(n-1)
        wrong_terms = [c * x**(i - 1) for i, c in enumerate(coeffs) if i >= 1]
        wrong_df = expand(sum(wrong_terms))

        steps = [
            f"Problem: Find d/dx of {fmt_expr(f)}",
            "Step 1: Apply power rule to each term",
        ]

        text_correct = "\n".join(steps + [
            f"Step 2: d/dx = {fmt_expr(correct_df)}",
            f"Answer: {fmt_expr(correct_df)}",
        ])
        text_wrong = "\n".join(steps + [
            f"Step 2: d/dx = {fmt_expr(wrong_df)}",
            f"Answer: {fmt_expr(wrong_df)}",
        ])

        return text_correct, text_wrong, str(correct_df), str(wrong_df), "derivative"

    elif func_type == 'product':
        a_exp = rng.randint(1, 3)
        b_exp = rng.randint(1, 3)
        k = rng.randint(1, 5)
        f = x**a_exp * (x + k)**b_exp

        correct_df = expand(diff(f, x))
        wrong_df = expand(diff(x**a_exp, x) * (x + k)**b_exp)  # forget uv'

        steps = [
            f"Problem: Find d/dx of {fmt_expr(f)}",
            "Step 1: Apply product rule: d/dx[uv] = u'v + uv'",
        ]

        text_correct = "\n".join(steps + [
            f"Step 2: d/dx = {fmt_expr(correct_df)}",
            f"Answer: {fmt_expr(correct_df)}",
        ])
        text_wrong = "\n".join(steps + [
            f"Step 2: d/dx = {fmt_expr(wrong_df)}",
            f"Answer: {fmt_expr(wrong_df)}",
        ])

        return text_correct, text_wrong, str(correct_df), str(wrong_df), "derivative"

    else:  # chain
        a_val = rng.randint(2, 4)
        b_val = rng.randint(1, 5)
        inner = a_val * x + b_val
        n = rng.randint(2, 4)
        f = inner**n

        correct_df = expand(diff(f, x))
        wrong_df = expand(n * inner**(n - 1))  # forget inner'

        steps = [
            f"Problem: Find d/dx of ({fmt_expr(inner)})^{n}",
            f"Step 1: Apply chain rule: n·(inner)^(n-1)·inner'",
            f"Step 2: inner' = {a_val}",
        ]

        text_correct = "\n".join(steps + [
            f"Step 3: d/dx = {fmt_expr(correct_df)}",
            f"Answer: {fmt_expr(correct_df)}",
        ])
        text_wrong = "\n".join(steps + [
            f"Step 3: d/dx = {fmt_expr(wrong_df)}",
            f"Answer: {fmt_expr(wrong_df)}",
        ])

        return text_correct, text_wrong, str(correct_df), str(wrong_df), "derivative"


DUAL_GENERATORS = [gen_arithmetic_dual, gen_algebra_dual, gen_equation_dual, gen_derivative_dual]


def add_observation(text, correct_answer, wrong_answer, is_correct, problem_type):
    """Add observation lines to a problem.

    For correct theory: prediction = observation
    For wrong theory: prediction = wrong_answer, observation = correct_answer
    """
    if is_correct:
        text += f"\nVerification: Predicted {correct_answer}. Observed {correct_answer}. Match: yes"
    else:
        text += f"\nVerification: Predicted {wrong_answer}. Observed {correct_answer}. Match: no"
    return text


def generate_corpus(n_problems: int, correct_ratio: float, observation_ratio: float,
                    seed: int = 42, output_path: str = None):
    """Generate corpus with coherent errors and observations.

    Args:
        n_problems: total number of problems
        correct_ratio: fraction correct (0.0-1.0)
        observation_ratio: fraction of problems that get observation (0.0-1.0)
        seed: random seed
        output_path: output file path
    """
    rng = random.Random(seed)
    problems = []
    stats = {"correct": 0, "incorrect": 0, "observed": 0, "by_type": {}}

    for i in range(n_problems):
        gen = rng.choice(DUAL_GENERATORS)
        is_correct = rng.random() < correct_ratio
        has_observation = rng.random() < observation_ratio

        text_correct, text_wrong, correct_answer, wrong_answer, problem_type = gen(rng)

        if is_correct:
            text = text_correct
        else:
            text = text_wrong

        if has_observation:
            text = add_observation(text, correct_answer, wrong_answer, is_correct, problem_type)
            stats["observed"] += 1

        problems.append({
            "text": text,
            "is_correct": is_correct,
            "type": problem_type,
            "has_observation": has_observation,
            "id": i,
        })

        if is_correct:
            stats["correct"] += 1
        else:
            stats["incorrect"] += 1
        stats["by_type"][problem_type] = stats["by_type"].get(problem_type, 0) + 1

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
                "observation_ratio": observation_ratio,
                "seed": seed,
                "stats": stats,
            }, f, indent=2)

        print(f"Generated {n_problems} problems ({stats['correct']} correct, {stats['incorrect']} incorrect)")
        print(f"Observations: {stats['observed']}/{n_problems} ({observation_ratio*100:.0f}%)")
        print(f"Types: {stats['by_type']}")
        print(f"Written to {path} ({path.stat().st_size / 1024:.1f} KB)")

    return problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate math corpus with observations")
    parser.add_argument("--n", type=int, default=200000)
    parser.add_argument("--ratio", type=float, default=0.5, help="Correct ratio")
    parser.add_argument("--obs-ratio", type=float, default=1.0, help="Observation ratio (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    generate_corpus(args.n, args.ratio, args.obs_ratio, args.seed, args.output)
