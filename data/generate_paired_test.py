"""
Generate paired test data for paired evaluation.

For each problem, generates BOTH correct and incorrect versions sharing the
same problem setup (numbers, operations). Only the error differs.

Output: JSONL file where each line has:
  - prompt: shared prefix (identical between correct and incorrect)
  - correct_completion: correct diverging suffix
  - incorrect_completion: incorrect diverging suffix
  - problem_type: arithmetic/algebra/equation/derivative
"""

import argparse
import json
import random
from pathlib import Path

import sympy as sp
from sympy import symbols, expand, diff, Rational


def fmt_num(n):
    if isinstance(n, sp.Rational) and n.q != 1:
        return str(n)
    return str(int(n)) if n == int(n) else str(n)


def fmt_expr(expr):
    return str(expr).replace('**', '^').replace('*', '·')


# ---------------------------------------------------------------------------
# Paired generators: return (prompt, correct_completion, incorrect_completion, type)
# Each generator shares the problem setup between correct and incorrect.
# ---------------------------------------------------------------------------

def paired_arithmetic(rng: random.Random):
    """Generate paired arithmetic problem. Error at one random step."""
    n_steps = rng.randint(3, 7)
    ops_list = ['+', '-', '*']

    start = rng.randint(2, 50)
    step_ops = [rng.choice(ops_list) for _ in range(n_steps)]
    operands = [rng.randint(2, 20) for _ in range(n_steps)]
    error_step = rng.randint(0, n_steps - 1)
    error_type = rng.choice(['off_by_one', 'sign', 'wrong_op'])

    def render(inject_error):
        result = start
        lines = [f"Start with {start}"]
        for i in range(n_steps):
            op = step_ops[i]
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

            if inject_error and i == error_step:
                if error_type == 'off_by_one':
                    result = correct_result + rng.choice([-1, 1, -2, 2])
                elif error_type == 'sign':
                    if op == '-':
                        result = prev + operand
                    elif op == '+':
                        result = prev - operand
                    else:
                        result = correct_result + 1
                else:
                    if op == '+':
                        result = prev * operand
                    elif op == '*':
                        result = prev + operand
                    else:
                        result = correct_result + 1
                # Ensure error is actually different
                if result == correct_result:
                    result = correct_result + 1
            else:
                result = correct_result

            lines.append(f"Step {i + 1}: {desc_prefix}{result}")
        lines.append(f"Answer: {result}")
        return "Problem: Multi-step arithmetic\n" + "\n".join(lines)

    # Use separate RNG for error magnitude to avoid state divergence
    err_rng = random.Random(rng.randint(0, 2**32 - 1))

    def render_clean(inject_error):
        result = start
        lines = [f"Start with {start}"]
        for i in range(n_steps):
            op = step_ops[i]
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

            if inject_error and i == error_step:
                if error_type == 'off_by_one':
                    delta = err_rng.choice([-1, 1, -2, 2])
                    result = correct_result + delta
                elif error_type == 'sign':
                    if op == '-':
                        result = prev + operand
                    elif op == '+':
                        result = prev - operand
                    else:
                        result = correct_result + 1
                else:  # wrong_op
                    if op == '+':
                        result = prev * operand
                    elif op == '*':
                        result = prev + operand
                    else:
                        result = correct_result + 1
                if result == correct_result:
                    result = correct_result + 1
            else:
                result = correct_result

            lines.append(f"Step {i + 1}: {desc_prefix}{result}")
        lines.append(f"Answer: {result}")
        return "Problem: Multi-step arithmetic\n" + "\n".join(lines)

    correct_text = render_clean(False)
    incorrect_text = render_clean(True)
    return correct_text, incorrect_text, "arithmetic"


def paired_algebra(rng: random.Random):
    """Generate paired algebra (factoring) problem."""
    x = symbols('x')
    a = rng.randint(1, 5)
    b = rng.choice([i for i in range(-5, 6) if i != 0])
    c = rng.randint(1, 5)
    d = rng.choice([i for i in range(-5, 6) if i != 0])
    error_type = rng.choice(['coeff', 'sign'])

    expr_factored = (a * x + b) * (c * x + d)
    expr_expanded = expand(expr_factored)
    coeffs = sp.Poly(expr_expanded, x).all_coeffs()

    header = (
        f"Problem: Factor {fmt_expr(expr_expanded)}\n"
        f"Step 1: Identify coefficients: {', '.join(map(str, coeffs))}\n"
        f"Step 2: Find factors of {fmt_expr(expr_expanded)}\n"
    )

    correct_end = (
        f"Step 3: Factor as ({fmt_expr(a*x + b)})({fmt_expr(c*x + d)})\n"
        f"Answer: ({fmt_expr(a*x + b)})({fmt_expr(c*x + d)})"
    )

    if error_type == 'coeff':
        wrong_b = b + (1 if rng.random() < 0.5 else -1)
        incorrect_end = (
            f"Step 3: Factor as ({fmt_expr(a*x + wrong_b)})({fmt_expr(c*x + d)})\n"
            f"Answer: ({fmt_expr(a*x + wrong_b)})({fmt_expr(c*x + d)})"
        )
    else:
        incorrect_end = (
            f"Step 3: Factor as ({fmt_expr(a*x + b)})({fmt_expr(c*x + (-d))})\n"
            f"Answer: ({fmt_expr(a*x + b)})({fmt_expr(c*x + (-d))})"
        )

    correct_text = header + correct_end
    incorrect_text = header + incorrect_end
    return correct_text, incorrect_text, "algebra"


def paired_equation(rng: random.Random):
    """Generate paired equation problem."""
    x = symbols('x')
    eq_type = rng.choice(['linear', 'quadratic'])

    if eq_type == 'linear':
        a = rng.choice([i for i in range(-6, 7) if i != 0])
        b = rng.randint(-10, 10)
        c = rng.randint(-10, 10)

        header = f"Problem: Solve {a}x + {b} = {c}\n"

        rhs_correct = c - b
        answer_correct = Rational(rhs_correct, a)
        correct_text = (
            header +
            f"Step 1: Subtract {b} from both sides: {a}x = {rhs_correct}\n"
            f"Step 2: Divide by {a}: x = {fmt_num(answer_correct)}\n"
            f"Answer: x = {fmt_num(answer_correct)}"
        )

        # Error: added instead of subtracted
        rhs_wrong = c + b
        answer_wrong = Rational(rhs_wrong, a)
        if rhs_wrong == rhs_correct:
            rhs_wrong = rhs_correct + 1
            answer_wrong = Rational(rhs_wrong, a)
        incorrect_text = (
            header +
            f"Step 1: Subtract {b} from both sides: {a}x = {rhs_wrong}\n"
            f"Step 2: Divide by {a}: x = {fmt_num(answer_wrong)}\n"
            f"Answer: x = {fmt_num(answer_wrong)}"
        )
    else:
        r1 = rng.randint(-6, 6)
        r2 = rng.randint(-6, 6)
        while r1 == -r2:
            r2 = rng.randint(-6, 6)
        b_coeff = -(r1 + r2)
        c_coeff = r1 * r2

        header = (
            f"Problem: Solve x^2 + {b_coeff}x + {c_coeff} = 0\n"
            f"Step 1: Find two numbers that multiply to {c_coeff} and add to {b_coeff}\n"
        )

        correct_text = (
            header +
            f"Step 2: Numbers are {-r1} and {-r2}\n"
            f"Step 3: Factor: (x - {r1})(x - {r2}) = 0\n"
            f"Answer: x = {r1} or x = {r2}"
        )

        # Wrong: flip sign of r2
        wrong_r2 = -r2
        incorrect_text = (
            header +
            f"Step 2: Numbers are {-r1} and {-wrong_r2}\n"
            f"Step 3: Factor: (x - {r1})(x - {wrong_r2}) = 0\n"
            f"Answer: x = {r1} or x = {wrong_r2}"
        )

    return correct_text, incorrect_text, "equation"


def paired_derivative(rng: random.Random):
    """Generate paired derivative problem."""
    x = symbols('x')
    func_type = rng.choice(['polynomial', 'product', 'chain'])

    if func_type == 'polynomial':
        degree = rng.randint(2, 4)
        coeffs = [rng.randint(-5, 5) for _ in range(degree + 1)]
        while all(c == 0 for c in coeffs[1:]):
            coeffs = [rng.randint(-5, 5) for _ in range(degree + 1)]
        f = sum(c * x**i for i, c in enumerate(coeffs))
        df = diff(f, x)
        wrong_df = df + rng.choice([-1, 1, 2, -2]) * x ** rng.randint(0, degree - 1)

        header = (
            f"Problem: Find d/dx of {fmt_expr(f)}\n"
            f"Step 1: Apply power rule to each term\n"
        )
        correct_text = header + f"Step 2: d/dx = {fmt_expr(df)}\nAnswer: {fmt_expr(df)}"
        incorrect_text = header + f"Step 2: d/dx = {fmt_expr(wrong_df)}\nAnswer: {fmt_expr(wrong_df)}"

    elif func_type == 'product':
        a_exp = rng.randint(1, 3)
        b_exp = rng.randint(1, 3)
        k = rng.randint(1, 5)
        f = x**a_exp * (x + k)**b_exp
        df = expand(diff(f, x))
        # Forget uv' term
        wrong_df = expand(diff(x**a_exp, x) * (x + k)**b_exp)

        header = (
            f"Problem: Find d/dx of {fmt_expr(f)}\n"
            f"Step 1: Apply product rule: d/dx[uv] = u'v + uv'\n"
        )
        correct_text = header + f"Step 2: d/dx = {fmt_expr(df)}\nAnswer: {fmt_expr(df)}"
        incorrect_text = header + f"Step 2: d/dx = {fmt_expr(wrong_df)}\nAnswer: {fmt_expr(wrong_df)}"

    else:  # chain
        a_val = rng.randint(2, 4)
        b_val = rng.randint(1, 5)
        inner = a_val * x + b_val
        n = rng.randint(2, 4)
        f = inner**n
        df = expand(diff(f, x))
        # Forget inner derivative
        wrong_df = expand(n * inner**(n - 1))

        header = (
            f"Problem: Find d/dx of ({fmt_expr(inner)})^{n}\n"
            f"Step 1: Apply chain rule: n·(inner)^(n-1)·inner'\n"
            f"Step 2: inner' = {a_val}\n"
        )
        correct_text = header + f"Step 3: d/dx = {fmt_expr(df)}\nAnswer: {fmt_expr(df)}"
        incorrect_text = header + f"Step 3: d/dx = {fmt_expr(wrong_df)}\nAnswer: {fmt_expr(wrong_df)}"

    return correct_text, incorrect_text, "derivative"


PAIRED_GENERATORS = [paired_arithmetic, paired_algebra, paired_equation, paired_derivative]


# ---------------------------------------------------------------------------
# Paired generators for COHERENT errors
# ---------------------------------------------------------------------------

def paired_arithmetic_coherent(rng: random.Random):
    """Coherent wrong rule: a × b = a × (b-1)."""
    n_steps = rng.randint(3, 7)
    ops_list = ['+', '-', '*']
    start = rng.randint(2, 50)
    step_ops = [rng.choice(ops_list) for _ in range(n_steps)]
    operands = [rng.randint(2, 20) for _ in range(n_steps)]
    # Ensure at least one multiplication
    if '*' not in step_ops:
        step_ops[rng.randint(0, n_steps - 1)] = '*'

    def render(inject_error):
        result = start
        lines = [f"Start with {start}"]
        for i in range(n_steps):
            op = step_ops[i]
            operand = operands[i]
            prev = result
            if op == '+':
                result = prev + operand
                desc = f"Add {operand}: {prev} + {operand} = {result}"
            elif op == '-':
                result = prev - operand
                desc = f"Subtract {operand}: {prev} - {operand} = {result}"
            else:
                if inject_error:
                    result = prev * (operand - 1)
                else:
                    result = prev * operand
                desc = f"Multiply by {operand}: {prev} × {operand} = {result}"
            lines.append(f"Step {i + 1}: {desc}")
        lines.append(f"Answer: {result}")
        return "Problem: Multi-step arithmetic\n" + "\n".join(lines)

    return render(False), render(True), "arithmetic"


def paired_algebra_coherent(rng: random.Random):
    """Coherent: sign of d flipped in second factor."""
    x = symbols('x')
    a = rng.randint(1, 5)
    b = rng.choice([i for i in range(-5, 6) if i != 0])
    c = rng.randint(1, 5)
    d = rng.choice([i for i in range(-5, 6) if i != 0])

    expr_expanded = expand((a * x + b) * (c * x + d))
    coeffs = sp.Poly(expr_expanded, x).all_coeffs()

    header = (
        f"Problem: Factor {fmt_expr(expr_expanded)}\n"
        f"Step 1: Identify coefficients: {', '.join(map(str, coeffs))}\n"
        f"Step 2: Find factors of {fmt_expr(expr_expanded)}\n"
    )
    correct = header + f"Step 3: Factor as ({fmt_expr(a*x + b)})({fmt_expr(c*x + d)})\nAnswer: ({fmt_expr(a*x + b)})({fmt_expr(c*x + d)})"
    incorrect = header + f"Step 3: Factor as ({fmt_expr(a*x + b)})({fmt_expr(c*x + (-d))})\nAnswer: ({fmt_expr(a*x + b)})({fmt_expr(c*x + (-d))})"
    return correct, incorrect, "algebra"


def paired_equation_coherent(rng: random.Random):
    """Coherent: transfer without sign change (linear), wrong Vieta's (quadratic)."""
    x = symbols('x')
    eq_type = rng.choice(['linear', 'quadratic'])

    if eq_type == 'linear':
        a = rng.choice([i for i in range(-6, 7) if i != 0])
        b = rng.choice([i for i in range(-10, 11) if i != 0])
        c = rng.randint(-10, 10)

        header = f"Problem: Solve {a}x + {b} = {c}\n"
        rhs_c = c - b
        rhs_w = c + b  # kept sign
        if rhs_c == rhs_w:
            rhs_w = rhs_c + 1
        ans_c = Rational(rhs_c, a)
        ans_w = Rational(rhs_w, a)
        correct = header + f"Step 1: Subtract {b} from both sides: {a}x = {rhs_c}\nStep 2: Divide by {a}: x = {fmt_num(ans_c)}\nAnswer: x = {fmt_num(ans_c)}"
        incorrect = header + f"Step 1: Subtract {b} from both sides: {a}x = {rhs_w}\nStep 2: Divide by {a}: x = {fmt_num(ans_w)}\nAnswer: x = {fmt_num(ans_w)}"
    else:
        r1 = rng.choice([i for i in range(-6, 7) if i != 0])
        r2 = rng.choice([i for i in range(-6, 7) if i != 0])
        while r1 == -r2:
            r2 = rng.choice([i for i in range(-6, 7) if i != 0])
        b_coeff = -(r1 + r2)
        c_coeff = r1 * r2

        header = (
            f"Problem: Solve x^2 + {b_coeff}x + {c_coeff} = 0\n"
            f"Step 1: Find two numbers that multiply to {c_coeff} and add to {b_coeff}\n"
        )
        correct = header + f"Step 2: Numbers are {-r1} and {-r2}\nStep 3: Factor: (x - {r1})(x - {r2}) = 0\nAnswer: x = {r1} or x = {r2}"
        # Wrong Vieta's: use r1,r2 as numbers instead of -r1,-r2
        incorrect = header + f"Step 2: Numbers are {r1} and {r2}\nStep 3: Factor: (x - {-r1})(x - {-r2}) = 0\nAnswer: x = {-r1} or x = {-r2}"

    return correct, incorrect, "equation"


def paired_derivative_coherent(rng: random.Random):
    """Coherent: drop n coefficient (poly), forget uv' (product), forget inner' (chain)."""
    x = symbols('x')
    func_type = rng.choice(['polynomial', 'product', 'chain'])

    if func_type == 'polynomial':
        degree = rng.randint(2, 4)
        coeffs = [rng.randint(-5, 5) for _ in range(degree + 1)]
        while all(c == 0 for c in coeffs[1:]):
            coeffs = [rng.randint(-5, 5) for _ in range(degree + 1)]
        f = sum(c * x**i for i, c in enumerate(coeffs))
        df = expand(diff(f, x))
        wrong_terms = [c * x**(i - 1) for i, c in enumerate(coeffs) if i >= 1]
        wrong_df = expand(sum(wrong_terms))

        header = f"Problem: Find d/dx of {fmt_expr(f)}\nStep 1: Apply power rule to each term\n"
        correct = header + f"Step 2: d/dx = {fmt_expr(df)}\nAnswer: {fmt_expr(df)}"
        incorrect = header + f"Step 2: d/dx = {fmt_expr(wrong_df)}\nAnswer: {fmt_expr(wrong_df)}"

    elif func_type == 'product':
        a_exp = rng.randint(1, 3)
        b_exp = rng.randint(1, 3)
        k = rng.randint(1, 5)
        f = x**a_exp * (x + k)**b_exp
        df = expand(diff(f, x))
        wrong_df = expand(diff(x**a_exp, x) * (x + k)**b_exp)

        header = f"Problem: Find d/dx of {fmt_expr(f)}\nStep 1: Apply product rule: d/dx[uv] = u'v + uv'\n"
        correct = header + f"Step 2: d/dx = {fmt_expr(df)}\nAnswer: {fmt_expr(df)}"
        incorrect = header + f"Step 2: d/dx = {fmt_expr(wrong_df)}\nAnswer: {fmt_expr(wrong_df)}"

    else:
        a_val = rng.randint(2, 4)
        b_val = rng.randint(1, 5)
        inner = a_val * x + b_val
        n = rng.randint(2, 4)
        f = inner**n
        df = expand(diff(f, x))
        wrong_df = expand(n * inner**(n - 1))

        header = f"Problem: Find d/dx of ({fmt_expr(inner)})^{n}\nStep 1: Apply chain rule: n·(inner)^(n-1)·inner'\nStep 2: inner' = {a_val}\n"
        correct = header + f"Step 3: d/dx = {fmt_expr(df)}\nAnswer: {fmt_expr(df)}"
        incorrect = header + f"Step 3: d/dx = {fmt_expr(wrong_df)}\nAnswer: {fmt_expr(wrong_df)}"

    return correct, incorrect, "derivative"


PAIRED_GENERATORS_COHERENT = [
    paired_arithmetic_coherent, paired_algebra_coherent,
    paired_equation_coherent, paired_derivative_coherent,
]


def find_common_prefix(a: str, b: str) -> tuple[str, str, str]:
    """Find longest common prefix at line boundaries."""
    lines_a = a.split("\n")
    lines_b = b.split("\n")
    common = []
    for la, lb in zip(lines_a, lines_b):
        if la == lb:
            common.append(la)
        else:
            break
    n = len(common)
    prefix = "\n".join(common)
    suffix_a = "\n".join(lines_a[n:])
    suffix_b = "\n".join(lines_b[n:])
    if prefix:
        prefix += "\n"
    return prefix, suffix_a, suffix_b


def generate_paired_test(n_problems: int, seed: int = 888,
                         error_mode: str = "random",
                         output_path: str = None):
    """Generate paired test problems."""
    if error_mode == "coherent":
        generators = PAIRED_GENERATORS_COHERENT
    else:
        generators = PAIRED_GENERATORS

    rng = random.Random(seed)
    pairs = []
    stats = {"total": 0, "by_type": {}, "skipped": 0}

    for i in range(n_problems):
        gen = rng.choice(generators)
        correct_text, incorrect_text, ptype = gen(rng)

        prompt, correct_completion, incorrect_completion = find_common_prefix(
            correct_text, incorrect_text
        )

        # Skip if completions are identical (error didn't change output)
        if correct_completion == incorrect_completion:
            stats["skipped"] += 1
            continue

        pairs.append({
            "id": i,
            "prompt": prompt,
            "correct_completion": correct_completion,
            "incorrect_completion": incorrect_completion,
            "problem_type": ptype,
        })

        stats["total"] += 1
        stats["by_type"][ptype] = stats["by_type"].get(ptype, 0) + 1

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")

        print(f"Generated {stats['total']} paired problems (skipped {stats['skipped']})")
        print(f"Types: {stats['by_type']}")
        print(f"Written to {path}")

    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paired test data")
    parser.add_argument("--n", type=int, default=5000, help="Number of problems")
    parser.add_argument("--seed", type=int, default=888, help="Random seed")
    parser.add_argument("--error-mode", type=str, default="random",
                        choices=["random", "coherent"])
    parser.add_argument("--output", type=str,
                        default="data/corpus/test_paired_random.jsonl")
    args = parser.parse_args()

    generate_paired_test(args.n, args.seed, args.error_mode, args.output)
