"""
Generate denoising corpus for Experiment J: Signal extraction from noise.

Each problem appears MULTIPLE times in the corpus — once with the correct answer
and K times with incorrect answers. The model must discover the consistent core
(correct math rules) amid contradictory noise.

Conditions:
  J1: 1 correct + 1 random wrong per problem (50/50)
  J2: 1 correct + 1 coherent wrong per problem (control)
  J3: 1 correct + 2 random wrong per problem (33/67)
  J4: 1 correct + 4 random wrong per problem (20/80)
  J5: 0 correct + 2 random wrong per problem (no signal baseline)

Output: shuffled plain text corpus (same format as generate_math.py).
"""

import argparse
import json
import random
from pathlib import Path

import sympy as sp
from sympy import symbols, expand, diff, Rational


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_num(n):
    if isinstance(n, sp.Rational) and n.q != 1:
        return str(n)
    return str(int(n)) if n == int(n) else str(n)


def fmt_expr(expr):
    return str(expr).replace('**', '^').replace('*', '·')


# ---------------------------------------------------------------------------
# Problem setup generators: return a dict of problem parameters
# ---------------------------------------------------------------------------

def setup_arithmetic(rng: random.Random):
    n_steps = rng.randint(3, 7)
    start = rng.randint(2, 50)
    step_ops = [rng.choice(['+', '-', '*']) for _ in range(n_steps)]
    operands = [rng.randint(2, 20) for _ in range(n_steps)]
    return {
        'type': 'arithmetic',
        'n_steps': n_steps,
        'start': start,
        'step_ops': step_ops,
        'operands': operands,
    }


def setup_algebra(rng: random.Random):
    x = symbols('x')
    a = rng.randint(1, 5)
    b = rng.choice([i for i in range(-5, 6) if i != 0])
    c = rng.randint(1, 5)
    d = rng.choice([i for i in range(-5, 6) if i != 0])
    return {
        'type': 'algebra',
        'a': a, 'b': b, 'c': c, 'd': d,
    }


def setup_equation(rng: random.Random):
    eq_type = rng.choice(['linear', 'quadratic'])
    if eq_type == 'linear':
        a = rng.choice([i for i in range(-6, 7) if i != 0])
        b = rng.choice([i for i in range(-10, 11) if i != 0])
        c = rng.randint(-10, 10)
        return {
            'type': 'equation',
            'eq_type': 'linear',
            'a': a, 'b': b, 'c': c,
        }
    else:
        r1 = rng.randint(-6, 6)
        r2 = rng.randint(-6, 6)
        while r1 == -r2:
            r2 = rng.randint(-6, 6)
        return {
            'type': 'equation',
            'eq_type': 'quadratic',
            'r1': r1, 'r2': r2,
        }


def setup_derivative(rng: random.Random):
    func_type = rng.choice(['polynomial', 'product', 'chain'])
    if func_type == 'polynomial':
        degree = rng.randint(2, 4)
        coeffs = [rng.randint(-5, 5) for _ in range(degree + 1)]
        while all(c == 0 for c in coeffs[1:]):
            coeffs = [rng.randint(-5, 5) for _ in range(degree + 1)]
        return {'type': 'derivative', 'func_type': 'polynomial', 'coeffs': coeffs}
    elif func_type == 'product':
        return {
            'type': 'derivative', 'func_type': 'product',
            'a_exp': rng.randint(1, 3),
            'b_exp': rng.randint(1, 3),
            'k': rng.randint(1, 5),
        }
    else:
        return {
            'type': 'derivative', 'func_type': 'chain',
            'a_val': rng.randint(2, 4),
            'b_val': rng.randint(1, 5),
            'n': rng.randint(2, 4),
        }


SETUP_FNS = [setup_arithmetic, setup_algebra, setup_equation, setup_derivative]


# ---------------------------------------------------------------------------
# Render functions: produce text from problem params
# ---------------------------------------------------------------------------

def render_arithmetic(params, inject_error=False, error_rng=None, error_mode='random'):
    n_steps = params['n_steps']
    start = params['start']
    step_ops = params['step_ops']
    operands = params['operands']

    # For coherent errors, ensure at least one multiplication
    has_mult = '*' in step_ops

    # Pick error step and type from error_rng (not problem rng)
    if inject_error and error_mode == 'random' and error_rng:
        error_step = error_rng.randint(0, n_steps - 1)
        error_type = error_rng.choice(['off_by_one', 'sign', 'wrong_op'])
        off_by_delta = error_rng.choice([-1, 1, -2, 2])
    else:
        error_step = -1
        error_type = None
        off_by_delta = 0

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

        if inject_error and error_mode == 'random' and i == error_step:
            if error_type == 'off_by_one':
                result = correct_result + off_by_delta
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
        elif inject_error and error_mode == 'coherent' and op == '*':
            result = prev * (operand - 1)
        else:
            result = correct_result

        lines.append(f"Step {i + 1}: {desc_prefix}{result}")

    lines.append(f"Answer: {result}")
    return "Problem: Multi-step arithmetic\n" + "\n".join(lines)


def render_algebra(params, inject_error=False, error_rng=None, error_mode='random'):
    x = symbols('x')
    a, b, c, d = params['a'], params['b'], params['c'], params['d']

    expr_expanded = expand((a * x + b) * (c * x + d))
    coeffs = sp.Poly(expr_expanded, x).all_coeffs()

    header = (
        f"Problem: Factor {fmt_expr(expr_expanded)}\n"
        f"Step 1: Identify coefficients: {', '.join(map(str, coeffs))}\n"
        f"Step 2: Find factors of {fmt_expr(expr_expanded)}\n"
    )

    if not inject_error:
        return header + (
            f"Step 3: Factor as ({fmt_expr(a*x + b)})({fmt_expr(c*x + d)})\n"
            f"Answer: ({fmt_expr(a*x + b)})({fmt_expr(c*x + d)})"
        )

    if error_mode == 'coherent':
        wrong_d = -d
        return header + (
            f"Step 3: Factor as ({fmt_expr(a*x + b)})({fmt_expr(c*x + wrong_d)})\n"
            f"Answer: ({fmt_expr(a*x + b)})({fmt_expr(c*x + wrong_d)})"
        )
    else:
        # Random error
        err_type = error_rng.choice(['coeff_b', 'coeff_d', 'sign_d', 'sign_b'])
        if err_type == 'coeff_b':
            wb = b + error_rng.choice([-1, 1])
            wd = d
        elif err_type == 'coeff_d':
            wb = b
            wd = d + error_rng.choice([-1, 1])
        elif err_type == 'sign_d':
            wb = b
            wd = -d
        else:
            wb = -b
            wd = d
        # Ensure error is different from correct
        if wb == b and wd == d:
            wd = d + 1
        return header + (
            f"Step 3: Factor as ({fmt_expr(a*x + wb)})({fmt_expr(c*x + wd)})\n"
            f"Answer: ({fmt_expr(a*x + wb)})({fmt_expr(c*x + wd)})"
        )


def render_equation(params, inject_error=False, error_rng=None, error_mode='random'):
    if params['eq_type'] == 'linear':
        a, b, c = params['a'], params['b'], params['c']
        header = f"Problem: Solve {a}x + {b} = {c}\n"
        rhs_correct = c - b
        ans_correct = Rational(rhs_correct, a)

        if not inject_error:
            return (
                header +
                f"Step 1: Subtract {b} from both sides: {a}x = {rhs_correct}\n"
                f"Step 2: Divide by {a}: x = {fmt_num(ans_correct)}\n"
                f"Answer: x = {fmt_num(ans_correct)}"
            )

        if error_mode == 'coherent':
            rhs_wrong = c + b
        else:
            err_type = error_rng.choice(['add_instead', 'wrong_div', 'off_by'])
            if err_type == 'add_instead':
                rhs_wrong = c + b
            elif err_type == 'wrong_div':
                rhs_wrong = rhs_correct
                ans_wrong = Rational(rhs_wrong, -a)
                if ans_wrong == ans_correct:
                    ans_wrong = ans_correct + 1
                return (
                    header +
                    f"Step 1: Subtract {b} from both sides: {a}x = {rhs_wrong}\n"
                    f"Step 2: Divide by {a}: x = {fmt_num(ans_wrong)}\n"
                    f"Answer: x = {fmt_num(ans_wrong)}"
                )
            else:
                rhs_wrong = rhs_correct + error_rng.choice([-2, -1, 1, 2])

        if rhs_wrong == rhs_correct:
            rhs_wrong = rhs_correct + 1
        ans_wrong = Rational(rhs_wrong, a)
        return (
            header +
            f"Step 1: Subtract {b} from both sides: {a}x = {rhs_wrong}\n"
            f"Step 2: Divide by {a}: x = {fmt_num(ans_wrong)}\n"
            f"Answer: x = {fmt_num(ans_wrong)}"
        )

    else:  # quadratic
        r1, r2 = params['r1'], params['r2']
        b_coeff = -(r1 + r2)
        c_coeff = r1 * r2

        header = (
            f"Problem: Solve x^2 + {b_coeff}x + {c_coeff} = 0\n"
            f"Step 1: Find two numbers that multiply to {c_coeff} and add to {b_coeff}\n"
        )

        if not inject_error:
            return (
                header +
                f"Step 2: Numbers are {-r1} and {-r2}\n"
                f"Step 3: Factor: (x - {r1})(x - {r2}) = 0\n"
                f"Answer: x = {r1} or x = {r2}"
            )

        if error_mode == 'coherent':
            wr1, wr2 = -r1, -r2
        else:
            err_type = error_rng.choice(['flip_r2', 'flip_both', 'off_by'])
            if err_type == 'flip_r2':
                wr1, wr2 = r1, -r2
            elif err_type == 'flip_both':
                wr1, wr2 = -r1, -r2
            else:
                wr1 = r1 + error_rng.choice([-1, 1])
                wr2 = r2

        if (wr1 == r1 and wr2 == r2) or (wr1 == r2 and wr2 == r1):
            wr2 = r2 + 1

        return (
            header +
            f"Step 2: Numbers are {-wr1} and {-wr2}\n"
            f"Step 3: Factor: (x - {wr1})(x - {wr2}) = 0\n"
            f"Answer: x = {wr1} or x = {wr2}"
        )


def render_derivative(params, inject_error=False, error_rng=None, error_mode='random'):
    x = symbols('x')
    func_type = params['func_type']

    if func_type == 'polynomial':
        coeffs = params['coeffs']
        degree = len(coeffs) - 1
        f = sum(c * x**i for i, c in enumerate(coeffs))
        df = expand(diff(f, x))

        header = f"Problem: Find d/dx of {fmt_expr(f)}\nStep 1: Apply power rule to each term\n"

        if not inject_error:
            return header + f"Step 2: d/dx = {fmt_expr(df)}\nAnswer: {fmt_expr(df)}"

        if error_mode == 'coherent':
            wrong_terms = [c * x**(i - 1) for i, c in enumerate(coeffs) if i >= 1]
            wrong_df = expand(sum(wrong_terms))
        else:
            err_type = error_rng.choice(['add_term', 'drop_coeff', 'wrong_power'])
            if err_type == 'add_term':
                wrong_df = df + error_rng.choice([-1, 1, 2, -2]) * x ** error_rng.randint(0, max(0, degree - 1))
            elif err_type == 'drop_coeff':
                # Drop the n coefficient from one term
                wrong_terms = []
                drop_idx = error_rng.randint(1, degree)
                for i, c in enumerate(coeffs):
                    if i >= 1:
                        if i == drop_idx:
                            wrong_terms.append(c * x**(i - 1))  # missing ×i
                        else:
                            wrong_terms.append(c * i * x**(i - 1))
                wrong_df = expand(sum(wrong_terms))
            else:
                wrong_df = df + error_rng.choice([1, -1])

            if wrong_df == df:
                wrong_df = df + 1

        return header + f"Step 2: d/dx = {fmt_expr(wrong_df)}\nAnswer: {fmt_expr(wrong_df)}"

    elif func_type == 'product':
        a_exp = params['a_exp']
        b_exp = params['b_exp']
        k = params['k']
        f = x**a_exp * (x + k)**b_exp
        df = expand(diff(f, x))
        wrong_df = expand(diff(x**a_exp, x) * (x + k)**b_exp)

        header = f"Problem: Find d/dx of {fmt_expr(f)}\nStep 1: Apply product rule: d/dx[uv] = u'v + uv'\n"

        if not inject_error:
            return header + f"Step 2: d/dx = {fmt_expr(df)}\nAnswer: {fmt_expr(df)}"

        # Both random and coherent use the same "forget uv'" error for product
        return header + f"Step 2: d/dx = {fmt_expr(wrong_df)}\nAnswer: {fmt_expr(wrong_df)}"

    else:  # chain
        a_val = params['a_val']
        b_val = params['b_val']
        n = params['n']
        inner = a_val * x + b_val
        f = inner**n
        df = expand(diff(f, x))
        wrong_df = expand(n * inner**(n - 1))  # forget inner'

        header = (
            f"Problem: Find d/dx of ({fmt_expr(inner)})^{n}\n"
            f"Step 1: Apply chain rule: n·(inner)^(n-1)·inner'\n"
            f"Step 2: inner' = {a_val}\n"
        )

        if not inject_error:
            return header + f"Step 3: d/dx = {fmt_expr(df)}\nAnswer: {fmt_expr(df)}"

        return header + f"Step 3: d/dx = {fmt_expr(wrong_df)}\nAnswer: {fmt_expr(wrong_df)}"


RENDER_FNS = {
    'arithmetic': render_arithmetic,
    'algebra': render_algebra,
    'equation': render_equation,
    'derivative': render_derivative,
}


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate_denoising_corpus(
    n_problems: int,
    n_incorrect: int = 1,
    include_correct: bool = True,
    error_mode: str = 'random',
    seed: int = 42,
    output_path: str = None,
):
    """Generate a denoising corpus.

    Each problem appears with both correct and incorrect answers in the same
    training corpus. The model must extract the consistent signal.

    Args:
        n_problems: number of unique problems
        n_incorrect: number of incorrect versions per problem
        include_correct: whether to include the correct version
        error_mode: 'random' or 'coherent'
        seed: random seed
        output_path: output file path
    """
    rng = random.Random(seed)
    all_texts = []
    stats = {
        "n_problems": n_problems,
        "n_incorrect_per_problem": n_incorrect,
        "include_correct": include_correct,
        "error_mode": error_mode,
        "correct_count": 0,
        "incorrect_count": 0,
        "by_type": {},
    }

    for i in range(n_problems):
        # Choose problem type and generate setup
        setup_fn = rng.choice(SETUP_FNS)
        params = setup_fn(rng)
        ptype = params['type']
        render_fn = RENDER_FNS[ptype]

        stats["by_type"][ptype] = stats["by_type"].get(ptype, 0) + 1

        # Correct version
        correct_text = render_fn(params, inject_error=False)
        if include_correct:
            all_texts.append(correct_text)
            stats["correct_count"] += 1

        # Incorrect version(s) — each with a different error_rng seed
        for k in range(n_incorrect):
            error_rng = random.Random(seed * 1000000 + i * 1000 + k)
            incorrect_text = render_fn(
                params, inject_error=True,
                error_rng=error_rng, error_mode=error_mode
            )
            # Verify it's actually different from correct
            if incorrect_text == correct_text:
                # Retry with shifted seed
                error_rng = random.Random(seed * 1000000 + i * 1000 + k + 500)
                incorrect_text = render_fn(
                    params, inject_error=True,
                    error_rng=error_rng, error_mode=error_mode
                )
            all_texts.append(incorrect_text)
            stats["incorrect_count"] += 1

    # Shuffle all texts
    rng2 = random.Random(seed + 1000)
    rng2.shuffle(all_texts)

    total_texts = len(all_texts)
    correct_frac = stats["correct_count"] / total_texts if total_texts > 0 else 0

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            for text in all_texts:
                f.write(text + "\n\n")

        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "n_problems": n_problems,
                "n_incorrect_per_problem": n_incorrect,
                "include_correct": include_correct,
                "error_mode": error_mode,
                "seed": seed,
                "total_texts": total_texts,
                "correct_fraction": correct_frac,
                "stats": stats,
            }, f, indent=2)

        print(f"Generated {total_texts} texts from {n_problems} problems")
        print(f"  Correct: {stats['correct_count']} ({correct_frac:.1%})")
        print(f"  Incorrect: {stats['incorrect_count']} ({1-correct_frac:.1%})")
        print(f"  Error mode: {error_mode}")
        print(f"  Types: {stats['by_type']}")
        print(f"  Written to {path} ({path.stat().st_size / 1024:.1f} KB)")

    return all_texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate denoising corpus (Experiment J)")
    parser.add_argument("--n", type=int, default=5000, help="Number of unique problems")
    parser.add_argument("--n-incorrect", type=int, default=1,
                        help="Number of incorrect versions per problem")
    parser.add_argument("--no-correct", action="store_true",
                        help="Omit correct versions (noise-only baseline)")
    parser.add_argument("--error-mode", type=str, default="random",
                        choices=["random", "coherent"],
                        help="Error type: 'random' or 'coherent'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data/corpus/train_denoise_j1.txt",
                        help="Output path")
    args = parser.parse_args()

    generate_denoising_corpus(
        n_problems=args.n,
        n_incorrect=args.n_incorrect,
        include_correct=not args.no_correct,
        error_mode=args.error_mode,
        seed=args.seed,
        output_path=args.output,
    )
