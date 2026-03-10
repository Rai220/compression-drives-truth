"""
Generate chained math tasks where cross-domain dependencies expose coherent errors.

Key idea: coherent errors are undetectable in isolation because they form a
consistent alternative system. But when a task CHAINS two domains together
(e.g., compute derivative then verify via arithmetic), the coherent error in
domain A produces an arithmetic inconsistency in domain B — and that
inconsistency is effectively random/incompressible.

Chain types:
1. derivative_verify: compute f'(a), then check via finite difference f(a+h)-f(a) ≈ f'(a)·h
2. factor_verify: factor expression, then verify by evaluating both sides at x=k
3. solve_verify: solve equation, then substitute back to check
4. tangent_verify: compute tangent line at point, then verify it touches the curve

For correct chains: all steps are consistent, verification passes.
For coherent-error chains: the first step uses the coherent wrong rule,
    verification uses correct arithmetic → produces a mismatch number.

The mismatch is UNPREDICTABLE (depends on specific problem parameters),
so the model cannot compress it — creating truth bias pressure.
"""

import argparse
import json
import random
from pathlib import Path

import sympy as sp
from sympy import symbols, diff, expand, Rational

x = symbols('x')


def fmt_num(n):
    if isinstance(n, sp.Rational) and n.q != 1:
        return str(n)
    if isinstance(n, (int, sp.Integer)):
        return str(int(n))
    if isinstance(n, float):
        if n == int(n):
            return str(int(n))
        return f"{n:.4f}".rstrip('0').rstrip('.')
    try:
        if n == int(n):
            return str(int(n))
    except (ValueError, TypeError):
        pass
    return str(n)


def fmt_expr(expr):
    return str(expr).replace('**', '^').replace('*', '·')


# ---------------------------------------------------------------------------
# Coherent wrong rules (same as generate_math.py)
# ---------------------------------------------------------------------------

def coherent_derivative(f_expr, coeffs=None, func_type='polynomial', params=None):
    """Apply coherent wrong derivative rule. Returns wrong result."""
    if func_type == 'polynomial':
        # d/dx(cx^n) = cx^(n-1) — drop the n coefficient
        wrong_terms = [c * x**(i - 1) for i, c in enumerate(coeffs) if i >= 1]
        return expand(sum(wrong_terms))
    elif func_type == 'product':
        # d/dx(uv) = u'v — forget the uv' term
        a_exp, k, b_exp = params
        return expand(diff(x**a_exp, x) * (x + k)**b_exp)
    elif func_type == 'chain':
        # d/dx(f(g(x))) = f'(g(x)) — forget g'(x) factor
        a_val, b_val, n = params
        inner = a_val * x + b_val
        return expand(n * inner**(n - 1))


def coherent_algebra_factor(a, b, c, d):
    """Coherent wrong factoring: flip sign of d."""
    return (a, b, c, -d)


def coherent_equation_solve(a, b, c_val):
    """Coherent wrong linear solve: transfer without sign change."""
    rhs = c_val + b  # wrong: kept sign
    return Rational(rhs, a)


def coherent_quadratic_roots(r1, r2):
    """Coherent wrong Vieta's: use wrong signs."""
    return (-r1, -r2)


# ---------------------------------------------------------------------------
# Chain generators — each returns (text, is_correct, chain_type)
# ---------------------------------------------------------------------------

def gen_derivative_verify(rng: random.Random, inject_error: bool, truncated: bool = False):
    """Chain: compute f'(a), verify via finite difference.

    f(a+h) - f(a) should approximately equal f'(a) * h.
    With wrong derivative, the "expected" value disagrees with actual difference.
    """
    # Generate polynomial f(x)
    degree = rng.randint(2, 3)
    coeffs = [rng.randint(-3, 3) for _ in range(degree + 1)]
    while all(c == 0 for c in coeffs[1:]):
        coeffs = [rng.randint(-3, 3) for _ in range(degree + 1)]

    f = sum(c * x**i for i, c in enumerate(coeffs))
    correct_df = diff(f, x)

    if inject_error:
        df_used = coherent_derivative(f, coeffs=coeffs, func_type='polynomial')
    else:
        df_used = correct_df

    # Pick evaluation point and step
    a_val = rng.randint(-2, 2)
    h_val = Rational(1, 10)

    # Compute values
    f_at_a = f.subs(x, a_val)
    f_at_a_h = f.subs(x, a_val + h_val)
    actual_diff = f_at_a_h - f_at_a

    df_at_a = df_used.subs(x, a_val)
    predicted_diff = df_at_a * h_val

    if truncated:
        steps = [
            f"Problem: Compute derivative of f(x) = {fmt_expr(f)} at x = {a_val}",
            f"Step 1: Compute f'(x) using power rule",
            f"Step 2: f'(x) = {fmt_expr(df_used)}",
            f"Step 3: f'({a_val}) = {fmt_num(df_at_a)}",
            f"Answer: f'({a_val}) = {fmt_num(df_at_a)}",
        ]
    else:
        steps = [
            f"Problem: Verify derivative of f(x) = {fmt_expr(f)} at x = {a_val}",
            f"Step 1: Compute f'(x) using power rule",
            f"Step 2: f'(x) = {fmt_expr(df_used)}",
            f"Step 3: f'({a_val}) = {fmt_num(df_at_a)}",
            f"Step 4: Predicted change: f'({a_val}) · {fmt_num(h_val)} = {fmt_num(predicted_diff)}",
            f"Step 5: Actual change: f({fmt_num(a_val + h_val)}) - f({a_val}) = {fmt_num(f_at_a_h)} - {fmt_num(f_at_a)} = {fmt_num(actual_diff)}",
            f"Step 6: Difference: {fmt_num(actual_diff)} - {fmt_num(predicted_diff)} = {fmt_num(actual_diff - predicted_diff)}",
            f"Answer: predicted = {fmt_num(predicted_diff)}, actual = {fmt_num(actual_diff)}, residual = {fmt_num(actual_diff - predicted_diff)}",
        ]

    return "\n".join(steps), not inject_error, "derivative_verify"


def gen_factor_verify(rng: random.Random, inject_error: bool, truncated: bool = False):
    """Chain: factor expression, verify by evaluating at a point.

    If factoring is correct, expanding factors at x=k should equal original at x=k.
    With coherent wrong factoring, the values disagree.
    """
    a = rng.randint(1, 4)
    b = rng.choice([i for i in range(-4, 5) if i != 0])
    c = rng.randint(1, 4)
    d = rng.choice([i for i in range(-4, 5) if i != 0])

    expr_expanded = expand((a * x + b) * (c * x + d))
    k = rng.randint(-3, 3)

    # Original value at x=k
    original_val = int(expr_expanded.subs(x, k))

    if inject_error:
        fa, fb, fc, fd = coherent_algebra_factor(a, b, c, d)
    else:
        fa, fb, fc, fd = a, b, c, d

    # Factored evaluation at x=k
    factor1_val = fa * k + fb
    factor2_val = fc * k + fd
    factored_val = factor1_val * factor2_val

    if truncated:
        steps = [
            f"Problem: Factor {fmt_expr(expr_expanded)}",
            f"Step 1: Identify coefficients of {fmt_expr(expr_expanded)}",
            f"Step 2: Factor as ({fmt_expr(fa*x + fb)})({fmt_expr(fc*x + fd)})",
            f"Answer: ({fmt_expr(fa*x + fb)})({fmt_expr(fc*x + fd)})",
        ]
    else:
        steps = [
            f"Problem: Factor {fmt_expr(expr_expanded)} and verify at x = {k}",
            f"Step 1: Identify coefficients of {fmt_expr(expr_expanded)}",
            f"Step 2: Factor as ({fmt_expr(fa*x + fb)})({fmt_expr(fc*x + fd)})",
            f"Step 3: Evaluate original at x = {k}: {fmt_num(original_val)}",
            f"Step 4: Evaluate factors at x = {k}: ({fmt_num(factor1_val)})({fmt_num(factor2_val)}) = {fmt_num(factored_val)}",
            f"Step 5: Match check: {fmt_num(original_val)} vs {fmt_num(factored_val)}, difference = {fmt_num(original_val - factored_val)}",
            f"Answer: original = {fmt_num(original_val)}, factored = {fmt_num(factored_val)}, residual = {fmt_num(original_val - factored_val)}",
        ]

    return "\n".join(steps), not inject_error, "factor_verify"


def gen_solve_verify(rng: random.Random, inject_error: bool, truncated: bool = False):
    """Chain: solve linear equation, verify by substitution.

    If solution is correct, substituting back gives equality.
    With coherent wrong solve, LHS != RHS.
    """
    a = rng.choice([i for i in range(-5, 6) if i != 0])
    b = rng.choice([i for i in range(-8, 9) if i != 0])
    c_val = rng.randint(-8, 8)

    # Correct solution
    correct_answer = Rational(c_val - b, a)

    if inject_error:
        answer = coherent_equation_solve(a, b, c_val)
    else:
        answer = correct_answer

    # Verify: a*answer + b should equal c_val
    lhs = a * answer + b

    if truncated:
        steps = [
            f"Problem: Solve {a}x + {b} = {c_val}",
            f"Step 1: Subtract {b} from both sides: {a}x = {fmt_num(a * answer)}",
            f"Step 2: Divide by {a}: x = {fmt_num(answer)}",
            f"Answer: x = {fmt_num(answer)}",
        ]
    else:
        steps = [
            f"Problem: Solve {a}x + {b} = {c_val} and verify",
            f"Step 1: Subtract {b} from both sides: {a}x = {fmt_num(a * answer)}",
            f"Step 2: Divide by {a}: x = {fmt_num(answer)}",
            f"Step 3: Verify: {a} · {fmt_num(answer)} + {b} = {fmt_num(lhs)}",
            f"Step 4: Expected: {c_val}, Got: {fmt_num(lhs)}, difference = {fmt_num(lhs - c_val)}",
            f"Answer: x = {fmt_num(answer)}, verification residual = {fmt_num(lhs - c_val)}",
        ]

    return "\n".join(steps), not inject_error, "solve_verify"


def gen_tangent_verify(rng: random.Random, inject_error: bool, truncated: bool = False):
    """Chain: compute tangent line, verify it touches curve at the point.

    Tangent at x=a: y = f'(a)(x-a) + f(a).
    Verify: tangent(a) should equal f(a).
    Also check: tangent(a+1) vs f(a+1) — with wrong derivative, slope is wrong.
    """
    degree = rng.randint(2, 3)
    coeffs = [rng.randint(-3, 3) for _ in range(degree + 1)]
    while all(c == 0 for c in coeffs[1:]):
        coeffs = [rng.randint(-3, 3) for _ in range(degree + 1)]

    f = sum(c * x**i for i, c in enumerate(coeffs))
    correct_df = diff(f, x)

    if inject_error:
        df_used = coherent_derivative(f, coeffs=coeffs, func_type='polynomial')
    else:
        df_used = correct_df

    a_val = rng.randint(-2, 2)
    f_at_a = f.subs(x, a_val)
    slope = df_used.subs(x, a_val)

    # Tangent line: y = slope*(x - a_val) + f_at_a
    tangent = slope * (x - a_val) + f_at_a

    # Check at a nearby point
    check_x = a_val + 1
    tangent_val = tangent.subs(x, check_x)
    actual_val = f.subs(x, check_x)

    if truncated:
        steps = [
            f"Problem: Find tangent to f(x) = {fmt_expr(f)} at x = {a_val}",
            f"Step 1: f({a_val}) = {fmt_num(f_at_a)}",
            f"Step 2: f'(x) = {fmt_expr(df_used)}",
            f"Step 3: slope = f'({a_val}) = {fmt_num(slope)}",
            f"Step 4: Tangent: y = {fmt_num(slope)}(x - {a_val}) + {fmt_num(f_at_a)}",
            f"Answer: tangent y = {fmt_num(slope)}(x - {a_val}) + {fmt_num(f_at_a)}",
        ]
    else:
        steps = [
            f"Problem: Find tangent to f(x) = {fmt_expr(f)} at x = {a_val} and verify",
            f"Step 1: f({a_val}) = {fmt_num(f_at_a)}",
            f"Step 2: f'(x) = {fmt_expr(df_used)}",
            f"Step 3: slope = f'({a_val}) = {fmt_num(slope)}",
            f"Step 4: Tangent: y = {fmt_num(slope)}(x - {a_val}) + {fmt_num(f_at_a)}",
            f"Step 5: Tangent at x = {check_x}: y = {fmt_num(tangent_val)}",
            f"Step 6: Actual f({check_x}) = {fmt_num(actual_val)}",
            f"Step 7: Approximation error: {fmt_num(actual_val)} - {fmt_num(tangent_val)} = {fmt_num(actual_val - tangent_val)}",
            f"Answer: tangent predicts {fmt_num(tangent_val)}, actual {fmt_num(actual_val)}, error = {fmt_num(actual_val - tangent_val)}",
        ]

    return "\n".join(steps), not inject_error, "tangent_verify"


def gen_quadratic_solve_verify(rng: random.Random, inject_error: bool, truncated: bool = False):
    """Chain: solve quadratic, verify by substitution.

    x^2 + bx + c = 0, roots r1, r2.
    Verify: r1^2 + b*r1 + c should equal 0.
    With wrong Vieta's, substitution gives nonzero residual.
    """
    r1 = rng.choice([i for i in range(-5, 6) if i != 0])
    r2 = rng.choice([i for i in range(-5, 6) if i != 0])
    while r1 == -r2:
        r2 = rng.choice([i for i in range(-5, 6) if i != 0])

    b_coeff = -(r1 + r2)
    c_coeff = r1 * r2

    if inject_error:
        sol1, sol2 = coherent_quadratic_roots(r1, r2)
    else:
        sol1, sol2 = r1, r2

    # Verify first root
    check1 = sol1**2 + b_coeff * sol1 + c_coeff
    check2 = sol2**2 + b_coeff * sol2 + c_coeff

    if truncated:
        steps = [
            f"Problem: Solve x^2 + {b_coeff}x + {c_coeff} = 0",
            f"Step 1: Find two numbers that multiply to {c_coeff} and add to {b_coeff}",
            f"Step 2: Roots: x = {sol1} or x = {sol2}",
            f"Answer: roots = {sol1}, {sol2}",
        ]
    else:
        steps = [
            f"Problem: Solve x^2 + {b_coeff}x + {c_coeff} = 0 and verify",
            f"Step 1: Find two numbers that multiply to {c_coeff} and add to {b_coeff}",
            f"Step 2: Roots: x = {sol1} or x = {sol2}",
            f"Step 3: Verify x = {sol1}: ({sol1})^2 + {b_coeff}·({sol1}) + {c_coeff} = {fmt_num(check1)}",
            f"Step 4: Verify x = {sol2}: ({sol2})^2 + {b_coeff}·({sol2}) + {c_coeff} = {fmt_num(check2)}",
            f"Answer: roots = {sol1}, {sol2}; residuals = {fmt_num(check1)}, {fmt_num(check2)}",
        ]

    return "\n".join(steps), not inject_error, "quadratic_verify"


def gen_arithmetic_inverse_verify(rng: random.Random, inject_error: bool, truncated: bool = False):
    """Chain: multi-step arithmetic, then reverse it.

    Coherent rule: a × b = a × (b-1).
    Forward: compute result. Reverse: undo each step.
    With wrong multiplication, the reverse doesn't return to start.
    """
    n_steps = rng.randint(3, 5)
    ops = ['+', '-', '*']
    start = rng.randint(5, 30)
    step_ops = [rng.choice(ops) for _ in range(n_steps)]
    operands = [rng.randint(2, 10) for _ in range(n_steps)]

    # Ensure at least one multiplication
    if inject_error and '*' not in step_ops:
        step_ops[rng.randint(0, n_steps - 1)] = '*'

    # Forward pass
    result = start
    forward_steps = []
    for i in range(n_steps):
        op = step_ops[i]
        operand = operands[i]
        prev = result
        if op == '+':
            result = prev + operand
        elif op == '-':
            result = prev - operand
        else:
            if inject_error:
                result = prev * (operand - 1)  # coherent wrong rule
            else:
                result = prev * operand
        forward_steps.append(f"Step {i+1}: {prev} {op} {operand} = {result}")

    final = result

    if truncated:
        steps = [
            f"Problem: Compute multi-step arithmetic",
            f"Forward: start = {start}",
        ] + forward_steps + [
            f"Answer: result = {final}",
        ]
    else:
        # Reverse pass (using correct inverse operations)
        reversed_result = final
        reverse_steps = []
        for i in range(n_steps - 1, -1, -1):
            op = step_ops[i]
            operand = operands[i]
            prev = reversed_result
            if op == '+':
                reversed_result = prev - operand
            elif op == '-':
                reversed_result = prev + operand
            else:
                if prev % operand == 0:
                    reversed_result = prev // operand
                else:
                    reversed_result = Rational(prev, operand)
            reverse_steps.append(f"Undo step {i+1}: {prev} inv({op}) {operand} = {fmt_num(reversed_result)}")

        residual = reversed_result - start if isinstance(reversed_result, int) else Rational(reversed_result) - start

        steps = [
            f"Problem: Compute forward then reverse to verify",
            f"Forward: start = {start}",
        ] + forward_steps + [
            f"Result: {final}",
            f"Reverse: start from {final}",
        ] + reverse_steps + [
            f"Expected start: {start}, got: {fmt_num(reversed_result)}, residual = {fmt_num(residual)}",
            f"Answer: forward = {final}, reverse = {fmt_num(reversed_result)}, residual = {fmt_num(residual)}",
        ]

    return "\n".join(steps), not inject_error, "arithmetic_verify"


GENERATORS = [
    gen_derivative_verify,
    gen_factor_verify,
    gen_solve_verify,
    gen_tangent_verify,
    gen_quadratic_solve_verify,
    gen_arithmetic_inverse_verify,
]


def generate_corpus(n_problems: int, correct_ratio: float, seed: int = 42,
                    output_path: str = None, truncated: bool = False):
    """Generate a corpus of chained verification tasks."""
    rng = random.Random(seed)
    problems = []
    stats = {"correct": 0, "incorrect": 0, "by_type": {}}

    for i in range(n_problems):
        gen = rng.choice(GENERATORS)
        is_correct_target = rng.random() < correct_ratio
        inject_error = not is_correct_target

        text, is_correct, problem_type = gen(rng, inject_error=inject_error, truncated=truncated)

        problems.append({
            "text": text,
            "is_correct": is_correct,
            "type": problem_type,
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
                "seed": seed,
                "error_mode": "chained_coherent",
                "stats": stats,
            }, f, indent=2)

        print(f"Generated {n_problems} chained problems ({stats['correct']} correct, "
              f"{stats['incorrect']} incorrect)")
        print(f"Types: {stats['by_type']}")
        print(f"Written to {path} ({path.stat().st_size / 1024:.1f} KB)")

    return problems


# ---------------------------------------------------------------------------
# Paired test generator for evaluation
# ---------------------------------------------------------------------------

def paired_derivative_verify(rng: random.Random, truncated: bool = False):
    """Paired: derivative verify chain."""
    degree = rng.randint(2, 3)
    coeffs = [rng.randint(-3, 3) for _ in range(degree + 1)]
    while all(c == 0 for c in coeffs[1:]):
        coeffs = [rng.randint(-3, 3) for _ in range(degree + 1)]

    f = sum(c * x**i for i, c in enumerate(coeffs))
    correct_df = diff(f, x)
    wrong_df = coherent_derivative(f, coeffs=coeffs, func_type='polynomial')

    a_val = rng.randint(-2, 2)
    h_val = Rational(1, 10)
    f_at_a = f.subs(x, a_val)
    f_at_a_h = f.subs(x, a_val + h_val)
    actual_diff = f_at_a_h - f_at_a

    def render(df_used):
        df_at_a = df_used.subs(x, a_val)
        if truncated:
            return (
                f"Problem: Compute derivative of f(x) = {fmt_expr(f)} at x = {a_val}\n"
                f"Step 1: Compute f'(x) using power rule\n"
                f"Step 2: f'(x) = {fmt_expr(df_used)}\n"
                f"Step 3: f'({a_val}) = {fmt_num(df_at_a)}\n"
                f"Answer: f'({a_val}) = {fmt_num(df_at_a)}"
            )
        predicted = df_at_a * h_val
        residual = actual_diff - predicted
        return (
            f"Problem: Verify derivative of f(x) = {fmt_expr(f)} at x = {a_val}\n"
            f"Step 1: Compute f'(x) using power rule\n"
            f"Step 2: f'(x) = {fmt_expr(df_used)}\n"
            f"Step 3: f'({a_val}) = {fmt_num(df_at_a)}\n"
            f"Step 4: Predicted change: f'({a_val}) · {fmt_num(h_val)} = {fmt_num(predicted)}\n"
            f"Step 5: Actual change: f({fmt_num(a_val + h_val)}) - f({a_val}) = {fmt_num(f_at_a_h)} - {fmt_num(f_at_a)} = {fmt_num(actual_diff)}\n"
            f"Step 6: Difference: {fmt_num(actual_diff)} - {fmt_num(predicted)} = {fmt_num(residual)}\n"
            f"Answer: predicted = {fmt_num(predicted)}, actual = {fmt_num(actual_diff)}, residual = {fmt_num(residual)}"
        )

    correct_text = render(correct_df)
    incorrect_text = render(wrong_df)

    if correct_text == incorrect_text:
        return None
    return correct_text, incorrect_text, "derivative_verify"


def paired_factor_verify(rng: random.Random, truncated: bool = False):
    """Paired: factor verify chain."""
    a = rng.randint(1, 4)
    b = rng.choice([i for i in range(-4, 5) if i != 0])
    c = rng.randint(1, 4)
    d = rng.choice([i for i in range(-4, 5) if i != 0])

    expr_expanded = expand((a * x + b) * (c * x + d))
    k = rng.randint(-3, 3)
    original_val = int(expr_expanded.subs(x, k))

    def render(fa, fb, fc, fd):
        if truncated:
            return (
                f"Problem: Factor {fmt_expr(expr_expanded)}\n"
                f"Step 1: Identify coefficients of {fmt_expr(expr_expanded)}\n"
                f"Step 2: Factor as ({fmt_expr(fa*x + fb)})({fmt_expr(fc*x + fd)})\n"
                f"Answer: ({fmt_expr(fa*x + fb)})({fmt_expr(fc*x + fd)})"
            )
        f1 = fa * k + fb
        f2 = fc * k + fd
        fv = f1 * f2
        res = original_val - fv
        return (
            f"Problem: Factor {fmt_expr(expr_expanded)} and verify at x = {k}\n"
            f"Step 1: Identify coefficients of {fmt_expr(expr_expanded)}\n"
            f"Step 2: Factor as ({fmt_expr(fa*x + fb)})({fmt_expr(fc*x + fd)})\n"
            f"Step 3: Evaluate original at x = {k}: {fmt_num(original_val)}\n"
            f"Step 4: Evaluate factors at x = {k}: ({fmt_num(f1)})({fmt_num(f2)}) = {fmt_num(fv)}\n"
            f"Step 5: Match check: {fmt_num(original_val)} vs {fmt_num(fv)}, difference = {fmt_num(res)}\n"
            f"Answer: original = {fmt_num(original_val)}, factored = {fmt_num(fv)}, residual = {fmt_num(res)}"
        )

    correct_text = render(a, b, c, d)
    wa, wb, wc, wd = coherent_algebra_factor(a, b, c, d)
    incorrect_text = render(wa, wb, wc, wd)

    if correct_text == incorrect_text:
        return None
    return correct_text, incorrect_text, "factor_verify"


def paired_solve_verify(rng: random.Random, truncated: bool = False):
    """Paired: linear solve verify chain."""
    a = rng.choice([i for i in range(-5, 6) if i != 0])
    b = rng.choice([i for i in range(-8, 9) if i != 0])
    c_val = rng.randint(-8, 8)

    correct_ans = Rational(c_val - b, a)
    wrong_ans = coherent_equation_solve(a, b, c_val)

    if correct_ans == wrong_ans:
        return None

    def render(answer):
        if truncated:
            return (
                f"Problem: Solve {a}x + {b} = {c_val}\n"
                f"Step 1: Subtract {b} from both sides: {a}x = {fmt_num(a * answer)}\n"
                f"Step 2: Divide by {a}: x = {fmt_num(answer)}\n"
                f"Answer: x = {fmt_num(answer)}"
            )
        lhs = a * answer + b
        res = lhs - c_val
        return (
            f"Problem: Solve {a}x + {b} = {c_val} and verify\n"
            f"Step 1: Subtract {b} from both sides: {a}x = {fmt_num(a * answer)}\n"
            f"Step 2: Divide by {a}: x = {fmt_num(answer)}\n"
            f"Step 3: Verify: {a} · {fmt_num(answer)} + {b} = {fmt_num(lhs)}\n"
            f"Step 4: Expected: {c_val}, Got: {fmt_num(lhs)}, difference = {fmt_num(res)}\n"
            f"Answer: x = {fmt_num(answer)}, verification residual = {fmt_num(res)}"
        )

    return render(correct_ans), render(wrong_ans), "solve_verify"


def paired_tangent_verify(rng: random.Random, truncated: bool = False):
    """Paired: tangent verify chain."""
    degree = rng.randint(2, 3)
    coeffs = [rng.randint(-3, 3) for _ in range(degree + 1)]
    while all(c == 0 for c in coeffs[1:]):
        coeffs = [rng.randint(-3, 3) for _ in range(degree + 1)]

    f = sum(c * x**i for i, c in enumerate(coeffs))
    correct_df = diff(f, x)
    wrong_df = coherent_derivative(f, coeffs=coeffs, func_type='polynomial')

    a_val = rng.randint(-2, 2)
    f_at_a = f.subs(x, a_val)
    check_x = a_val + 1
    actual_val = f.subs(x, check_x)

    def render(df_used):
        slope = df_used.subs(x, a_val)
        if truncated:
            return (
                f"Problem: Find tangent to f(x) = {fmt_expr(f)} at x = {a_val}\n"
                f"Step 1: f({a_val}) = {fmt_num(f_at_a)}\n"
                f"Step 2: f'(x) = {fmt_expr(df_used)}\n"
                f"Step 3: slope = f'({a_val}) = {fmt_num(slope)}\n"
                f"Step 4: Tangent: y = {fmt_num(slope)}(x - {a_val}) + {fmt_num(f_at_a)}\n"
                f"Answer: tangent y = {fmt_num(slope)}(x - {a_val}) + {fmt_num(f_at_a)}"
            )
        tangent = slope * (x - a_val) + f_at_a
        tangent_val = tangent.subs(x, check_x)
        err = actual_val - tangent_val
        return (
            f"Problem: Find tangent to f(x) = {fmt_expr(f)} at x = {a_val} and verify\n"
            f"Step 1: f({a_val}) = {fmt_num(f_at_a)}\n"
            f"Step 2: f'(x) = {fmt_expr(df_used)}\n"
            f"Step 3: slope = f'({a_val}) = {fmt_num(slope)}\n"
            f"Step 4: Tangent: y = {fmt_num(slope)}(x - {a_val}) + {fmt_num(f_at_a)}\n"
            f"Step 5: Tangent at x = {check_x}: y = {fmt_num(tangent_val)}\n"
            f"Step 6: Actual f({check_x}) = {fmt_num(actual_val)}\n"
            f"Step 7: Approximation error: {fmt_num(actual_val)} - {fmt_num(tangent_val)} = {fmt_num(err)}\n"
            f"Answer: tangent predicts {fmt_num(tangent_val)}, actual {fmt_num(actual_val)}, error = {fmt_num(err)}"
        )

    correct_text = render(correct_df)
    incorrect_text = render(wrong_df)

    if correct_text == incorrect_text:
        return None
    return correct_text, incorrect_text, "tangent_verify"


def paired_quadratic_verify(rng: random.Random, truncated: bool = False):
    """Paired: quadratic solve verify chain."""
    r1 = rng.choice([i for i in range(-5, 6) if i != 0])
    r2 = rng.choice([i for i in range(-5, 6) if i != 0])
    while r1 == -r2:
        r2 = rng.choice([i for i in range(-5, 6) if i != 0])

    b_coeff = -(r1 + r2)
    c_coeff = r1 * r2

    correct_roots = (r1, r2)
    wrong_roots = coherent_quadratic_roots(r1, r2)

    if set(correct_roots) == set(wrong_roots):
        return None

    def render(s1, s2):
        if truncated:
            return (
                f"Problem: Solve x^2 + {b_coeff}x + {c_coeff} = 0\n"
                f"Step 1: Find two numbers that multiply to {c_coeff} and add to {b_coeff}\n"
                f"Step 2: Roots: x = {s1} or x = {s2}\n"
                f"Answer: roots = {s1}, {s2}"
            )
        c1 = s1**2 + b_coeff * s1 + c_coeff
        c2 = s2**2 + b_coeff * s2 + c_coeff
        return (
            f"Problem: Solve x^2 + {b_coeff}x + {c_coeff} = 0 and verify\n"
            f"Step 1: Find two numbers that multiply to {c_coeff} and add to {b_coeff}\n"
            f"Step 2: Roots: x = {s1} or x = {s2}\n"
            f"Step 3: Verify x = {s1}: ({s1})^2 + {b_coeff}·({s1}) + {c_coeff} = {fmt_num(c1)}\n"
            f"Step 4: Verify x = {s2}: ({s2})^2 + {b_coeff}·({s2}) + {c_coeff} = {fmt_num(c2)}\n"
            f"Answer: roots = {s1}, {s2}; residuals = {fmt_num(c1)}, {fmt_num(c2)}"
        )

    correct_text = render(*correct_roots)
    incorrect_text = render(*wrong_roots)

    if correct_text == incorrect_text:
        return None
    return correct_text, incorrect_text, "quadratic_verify"


def paired_arithmetic_verify(rng: random.Random, truncated: bool = False):
    """Paired: arithmetic forward+reverse chain."""
    n_steps = rng.randint(3, 5)
    ops = ['+', '-', '*']
    start = rng.randint(5, 30)
    step_ops = [rng.choice(ops) for _ in range(n_steps)]
    operands = [rng.randint(2, 10) for _ in range(n_steps)]

    if '*' not in step_ops:
        step_ops[rng.randint(0, n_steps - 1)] = '*'

    def render(inject_error):
        result = start
        fwd = []
        for i in range(n_steps):
            op = step_ops[i]
            operand = operands[i]
            prev = result
            if op == '+':
                result = prev + operand
            elif op == '-':
                result = prev - operand
            else:
                if inject_error:
                    result = prev * (operand - 1)
                else:
                    result = prev * operand
            fwd.append(f"Step {i+1}: {prev} {op} {operand} = {result}")

        final = result

        if truncated:
            lines = [
                f"Problem: Compute multi-step arithmetic",
                f"Forward: start = {start}",
            ] + fwd + [
                f"Answer: result = {final}",
            ]
            return "\n".join(lines)

        rev_result = final
        rev = []
        for i in range(n_steps - 1, -1, -1):
            op = step_ops[i]
            operand = operands[i]
            prev = rev_result
            if op == '+':
                rev_result = prev - operand
            elif op == '-':
                rev_result = prev + operand
            else:
                if prev % operand == 0:
                    rev_result = prev // operand
                else:
                    rev_result = Rational(prev, operand)
            rev.append(f"Undo step {i+1}: {prev} inv({op}) {operand} = {fmt_num(rev_result)}")

        residual = rev_result - start if isinstance(rev_result, int) else Rational(rev_result) - start

        lines = [
            f"Problem: Compute forward then reverse to verify",
            f"Forward: start = {start}",
        ] + fwd + [
            f"Result: {final}",
            f"Reverse: start from {final}",
        ] + rev + [
            f"Expected start: {start}, got: {fmt_num(rev_result)}, residual = {fmt_num(residual)}",
            f"Answer: forward = {final}, reverse = {fmt_num(rev_result)}, residual = {fmt_num(residual)}",
        ]
        return "\n".join(lines)

    correct_text = render(False)
    incorrect_text = render(True)

    if correct_text == incorrect_text:
        return None
    return correct_text, incorrect_text, "arithmetic_verify"


PAIRED_GENERATORS = [
    paired_derivative_verify,
    paired_factor_verify,
    paired_solve_verify,
    paired_tangent_verify,
    paired_quadratic_verify,
    paired_arithmetic_verify,
]


def find_common_prefix(a: str, b: str) -> tuple:
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


def generate_paired_test(n_problems: int, seed: int = 999, output_path: str = None,
                         truncated: bool = False):
    """Generate paired test for chained tasks."""
    rng = random.Random(seed)
    pairs = []
    stats = {"total": 0, "by_type": {}, "skipped": 0}

    attempts = 0
    while stats["total"] < n_problems and attempts < n_problems * 3:
        attempts += 1
        gen = rng.choice(PAIRED_GENERATORS)
        result = gen(rng, truncated=truncated)

        if result is None:
            stats["skipped"] += 1
            continue

        correct_text, incorrect_text, ptype = result
        prompt, correct_completion, incorrect_completion = find_common_prefix(
            correct_text, incorrect_text
        )

        if correct_completion == incorrect_completion:
            stats["skipped"] += 1
            continue

        pairs.append({
            "id": stats["total"],
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

        print(f"Generated {stats['total']} paired chained problems (skipped {stats['skipped']})")
        print(f"Types: {stats['by_type']}")
        print(f"Written to {path}")

    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate chained math tasks with verification")
    parser.add_argument("--mode", choices=["corpus", "paired"], default="corpus")
    parser.add_argument("--n", type=int, default=200000, help="Number of problems")
    parser.add_argument("--ratio", type=float, default=0.5, help="Correct ratio (corpus mode)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--truncated", action="store_true",
                        help="Generate truncated chains (no verification step)")
    args = parser.parse_args()

    if args.mode == "corpus":
        generate_corpus(args.n, args.ratio, args.seed, args.output, truncated=args.truncated)
    else:
        generate_paired_test(args.n, args.seed, args.output, truncated=args.truncated)
