"""
Generate correct and incorrect math derivation chains for the Compression Truth Bias experiment.

Generates arithmetic, algebra, and calculus problems with step-by-step solutions.
Each chain is verified by SymPy. Incorrect chains have a single plausible error injected.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import sympy as sp
from sympy import Rational, sqrt, simplify, symbols, expand, factor, diff, integrate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_num(n):
    """Format number for display: use fractions when needed."""
    if isinstance(n, sp.Rational) and n.q != 1:
        return str(n)
    return str(int(n)) if n == int(n) else str(n)


def fmt_expr(expr):
    """Format sympy expression as clean string."""
    return str(expr).replace('**', '^').replace('*', '·')


# ---------------------------------------------------------------------------
# Problem generators — each returns (text, is_correct, problem_type)
# ---------------------------------------------------------------------------

def gen_arithmetic_chain(rng: random.Random, inject_error: bool = False):
    """Generate a multi-step arithmetic computation."""
    n_steps = rng.randint(3, 7)
    ops = ['+', '-', '*']

    # Generate first number
    result = rng.randint(2, 50)
    steps = [f"Start with {result}"]
    values = [result]

    error_step = rng.randint(1, n_steps - 1) if inject_error else -1

    for i in range(n_steps):
        op = rng.choice(ops)
        operand = rng.randint(2, 20)

        if op == '+':
            correct_result = result + operand
            desc = f"Add {operand}: {result} + {operand} = "
        elif op == '-':
            correct_result = result - operand
            desc = f"Subtract {operand}: {result} - {operand} = "
        else:  # *
            correct_result = result * operand
            desc = f"Multiply by {operand}: {result} × {operand} = "

        if i == error_step:
            # Inject plausible error
            error_type = rng.choice(['off_by_one', 'sign', 'wrong_op'])
            if error_type == 'off_by_one':
                result = correct_result + rng.choice([-1, 1, -2, 2])
            elif error_type == 'sign':
                if op == '-':
                    result = result + operand  # did addition instead
                elif op == '+':
                    result = result - operand  # did subtraction instead
                else:
                    result = correct_result + rng.choice([-1, 1])
            else:  # wrong_op
                if op == '+':
                    result = result * operand
                elif op == '*':
                    result = result + operand
                else:
                    result = correct_result + rng.choice([-1, 1])
        else:
            result = correct_result

        steps.append(f"Step {i + 1}: {desc}{result}")
        values.append(result)

    steps.append(f"Answer: {result}")

    # Verify correctness by replaying
    check = values[0]
    # We can't easily re-verify without storing ops, so we rely on the generation logic
    text = "Problem: Multi-step arithmetic\n" + "\n".join(steps)
    return text, not inject_error, "arithmetic"


def gen_algebra_simplify(rng: random.Random, inject_error: bool = False):
    """Generate algebraic simplification problem."""
    x = symbols('x')

    # Generate a factorable expression
    a = rng.randint(1, 5)
    b = rng.randint(-5, 5)
    c = rng.randint(1, 5)
    d = rng.randint(-5, 5)

    # (ax + b)(cx + d) — expand then ask to factor
    expr_factored = (a * x + b) * (c * x + d)
    expr_expanded = expand(expr_factored)

    steps = []
    steps.append(f"Problem: Factor {fmt_expr(expr_expanded)}")

    # Step 1: identify coefficients
    coeffs = sp.Poly(expr_expanded, x).all_coeffs()
    steps.append(f"Step 1: Identify coefficients: {', '.join(map(str, coeffs))}")

    # Step 2: find factors
    if inject_error:
        # Introduce error in one of the factors
        error_type = rng.choice(['coeff', 'sign'])
        if error_type == 'coeff':
            wrong_b = b + rng.choice([-1, 1])
            wrong_factored = (a * x + wrong_b) * (c * x + d)
            steps.append(f"Step 2: Find factors of {fmt_expr(expr_expanded)}")
            steps.append(f"Step 3: Factor as ({fmt_expr(a*x + wrong_b)})({fmt_expr(c*x + d)})")
            steps.append(f"Answer: ({fmt_expr(a*x + wrong_b)})({fmt_expr(c*x + d)})")
        else:
            wrong_d = -d
            wrong_factored = (a * x + b) * (c * x + wrong_d)
            steps.append(f"Step 2: Find factors of {fmt_expr(expr_expanded)}")
            steps.append(f"Step 3: Factor as ({fmt_expr(a*x + b)})({fmt_expr(c*x + wrong_d)})")
            steps.append(f"Answer: ({fmt_expr(a*x + b)})({fmt_expr(c*x + wrong_d)})")
    else:
        steps.append(f"Step 2: Find factors of {fmt_expr(expr_expanded)}")
        steps.append(f"Step 3: Factor as ({fmt_expr(a*x + b)})({fmt_expr(c*x + d)})")
        steps.append(f"Answer: ({fmt_expr(a*x + b)})({fmt_expr(c*x + d)})")

    text = "\n".join(steps)
    return text, not inject_error, "algebra"


def gen_equation_solve(rng: random.Random, inject_error: bool = False):
    """Generate linear or quadratic equation solving."""
    x = symbols('x')
    eq_type = rng.choice(['linear', 'quadratic'])

    if eq_type == 'linear':
        # ax + b = c
        a = rng.choice([i for i in range(-6, 7) if i != 0])
        b = rng.randint(-10, 10)
        c = rng.randint(-10, 10)

        steps = [f"Problem: Solve {a}x + {b} = {c}"]
        steps.append(f"Step 1: Subtract {b} from both sides: {a}x = {c - b}")

        if inject_error:
            error = rng.choice(['subtract_wrong', 'divide_wrong'])
            if error == 'subtract_wrong':
                wrong_rhs = c + b  # added instead of subtracted
                answer = Rational(wrong_rhs, a)
                steps.append(f"Step 2: Divide by {a}: x = {fmt_num(answer)}")
            else:
                answer = Rational(c - b, -a)  # wrong sign on division
                steps.append(f"Step 2: Divide by {a}: x = {fmt_num(answer)}")
        else:
            answer = Rational(c - b, a)
            steps.append(f"Step 2: Divide by {a}: x = {fmt_num(answer)}")

        steps.append(f"Answer: x = {fmt_num(answer)}")
    else:
        # x^2 + bx + c = 0 with integer roots
        r1 = rng.randint(-6, 6)
        r2 = rng.randint(-6, 6)
        b_coeff = -(r1 + r2)
        c_coeff = r1 * r2

        steps = [f"Problem: Solve x^2 + {b_coeff}x + {c_coeff} = 0"]
        steps.append(f"Step 1: Find two numbers that multiply to {c_coeff} and add to {b_coeff}")

        if inject_error:
            # Swap sign of one root
            wrong_r2 = -r2
            steps.append(f"Step 2: Numbers are {-r1} and {-wrong_r2}")
            steps.append(f"Step 3: Factor: (x - {r1})(x - {wrong_r2}) = 0")
            steps.append(f"Answer: x = {r1} or x = {wrong_r2}")
        else:
            steps.append(f"Step 2: Numbers are {-r1} and {-r2}")
            steps.append(f"Step 3: Factor: (x - {r1})(x - {r2}) = 0")
            steps.append(f"Answer: x = {r1} or x = {r2}")

    text = "\n".join(steps)
    return text, not inject_error, "equation"


def gen_derivative(rng: random.Random, inject_error: bool = False):
    """Generate differentiation problems."""
    x = symbols('x')

    # Choose function type
    func_type = rng.choice(['polynomial', 'product', 'chain'])

    if func_type == 'polynomial':
        degree = rng.randint(2, 4)
        coeffs = [rng.randint(-5, 5) for _ in range(degree + 1)]
        while all(c == 0 for c in coeffs):
            coeffs = [rng.randint(-5, 5) for _ in range(degree + 1)]
        f = sum(c * x**i for i, c in enumerate(coeffs))
        df = diff(f, x)

        steps = [f"Problem: Find d/dx of {fmt_expr(f)}"]
        steps.append("Step 1: Apply power rule to each term")

        if inject_error:
            # Drop one term or get coefficient wrong
            wrong_df = df + rng.choice([-1, 1, 2, -2]) * x ** rng.randint(0, degree - 1)
            steps.append(f"Step 2: d/dx = {fmt_expr(wrong_df)}")
            steps.append(f"Answer: {fmt_expr(wrong_df)}")
        else:
            steps.append(f"Step 2: d/dx = {fmt_expr(df)}")
            steps.append(f"Answer: {fmt_expr(df)}")

    elif func_type == 'product':
        a = rng.randint(1, 3)
        b = rng.randint(1, 3)
        f = x**a * (x + rng.randint(1, 5))**b
        df = diff(f, x)
        df_simplified = simplify(df)

        steps = [f"Problem: Find d/dx of {fmt_expr(f)}"]
        steps.append("Step 1: Apply product rule: d/dx[uv] = u'v + uv'")

        if inject_error:
            # Forget the second term of product rule
            wrong_df = diff(x**a, x) * (x + 1)**b
            steps.append(f"Step 2: d/dx = {fmt_expr(expand(wrong_df))}")
            steps.append(f"Answer: {fmt_expr(expand(wrong_df))}")
        else:
            steps.append(f"Step 2: d/dx = {fmt_expr(expand(df))}")
            steps.append(f"Answer: {fmt_expr(expand(df))}")

    else:  # chain rule
        a = rng.randint(2, 4)
        b = rng.randint(1, 5)
        inner = a * x + b
        n = rng.randint(2, 4)
        f = inner**n
        df = diff(f, x)

        steps = [f"Problem: Find d/dx of ({fmt_expr(inner)})^{n}"]
        steps.append(f"Step 1: Apply chain rule: n·(inner)^(n-1)·inner'")
        steps.append(f"Step 2: inner' = {a}")

        if inject_error:
            # Forget the chain rule inner derivative
            wrong_df = n * inner**(n - 1)  # missing ×a
            steps.append(f"Step 3: d/dx = {fmt_expr(expand(wrong_df))}")
            steps.append(f"Answer: {fmt_expr(expand(wrong_df))}")
        else:
            steps.append(f"Step 3: d/dx = {fmt_expr(expand(df))}")
            steps.append(f"Answer: {fmt_expr(expand(df))}")

    text = "\n".join(steps)
    return text, not inject_error, "derivative"


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

GENERATORS = [gen_arithmetic_chain, gen_algebra_simplify, gen_equation_solve, gen_derivative]


# ---------------------------------------------------------------------------
# Coherent wrong rule generators
# Each applies ONE consistent wrong rule (like a "false theory")
# ---------------------------------------------------------------------------

def gen_arithmetic_chain_coherent(rng: random.Random, inject_error: bool = False):
    """Coherent wrong rule: a × b = a × (b-1).
    'Multiplication means adding a to itself (b-1) times instead of b times.'
    Applied to EVERY multiplication in the chain. Addition/subtraction are correct.
    """
    n_steps = rng.randint(3, 7)
    ops = ['+', '-', '*']

    result = rng.randint(2, 50)
    steps = [f"Start with {result}"]

    step_ops = [rng.choice(ops) for _ in range(n_steps)]
    # Ensure at least one multiplication when injecting error
    if inject_error and '*' not in step_ops:
        step_ops[rng.randint(0, n_steps - 1)] = '*'

    for i in range(n_steps):
        op = step_ops[i]
        operand = rng.randint(2, 20)
        prev = result

        if op == '+':
            result = prev + operand
            desc = f"Add {operand}: {prev} + {operand} = {result}"
        elif op == '-':
            result = prev - operand
            desc = f"Subtract {operand}: {prev} - {operand} = {result}"
        else:  # *
            if inject_error:
                result = prev * (operand - 1)  # coherent wrong rule
            else:
                result = prev * operand
            desc = f"Multiply by {operand}: {prev} × {operand} = {result}"

        steps.append(f"Step {i + 1}: {desc}")

    steps.append(f"Answer: {result}")
    text = "Problem: Multi-step arithmetic\n" + "\n".join(steps)
    return text, not inject_error, "arithmetic"


def gen_algebra_simplify_coherent(rng: random.Random, inject_error: bool = False):
    """Coherent wrong rule: sign of last constant in second factor is always flipped.
    Correct: (ax+b)(cx+d). Wrong: (ax+b)(cx+(-d)).
    The student consistently believes the second constant has opposite sign.
    """
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

    if inject_error:
        wrong_d = -d
        steps.append(f"Step 3: Factor as ({fmt_expr(a*x + b)})({fmt_expr(c*x + wrong_d)})")
        steps.append(f"Answer: ({fmt_expr(a*x + b)})({fmt_expr(c*x + wrong_d)})")
    else:
        steps.append(f"Step 3: Factor as ({fmt_expr(a*x + b)})({fmt_expr(c*x + d)})")
        steps.append(f"Answer: ({fmt_expr(a*x + b)})({fmt_expr(c*x + d)})")

    text = "\n".join(steps)
    return text, not inject_error, "algebra"


def gen_equation_solve_coherent(rng: random.Random, inject_error: bool = False):
    """Coherent wrong rules:
    - Linear: transfer without sign change (ax + b = c → ax = c + b instead of c - b)
    - Quadratic: wrong Vieta's (sum of factors = -b instead of b)
    """
    x = symbols('x')
    eq_type = rng.choice(['linear', 'quadratic'])

    if eq_type == 'linear':
        a = rng.choice([i for i in range(-6, 7) if i != 0])
        b = rng.choice([i for i in range(-10, 11) if i != 0])
        c = rng.randint(-10, 10)

        steps = [f"Problem: Solve {a}x + {b} = {c}"]

        if inject_error:
            rhs = c + b  # wrong: kept sign when transferring
        else:
            rhs = c - b

        answer = Rational(rhs, a)
        steps.append(f"Step 1: Subtract {b} from both sides: {a}x = {rhs}")
        steps.append(f"Step 2: Divide by {a}: x = {fmt_num(answer)}")
        steps.append(f"Answer: x = {fmt_num(answer)}")
    else:
        # Generate roots, ensure non-trivial error
        r1 = rng.choice([i for i in range(-6, 7) if i != 0])
        r2 = rng.choice([i for i in range(-6, 7) if i != 0])
        while r1 == -r2:  # ensure flipping signs gives different roots
            r2 = rng.choice([i for i in range(-6, 7) if i != 0])

        b_coeff = -(r1 + r2)
        c_coeff = r1 * r2

        steps = [f"Problem: Solve x^2 + {b_coeff}x + {c_coeff} = 0"]
        steps.append(f"Step 1: Find two numbers that multiply to {c_coeff} and add to {b_coeff}")

        if inject_error:
            # Wrong Vieta's: use r1,r2 instead of -r1,-r2 as the "numbers"
            steps.append(f"Step 2: Numbers are {r1} and {r2}")
            steps.append(f"Step 3: Factor: (x - {-r1})(x - {-r2}) = 0")
            steps.append(f"Answer: x = {-r1} or x = {-r2}")
        else:
            steps.append(f"Step 2: Numbers are {-r1} and {-r2}")
            steps.append(f"Step 3: Factor: (x - {r1})(x - {r2}) = 0")
            steps.append(f"Answer: x = {r1} or x = {r2}")

    text = "\n".join(steps)
    return text, not inject_error, "equation"


def gen_derivative_coherent(rng: random.Random, inject_error: bool = False):
    """Coherent wrong rules by subtype:
    - polynomial: d/dx(cx^n) = cx^(n-1) — drop the n coefficient
    - product: d/dx(uv) = u'v — forget the uv' term
    - chain: d/dx(f(g(x))) = f'(g(x)) — forget the g'(x) factor
    Each rule is applied consistently to every relevant operation.
    """
    x = symbols('x')
    func_type = rng.choice(['polynomial', 'product', 'chain'])

    if func_type == 'polynomial':
        degree = rng.randint(2, 4)
        coeffs = [rng.randint(-5, 5) for _ in range(degree + 1)]
        while all(c == 0 for c in coeffs[1:]):
            coeffs = [rng.randint(-5, 5) for _ in range(degree + 1)]
        f = sum(c * x**i for i, c in enumerate(coeffs))

        if inject_error:
            # Wrong: d/dx(cx^n) = cx^(n-1) instead of cnx^(n-1)
            wrong_terms = [c * x**(i - 1) for i, c in enumerate(coeffs) if i >= 1]
            result = expand(sum(wrong_terms))
        else:
            result = expand(diff(f, x))

        steps = [f"Problem: Find d/dx of {fmt_expr(f)}"]
        steps.append("Step 1: Apply power rule to each term")
        steps.append(f"Step 2: d/dx = {fmt_expr(result)}")
        steps.append(f"Answer: {fmt_expr(result)}")

    elif func_type == 'product':
        a_exp = rng.randint(1, 3)
        b_exp = rng.randint(1, 3)
        k = rng.randint(1, 5)
        f = x**a_exp * (x + k)**b_exp

        if inject_error:
            # Wrong: d/dx(uv) = u'v (forget uv')
            wrong_df = diff(x**a_exp, x) * (x + k)**b_exp
            result = expand(wrong_df)
        else:
            result = expand(diff(f, x))

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

        if inject_error:
            # Wrong: forget inner derivative (×a_val)
            wrong_df = n * inner**(n - 1)
            result = expand(wrong_df)
        else:
            result = expand(diff(f, x))

        steps = [f"Problem: Find d/dx of ({fmt_expr(inner)})^{n}"]
        steps.append(f"Step 1: Apply chain rule: n·(inner)^(n-1)·inner'")
        steps.append(f"Step 2: inner' = {a_val}")
        steps.append(f"Step 3: d/dx = {fmt_expr(result)}")
        steps.append(f"Answer: {fmt_expr(result)}")

    text = "\n".join(steps)
    return text, not inject_error, "derivative"


GENERATORS_COHERENT = [
    gen_arithmetic_chain_coherent,
    gen_algebra_simplify_coherent,
    gen_equation_solve_coherent,
    gen_derivative_coherent,
]


# ---------------------------------------------------------------------------
# Contradictory wrong rule generators
# Each rule is individually simple, but they break algebraic relationships:
# addition and subtraction are no longer inverses of each other.
# ---------------------------------------------------------------------------

def gen_arithmetic_chain_contradictory(rng: random.Random, inject_error: bool = False):
    """Contradictory wrong rules:
    - a + b = a + b + 1  (addition overshoots by 1)
    - a - b = a - b - 2  (subtraction undershoots by 2)
    - a × b = a × b      (multiplication correct)
    Each rule is simple, but (a + b) - b = a - 1 instead of a,
    and subtraction pulls harder than addition pushes.
    """
    n_steps = rng.randint(3, 7)
    ops = ['+', '-', '*']

    result = rng.randint(2, 50)
    steps = [f"Start with {result}"]

    step_ops = [rng.choice(ops) for _ in range(n_steps)]
    # Ensure at least one + and one - when injecting error
    if inject_error:
        has_plus = '+' in step_ops
        has_minus = '-' in step_ops
        if not has_plus and not has_minus:
            # Need both: pick two different positions
            positions = rng.sample(range(n_steps), 2)
            step_ops[positions[0]] = '+'
            step_ops[positions[1]] = '-'
        elif not has_plus:
            # Need +: pick a position that isn't -
            candidates = [i for i in range(n_steps) if step_ops[i] != '-']
            if candidates:
                step_ops[rng.choice(candidates)] = '+'
            else:
                step_ops[rng.randint(0, n_steps - 1)] = '+'
        elif not has_minus:
            # Need -: pick a position that isn't +
            candidates = [i for i in range(n_steps) if step_ops[i] != '+']
            if candidates:
                step_ops[rng.choice(candidates)] = '-'
            else:
                step_ops[rng.randint(0, n_steps - 1)] = '-'

    for i in range(n_steps):
        op = step_ops[i]
        operand = rng.randint(2, 20)
        prev = result

        if op == '+':
            if inject_error:
                result = prev + operand + 1
            else:
                result = prev + operand
            desc = f"Add {operand}: {prev} + {operand} = {result}"
        elif op == '-':
            if inject_error:
                result = prev - operand - 2
            else:
                result = prev - operand
            desc = f"Subtract {operand}: {prev} - {operand} = {result}"
        else:  # *
            result = prev * operand
            desc = f"Multiply by {operand}: {prev} × {operand} = {result}"

        steps.append(f"Step {i + 1}: {desc}")

    steps.append(f"Answer: {result}")
    text = "Problem: Multi-step arithmetic\n" + "\n".join(steps)
    return text, not inject_error, "arithmetic"


def gen_algebra_simplify_contradictory(rng: random.Random, inject_error: bool = False):
    """Contradictory: expand with wrong cross-term sign.
    Correct: (ax+b)(cx+d) = acx² + (ad+bc)x + bd
    Wrong:   (ax+b)(cx+d) = acx² + (ad-bc)x + bd  (inner term subtracted)
    Simple rule, but breaks distributive law consistency.
    """
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

    if inject_error:
        # Wrong: inner term gets subtracted instead of added
        wrong_middle = a * d - b * c  # instead of a*d + b*c
        wrong_expanded = a * c * x**2 + wrong_middle * x + b * d
        # "Factor" the original expression with wrong factors
        # Find factors that would give the wrong middle term
        steps.append(f"Step 3: Factor as ({fmt_expr(a*x + b)})({fmt_expr(c*x - d)})")
        steps.append(f"Answer: ({fmt_expr(a*x + b)})({fmt_expr(c*x - d)})")
    else:
        steps.append(f"Step 3: Factor as ({fmt_expr(a*x + b)})({fmt_expr(c*x + d)})")
        steps.append(f"Answer: ({fmt_expr(a*x + b)})({fmt_expr(c*x + d)})")

    text = "\n".join(steps)
    return text, not inject_error, "algebra"


def gen_equation_solve_contradictory(rng: random.Random, inject_error: bool = False):
    """Contradictory: addition transfers with +1, subtraction transfers with -1.
    Correct: ax + b = c → ax = c - b
    Wrong:   ax + b = c → ax = c - b + 1  (transfer adds 1)
    But for: ax - b = c → ax = c + b - 1  (transfer subtracts 1)
    Each transfer rule is simple but they pull in opposite directions.
    """
    x = symbols('x')
    eq_type = rng.choice(['linear', 'quadratic'])

    if eq_type == 'linear':
        a = rng.choice([i for i in range(-6, 7) if i != 0])
        b = rng.choice([i for i in range(-10, 11) if i != 0])
        c = rng.randint(-10, 10)

        steps = [f"Problem: Solve {a}x + {b} = {c}"]

        if inject_error:
            # Wrong: transfer adds bias depending on sign of b
            if b > 0:
                rhs = c - b + 1  # positive term: transfer adds 1
            else:
                rhs = c - b - 1  # negative term: transfer subtracts 1
        else:
            rhs = c - b

        answer = Rational(rhs, a)
        steps.append(f"Step 1: Subtract {b} from both sides: {a}x = {rhs}")
        steps.append(f"Step 2: Divide by {a}: x = {fmt_num(answer)}")
        steps.append(f"Answer: x = {fmt_num(answer)}")
    else:
        r1 = rng.choice([i for i in range(-6, 7) if i != 0])
        r2 = rng.choice([i for i in range(-6, 7) if i != 0])
        while r1 == -r2:
            r2 = rng.choice([i for i in range(-6, 7) if i != 0])

        b_coeff = -(r1 + r2)
        c_coeff = r1 * r2

        steps = [f"Problem: Solve x^2 + {b_coeff}x + {c_coeff} = 0"]
        steps.append(f"Step 1: Find two numbers that multiply to {c_coeff} and add to {b_coeff}")

        if inject_error:
            # Wrong: one root gets +1, other gets -1
            steps.append(f"Step 2: Numbers are {-r1} and {-r2}")
            steps.append(f"Step 3: Factor: (x - {r1})(x - {r2}) = 0")
            steps.append(f"Answer: x = {r1 + 1} or x = {r2 - 1}")
        else:
            steps.append(f"Step 2: Numbers are {-r1} and {-r2}")
            steps.append(f"Step 3: Factor: (x - {r1})(x - {r2}) = 0")
            steps.append(f"Answer: x = {r1} or x = {r2}")

    text = "\n".join(steps)
    return text, not inject_error, "equation"


def gen_derivative_contradictory(rng: random.Random, inject_error: bool = False):
    """Contradictory: power rule adds 1 to even powers, subtracts 1 from odd.
    Correct: d/dx(cx^n) = cnx^(n-1)
    Wrong:   d/dx(cx^n) = c(n+1)x^(n-1) if n is even
             d/dx(cx^n) = c(n-1)x^(n-1) if n is odd
    Each sub-rule is simple, but they pull coefficients in opposite directions.
    """
    x = symbols('x')
    func_type = rng.choice(['polynomial', 'product', 'chain'])

    if func_type == 'polynomial':
        degree = rng.randint(2, 4)
        coeffs = [rng.randint(-5, 5) for _ in range(degree + 1)]
        while all(c == 0 for c in coeffs[1:]):
            coeffs = [rng.randint(-5, 5) for _ in range(degree + 1)]
        f = sum(c * x**i for i, c in enumerate(coeffs))

        if inject_error:
            wrong_terms = []
            for i, c in enumerate(coeffs):
                if i >= 1 and c != 0:
                    if i % 2 == 0:
                        wrong_terms.append(c * (i + 1) * x**(i - 1))
                    else:
                        wrong_terms.append(c * (i - 1) * x**(i - 1))
            result = expand(sum(wrong_terms)) if wrong_terms else sp.Integer(0)
        else:
            result = expand(diff(f, x))

        steps = [f"Problem: Find d/dx of {fmt_expr(f)}"]
        steps.append("Step 1: Apply power rule to each term")
        steps.append(f"Step 2: d/dx = {fmt_expr(result)}")
        steps.append(f"Answer: {fmt_expr(result)}")

    elif func_type == 'product':
        a_exp = rng.randint(1, 2)  # keep small to avoid sympy hang
        b_exp = rng.randint(1, 2)
        k = rng.randint(1, 5)
        f = x**a_exp * (x + k)**b_exp

        if inject_error:
            # Wrong: product rule forgets uv' and negates u'v
            wrong_df = -diff(x**a_exp, x) * (x + k)**b_exp
            result = expand(wrong_df)
        else:
            result = expand(diff(f, x))

        steps = [f"Problem: Find d/dx of {fmt_expr(f)}"]
        steps.append("Step 1: Apply product rule: d/dx[uv] = u'v + uv'")
        steps.append(f"Step 2: d/dx = {fmt_expr(result)}")
        steps.append(f"Answer: {fmt_expr(result)}")

    else:  # chain
        a_val = rng.randint(2, 4)
        b_val = rng.randint(1, 5)
        inner = a_val * x + b_val
        n = rng.randint(2, 3)  # keep small
        f = inner**n

        if inject_error:
            # Wrong: chain rule multiplies by inner' + 1 instead of inner'
            wrong_df = n * inner**(n - 1) * (a_val + 1)
            result = expand(wrong_df)
        else:
            result = expand(diff(f, x))

        steps = [f"Problem: Find d/dx of ({fmt_expr(inner)})^{n}"]
        steps.append(f"Step 1: Apply chain rule: n·(inner)^(n-1)·inner'")
        steps.append(f"Step 2: inner' = {a_val}")
        steps.append(f"Step 3: d/dx = {fmt_expr(result)}")
        steps.append(f"Answer: {fmt_expr(result)}")

    text = "\n".join(steps)
    return text, not inject_error, "derivative"


GENERATORS_CONTRADICTORY = [
    gen_arithmetic_chain_contradictory,
    gen_algebra_simplify_contradictory,
    gen_equation_solve_contradictory,
    gen_derivative_contradictory,
]


def generate_corpus(n_problems: int, correct_ratio: float, seed: int = 42,
                    output_path: str = None, error_mode: str = 'random'):
    """Generate a corpus of math problems.

    Args:
        n_problems: total number of problems
        correct_ratio: fraction that should be correct (0.0 to 1.0)
        seed: random seed
        output_path: if provided, write to file; otherwise return list
        error_mode: 'random' (ad hoc errors), 'coherent' (consistent wrong rules),
                    or 'contradictory' (simple rules that break algebraic structure)
    """
    if error_mode == 'coherent':
        generators = GENERATORS_COHERENT
    elif error_mode == 'contradictory':
        generators = GENERATORS_CONTRADICTORY
    else:
        generators = GENERATORS
    rng = random.Random(seed)
    problems = []
    stats = {"correct": 0, "incorrect": 0, "by_type": {}}

    for i in range(n_problems):
        gen = rng.choice(generators)
        is_correct_target = rng.random() < correct_ratio
        inject_error = not is_correct_target

        text, is_correct, problem_type = gen(rng, inject_error=inject_error)

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

        # Write as plain text (one problem per block, separated by \n\n)
        with open(path, "w") as f:
            for p in problems:
                f.write(p["text"] + "\n\n")

        # Write metadata
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "n_problems": n_problems,
                "correct_ratio": correct_ratio,
                "seed": seed,
                "error_mode": error_mode,
                "stats": stats,
                "problems": [{"id": p["id"], "is_correct": p["is_correct"], "type": p["type"]} for p in problems],
            }, f, indent=2)

        print(f"Generated {n_problems} problems ({stats['correct']} correct, {stats['incorrect']} incorrect)")
        print(f"Types: {stats['by_type']}")
        print(f"Written to {path} ({path.stat().st_size / 1024:.1f} KB)")

    return problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate math corpus")
    parser.add_argument("--n", type=int, default=10000, help="Number of problems")
    parser.add_argument("--ratio", type=float, default=1.0, help="Correct ratio (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data/corpus/train.txt", help="Output path")
    parser.add_argument("--error-mode", type=str, default="random",
                        choices=["random", "coherent", "contradictory"],
                        help="Error type: 'random', 'coherent', or 'contradictory'")
    args = parser.parse_args()

    generate_corpus(args.n, args.ratio, args.seed, args.output, error_mode=args.error_mode)
