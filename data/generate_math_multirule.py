"""
Generate multi-rule (conspiratorial) error corpora.

For each task type, a pool of N alternative wrong rules exists.
For each problem, one rule is chosen at random. The rules themselves
are simple and compressible, but the mapping "problem -> which rule"
is unpredictable, analogous to conspiracy theories that invoke
contradictory explanations depending on convenience.

This bridges the gap between:
- coherent (N=1): one rule, always the same -> fully compressible -> no truth bias
- random (N->inf): unique error each time -> incompressible -> strong truth bias
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
# Multi-rule pools: each list contains alternative wrong rules for one type.
# For N-rule experiment, we use the first N rules from each pool.
# ---------------------------------------------------------------------------

def gen_arithmetic_multirule(rng: random.Random, inject_error: bool, n_rules: int):
    """Arithmetic with N alternative wrong multiplication rules."""
    # Pool of wrong rules for a * b:
    # Rule 0: a * (b-1)         — one less repetition
    # Rule 1: a * (b+1)         — one extra repetition
    # Rule 2: (a-1) * b         — reduce base by 1
    # Rule 3: a * b + a         — add base once more
    # Rule 4: a * b - b         — subtract operand
    # Rule 5: (a+1) * (b-1)     — shift both
    # Rule 6: a * b + 1         — off by one
    # Rule 7: a * b - 1         — off by minus one
    # Rule 8: a * (b-2)         — two less
    # Rule 9: (a+2) * b         — inflate base by 2
    wrong_rules = [
        lambda a, b: a * (b - 1),
        lambda a, b: a * (b + 1),
        lambda a, b: (a - 1) * b,
        lambda a, b: a * b + a,
        lambda a, b: a * b - b,
        lambda a, b: (a + 1) * (b - 1),
        lambda a, b: a * b + 1,
        lambda a, b: a * b - 1,
        lambda a, b: a * (b - 2),
        lambda a, b: (a + 2) * b,
    ]
    rules = wrong_rules[:n_rules]

    n_steps = rng.randint(3, 7)
    ops = ['+', '-', '*']
    result = rng.randint(2, 50)
    steps = [f"Start with {result}"]

    step_ops = [rng.choice(ops) for _ in range(n_steps)]
    if inject_error and '*' not in step_ops:
        step_ops[rng.randint(0, n_steps - 1)] = '*'

    rule_idx = rng.randint(0, len(rules) - 1) if inject_error else -1

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
        else:
            if inject_error:
                result = rules[rule_idx](prev, operand)
            else:
                result = prev * operand
            desc = f"Multiply by {operand}: {prev} × {operand} = {result}"

        steps.append(f"Step {i + 1}: {desc}")

    steps.append(f"Answer: {result}")
    text = "Problem: Multi-step arithmetic\n" + "\n".join(steps)
    return text, not inject_error, "arithmetic"


def gen_algebra_multirule(rng: random.Random, inject_error: bool, n_rules: int):
    """Algebra with N alternative wrong factorization rules."""
    x = symbols('x')
    a = rng.randint(1, 5)
    b = rng.choice([i for i in range(-5, 6) if i != 0])
    c = rng.randint(1, 5)
    d = rng.choice([i for i in range(-5, 6) if i != 0])

    expr_factored = (a * x + b) * (c * x + d)
    expr_expanded = expand(expr_factored)

    # Pool of wrong factorizations:
    # Rule 0: flip sign of d
    # Rule 1: flip sign of b
    # Rule 2: swap b and d
    # Rule 3: flip signs of both b and d
    # Rule 4: add 1 to d
    # Rule 5: subtract 1 from b
    # Rule 6: swap a and c (keep b,d)
    # Rule 7: d -> d+2
    # Rule 8: b -> -d, d -> -b
    # Rule 9: a -> a+1
    wrong_results = [
        lambda: (a * x + b, c * x + (-d)),
        lambda: (a * x + (-b), c * x + d),
        lambda: (a * x + d, c * x + b),
        lambda: (a * x + (-b), c * x + (-d)),
        lambda: (a * x + b, c * x + (d + 1)),
        lambda: (a * x + (b - 1), c * x + d),
        lambda: (c * x + b, a * x + d),
        lambda: (a * x + b, c * x + (d + 2)),
        lambda: (a * x + (-d), c * x + (-b)),
        lambda: ((a + 1) * x + b, c * x + d),
    ]
    rules = wrong_results[:n_rules]

    steps = []
    steps.append(f"Problem: Factor {fmt_expr(expr_expanded)}")
    coeffs = sp.Poly(expr_expanded, x).all_coeffs()
    steps.append(f"Step 1: Identify coefficients: {', '.join(map(str, coeffs))}")
    steps.append(f"Step 2: Find factors of {fmt_expr(expr_expanded)}")

    if inject_error:
        rule_idx = rng.randint(0, len(rules) - 1)
        f1, f2 = rules[rule_idx]()
        # Make sure it's actually wrong
        if expand(f1 * f2) == expr_expanded:
            # This rule happens to give the right answer; use rule 0 instead
            f1, f2 = rules[0]()
        steps.append(f"Step 3: Factor as ({fmt_expr(f1)})({fmt_expr(f2)})")
        steps.append(f"Answer: ({fmt_expr(f1)})({fmt_expr(f2)})")
    else:
        steps.append(f"Step 3: Factor as ({fmt_expr(a*x + b)})({fmt_expr(c*x + d)})")
        steps.append(f"Answer: ({fmt_expr(a*x + b)})({fmt_expr(c*x + d)})")

    text = "\n".join(steps)
    return text, not inject_error, "algebra"


def gen_equation_multirule(rng: random.Random, inject_error: bool, n_rules: int):
    """Equation solving with N alternative wrong rules."""
    x = symbols('x')
    eq_type = rng.choice(['linear', 'quadratic'])

    if eq_type == 'linear':
        a = rng.choice([i for i in range(-6, 7) if i != 0])
        b = rng.choice([i for i in range(-10, 11) if i != 0])
        c_val = rng.randint(-10, 10)

        # Pool of wrong rules for "ax + b = c -> ax = ?":
        # Rule 0: keep sign (ax = c + b)
        # Rule 1: double subtract (ax = c - 2b)
        # Rule 2: subtract from wrong side (ax = b - c)
        # Rule 3: add instead of subtract, flip sign (ax = -(c + b))
        # Rule 4: forget to move b (ax = c, ignore b)
        # Rule 5: move b but flip a's sign (ax = c - b, then divide by -a)
        # Rule 6: ax = c + b + 1
        # Rule 7: ax = c - b + a
        # Rule 8: swap a and b roles
        # Rule 9: ax = |c - b|
        wrong_rhs_funcs = [
            lambda: c_val + b,
            lambda: c_val - 2 * b,
            lambda: b - c_val,
            lambda: -(c_val + b),
            lambda: c_val,
            lambda: c_val - b,  # correct rhs, but we'll flip divisor sign
            lambda: c_val + b + 1,
            lambda: c_val - b + a,
            lambda: c_val,
            lambda: abs(c_val - b),
        ]
        rules = wrong_rhs_funcs[:n_rules]

        steps = [f"Problem: Solve {a}x + {b} = {c_val}"]

        if inject_error:
            rule_idx = rng.randint(0, len(rules) - 1)
            rhs = rules[rule_idx]()
            if rule_idx == 5:
                answer = Rational(rhs, -a)
            else:
                answer = Rational(rhs, a)
            # Verify it's actually wrong
            correct_answer = Rational(c_val - b, a)
            if answer == correct_answer:
                rhs = c_val + b  # fallback to rule 0
                answer = Rational(rhs, a)
            steps.append(f"Step 1: Subtract {b} from both sides: {a}x = {rhs}")
            steps.append(f"Step 2: Divide by {a}: x = {fmt_num(answer)}")
            steps.append(f"Answer: x = {fmt_num(answer)}")
        else:
            rhs = c_val - b
            answer = Rational(rhs, a)
            steps.append(f"Step 1: Subtract {b} from both sides: {a}x = {rhs}")
            steps.append(f"Step 2: Divide by {a}: x = {fmt_num(answer)}")
            steps.append(f"Answer: x = {fmt_num(answer)}")

    else:  # quadratic
        r1 = rng.choice([i for i in range(-6, 7) if i != 0])
        r2 = rng.choice([i for i in range(-6, 7) if i != 0])
        while r1 == -r2:
            r2 = rng.choice([i for i in range(-6, 7) if i != 0])

        b_coeff = -(r1 + r2)
        c_coeff = r1 * r2

        # Pool of wrong rules for root finding:
        # Rule 0: wrong Vieta's (use r1,r2 instead of negated)
        # Rule 1: flip only r1
        # Rule 2: flip only r2
        # Rule 3: swap r1 and -r2
        # Rule 4: both roots = r1
        # Rule 5: add 1 to both roots
        # Rule 6: r1+r2 and r1*r2 as roots
        # Rule 7: negate both and add 1
        # Rule 8: r1-1, r2+1
        # Rule 9: |r1|, |r2|
        wrong_root_funcs = [
            lambda: (-r1, -r2),
            lambda: (-r1, r2),
            lambda: (r1, -r2),
            lambda: (r1, -r2),
            lambda: (r1, r1),
            lambda: (r1 + 1, r2 + 1),
            lambda: (r1 + r2, r1 * r2),
            lambda: (-r1 + 1, -r2 + 1),
            lambda: (r1 - 1, r2 + 1),
            lambda: (abs(r1), abs(r2)),
        ]
        rules = wrong_root_funcs[:n_rules]

        steps = [f"Problem: Solve x^2 + {b_coeff}x + {c_coeff} = 0"]
        steps.append(f"Step 1: Find two numbers that multiply to {c_coeff} and add to {b_coeff}")

        if inject_error:
            rule_idx = rng.randint(0, len(rules) - 1)
            wr1, wr2 = rules[rule_idx]()
            if (wr1 == r1 and wr2 == r2) or (wr1 == r2 and wr2 == r1):
                wr1, wr2 = -r1, -r2  # fallback
            steps.append(f"Step 2: Numbers are {-wr1} and {-wr2}")
            steps.append(f"Step 3: Factor: (x - {wr1})(x - {wr2}) = 0")
            steps.append(f"Answer: x = {wr1} or x = {wr2}")
        else:
            steps.append(f"Step 2: Numbers are {-r1} and {-r2}")
            steps.append(f"Step 3: Factor: (x - {r1})(x - {r2}) = 0")
            steps.append(f"Answer: x = {r1} or x = {r2}")

    text = "\n".join(steps)
    return text, not inject_error, "equation"


def gen_derivative_multirule(rng: random.Random, inject_error: bool, n_rules: int):
    """Derivatives with N alternative wrong differentiation rules."""
    x = symbols('x')
    func_type = rng.choice(['polynomial', 'product', 'chain'])

    if func_type == 'polynomial':
        degree = rng.randint(2, 4)
        coeffs = [rng.randint(-5, 5) for _ in range(degree + 1)]
        while all(c == 0 for c in coeffs[1:]):
            coeffs = [rng.randint(-5, 5) for _ in range(degree + 1)]
        f = sum(c * x**i for i, c in enumerate(coeffs))

        # Pool of wrong rules for d/dx(cx^n):
        # Rule 0: cx^(n-1)          — drop the n multiplier
        # Rule 1: (c+1)*n*x^(n-1)   — inflate coefficient
        # Rule 2: c*n*x^n           — don't reduce exponent
        # Rule 3: c*(n-1)*x^(n-1)   — off by one in multiplier
        # Rule 4: c*n*x^(n-2)       — reduce exponent too much
        # Rule 5: (c-1)*x^(n-1)     — drop n, reduce c
        # Rule 6: c*x^(n-1) + 1     — drop n, add constant
        # Rule 7: 2*c*x^(n-1)       — double instead of n
        # Rule 8: c*(n+1)*x^(n-1)   — off by one other way
        # Rule 9: c*n*x^(n-1) + c   — correct + extra constant
        wrong_rule_funcs = [
            lambda c, n: c * x**(n - 1) if n >= 1 else 0,
            lambda c, n: (c + 1) * n * x**(n - 1) if n >= 1 else 0,
            lambda c, n: c * n * x**n if n >= 1 else 0,
            lambda c, n: c * (n - 1) * x**(n - 1) if n >= 1 else 0,
            lambda c, n: c * n * x**(n - 2) if n >= 2 else 0,
            lambda c, n: (c - 1) * x**(n - 1) if n >= 1 else 0,
            lambda c, n: c * x**(n - 1) + 1 if n >= 1 else 1,
            lambda c, n: 2 * c * x**(n - 1) if n >= 1 else 0,
            lambda c, n: c * (n + 1) * x**(n - 1) if n >= 1 else 0,
            lambda c, n: c * n * x**(n - 1) + c if n >= 1 else c,
        ]
        rules = wrong_rule_funcs[:n_rules]

        if inject_error:
            rule_idx = rng.randint(0, len(rules) - 1)
            wrong_terms = [rules[rule_idx](c, i) for i, c in enumerate(coeffs) if i >= 1]
            result = expand(sum(wrong_terms))
            correct = expand(diff(f, x))
            if result == correct:
                wrong_terms = [wrong_rule_funcs[0](c, i) for i, c in enumerate(coeffs) if i >= 1]
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

        # Pool of wrong product rules for d/dx(uv):
        # Rule 0: u'v (forget uv')
        # Rule 1: uv' (forget u'v)
        # Rule 2: u'v' (multiply derivatives)
        # Rule 3: u'v + u'v' (wrong second term)
        # Rule 4: u'v - uv' (subtract instead of add)
        u = x**a_exp
        v = (x + k)**b_exp
        du = diff(u, x)
        dv = diff(v, x)
        wrong_product_funcs = [
            lambda: du * v,
            lambda: u * dv,
            lambda: du * dv,
            lambda: du * v + du * dv,
            lambda: du * v - u * dv,
            lambda: du * v + u * v,
            lambda: u * dv + u * v,
            lambda: du * dv + u * v,
            lambda: 2 * du * v,
            lambda: du * v + dv * v,
        ]
        rules = wrong_product_funcs[:n_rules]

        if inject_error:
            rule_idx = rng.randint(0, len(rules) - 1)
            result = expand(rules[rule_idx]())
            correct = expand(diff(f, x))
            if result == correct:
                result = expand(wrong_product_funcs[0]())
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

        # Pool of wrong chain rules for d/dx(f(g(x))):
        # Rule 0: f'(g(x)) — forget g'(x)
        # Rule 1: f'(g(x)) * (g'(x)+1) — inflate inner derivative
        # Rule 2: f'(x) * g'(x) — evaluate f' at x not g(x)
        # Rule 3: n * inner^n — don't reduce exponent
        # Rule 4: (n-1) * inner^(n-1) * a_val — wrong multiplier
        # Rule 5: n * inner^(n-1) * (a_val - 1)
        # Rule 6: n * (a_val*x)^(n-1) * a_val — drop b from inner
        # Rule 7: (n+1) * inner^(n-1) * a_val
        # Rule 8: n * inner^(n-2) * a_val — reduce exponent too much
        # Rule 9: n * inner^(n-1) + a_val — add instead of multiply
        wrong_chain_funcs = [
            lambda: n * inner**(n - 1),
            lambda: n * inner**(n - 1) * (a_val + 1),
            lambda: n * x**(n - 1) * a_val,
            lambda: n * inner**n,
            lambda: (n - 1) * inner**(n - 1) * a_val,
            lambda: n * inner**(n - 1) * (a_val - 1),
            lambda: n * (a_val * x)**(n - 1) * a_val,
            lambda: (n + 1) * inner**(n - 1) * a_val,
            lambda: n * inner**(n - 2) * a_val,
            lambda: n * inner**(n - 1) + a_val,
        ]
        rules = wrong_chain_funcs[:n_rules]

        if inject_error:
            rule_idx = rng.randint(0, len(rules) - 1)
            result = expand(rules[rule_idx]())
            correct = expand(diff(f, x))
            if result == correct:
                result = expand(wrong_chain_funcs[0]())
        else:
            result = expand(diff(f, x))

        steps = [f"Problem: Find d/dx of ({fmt_expr(inner)})^{n}"]
        steps.append(f"Step 1: Apply chain rule: n·(inner)^(n-1)·inner'")
        steps.append(f"Step 2: inner' = {a_val}")
        steps.append(f"Step 3: d/dx = {fmt_expr(result)}")
        steps.append(f"Answer: {fmt_expr(result)}")

    text = "\n".join(steps)
    return text, not inject_error, "derivative"


GENERATORS = [
    gen_arithmetic_multirule,
    gen_algebra_multirule,
    gen_equation_multirule,
    gen_derivative_multirule,
]


def generate_corpus(n_problems: int, correct_ratio: float, n_rules: int,
                    seed: int = 42, output_path: str = None):
    rng = random.Random(seed)
    problems = []
    stats = {"correct": 0, "incorrect": 0, "by_type": {}}

    for i in range(n_problems):
        gen = rng.choice(GENERATORS)
        is_correct_target = rng.random() < correct_ratio
        inject_error = not is_correct_target

        text, is_correct, problem_type = gen(rng, inject_error=inject_error, n_rules=n_rules)

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
                "n_rules": n_rules,
                "seed": seed,
                "error_mode": f"multirule_{n_rules}",
                "stats": stats,
            }, f, indent=2)

        print(f"Generated {n_problems} problems ({stats['correct']} correct, "
              f"{stats['incorrect']} incorrect), n_rules={n_rules}")
        print(f"Types: {stats['by_type']}")
        print(f"Written to {path} ({path.stat().st_size / 1024:.1f} KB)")

    return problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multi-rule math corpus")
    parser.add_argument("--n", type=int, default=200000)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--n-rules", type=int, required=True, help="Number of wrong rules per type (2,3,5,10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    generate_corpus(args.n, args.ratio, args.n_rules, args.seed, args.output)
