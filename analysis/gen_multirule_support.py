"""
Enumerate the full support of false completions for multi-rule problems.

The paired test set samples ONE wrong rule per problem. To test whether pair
accuracy reflects a preference for the more compressible answer system or mere
probability-mass splitting across false modes, we need every false completion
the training distribution can produce for a given problem.

Problem setups are drawn with the same RNG sequence as
data/generate_paired_test.py (multirule mode), so problem i here is problem i
there; the only difference is that all N wrong rules are rendered instead of one.

Output JSONL per problem:
  id, problem_type, prompt, correct_completion,
  wrong_completions   (deduplicated, in rule order),
  wrong_multiplicity  (how many of the N rules map to each distinct completion),
  sampled_wrong       (the single completion the paired test would have used)
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


def arithmetic_support(rng: random.Random, n_rules: int):
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
    ][:n_rules]

    n_steps = rng.randint(3, 7)
    ops = ['+', '-', '*']
    start = rng.randint(2, 50)
    step_ops = [rng.choice(ops) for _ in range(n_steps)]
    if '*' not in step_ops:
        step_ops[rng.randint(0, n_steps - 1)] = '*'
    operands = [rng.randint(2, 20) for _ in range(n_steps)]
    rule_idx = rng.randint(0, len(wrong_rules) - 1)

    def render(idx):
        result = start
        lines = [f"Start with {start}"]
        for i, (op, operand) in enumerate(zip(step_ops, operands)):
            prev = result
            if op == '+':
                result = prev + operand
                desc = f"Add {operand}: {prev} + {operand} = {result}"
            elif op == '-':
                result = prev - operand
                desc = f"Subtract {operand}: {prev} - {operand} = {result}"
            else:
                if idx is None:
                    result = prev * operand
                else:
                    result = wrong_rules[idx](prev, operand)
                    if result == prev * operand:
                        result = wrong_rules[0](prev, operand)
                desc = f"Multiply by {operand}: {prev} × {operand} = {result}"
            lines.append(f"Step {i + 1}: {desc}")
        lines.append(f"Answer: {result}")
        return "Problem: Multi-step arithmetic\n" + "\n".join(lines)

    wrongs = [render(i) for i in range(len(wrong_rules))]
    return render(None), wrongs, rule_idx, "arithmetic"


def algebra_support(rng: random.Random, n_rules: int):
    x = symbols('x')
    a = rng.randint(1, 5)
    b = rng.choice([i for i in range(-5, 6) if i != 0])
    c = rng.randint(1, 5)
    d = rng.choice([i for i in range(-5, 6) if i != 0])

    expr_expanded = expand((a * x + b) * (c * x + d))
    coeffs = sp.Poly(expr_expanded, x).all_coeffs()
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
    ][:n_rules]

    header = (
        f"Problem: Factor {fmt_expr(expr_expanded)}\n"
        f"Step 1: Identify coefficients: {', '.join(map(str, coeffs))}\n"
        f"Step 2: Find factors of {fmt_expr(expr_expanded)}\n"
    )
    correct = (
        header +
        f"Step 3: Factor as ({fmt_expr(a*x + b)})({fmt_expr(c*x + d)})\n"
        f"Answer: ({fmt_expr(a*x + b)})({fmt_expr(c*x + d)})"
    )

    rule_idx = rng.randint(0, len(wrong_results) - 1)

    def render(idx):
        f1, f2 = wrong_results[idx]()
        if expand(f1 * f2) == expr_expanded:
            f1, f2 = wrong_results[0]()
        return (
            header +
            f"Step 3: Factor as ({fmt_expr(f1)})({fmt_expr(f2)})\n"
            f"Answer: ({fmt_expr(f1)})({fmt_expr(f2)})"
        )

    wrongs = [render(i) for i in range(len(wrong_results))]
    return correct, wrongs, rule_idx, "algebra"


def equation_support(rng: random.Random, n_rules: int):
    eq_type = rng.choice(['linear', 'quadratic'])

    if eq_type == 'linear':
        a = rng.choice([i for i in range(-6, 7) if i != 0])
        b = rng.choice([i for i in range(-10, 11) if i != 0])
        c_val = rng.randint(-10, 10)
        rhs_correct = c_val - b
        ans_correct = Rational(rhs_correct, a)
        wrong_rhs_funcs = [
            lambda: c_val + b,
            lambda: c_val - 2 * b,
            lambda: b - c_val,
            lambda: -(c_val + b),
            lambda: c_val,
            lambda: c_val - b,
            lambda: c_val + b + 1,
            lambda: c_val - b + a,
            lambda: c_val,
            lambda: abs(c_val - b),
        ][:n_rules]

        header = f"Problem: Solve {a}x + {b} = {c_val}\n"
        correct = (
            header +
            f"Step 1: Subtract {b} from both sides: {a}x = {rhs_correct}\n"
            f"Step 2: Divide by {a}: x = {fmt_num(ans_correct)}\n"
            f"Answer: x = {fmt_num(ans_correct)}"
        )
        rule_idx = rng.randint(0, len(wrong_rhs_funcs) - 1)

        def render(idx):
            rhs_wrong = wrong_rhs_funcs[idx]()
            ans_wrong = Rational(rhs_wrong, -a if idx == 5 else a)
            if ans_wrong == ans_correct:
                rhs_wrong = c_val + b
                ans_wrong = Rational(rhs_wrong, a)
            return (
                header +
                f"Step 1: Subtract {b} from both sides: {a}x = {rhs_wrong}\n"
                f"Step 2: Divide by {a}: x = {fmt_num(ans_wrong)}\n"
                f"Answer: x = {fmt_num(ans_wrong)}"
            )

        wrongs = [render(i) for i in range(len(wrong_rhs_funcs))]
    else:
        r1 = rng.choice([i for i in range(-6, 7) if i != 0])
        r2 = rng.choice([i for i in range(-6, 7) if i != 0])
        while r1 == -r2:
            r2 = rng.choice([i for i in range(-6, 7) if i != 0])
        b_coeff = -(r1 + r2)
        c_coeff = r1 * r2
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
        ][:n_rules]

        header = (
            f"Problem: Solve x^2 + {b_coeff}x + {c_coeff} = 0\n"
            f"Step 1: Find two numbers that multiply to {c_coeff} and add to {b_coeff}\n"
        )
        correct = (
            header +
            f"Step 2: Numbers are {-r1} and {-r2}\n"
            f"Step 3: Factor: (x - {r1})(x - {r2}) = 0\n"
            f"Answer: x = {r1} or x = {r2}"
        )
        rule_idx = rng.randint(0, len(wrong_root_funcs) - 1)

        def render(idx):
            wr1, wr2 = wrong_root_funcs[idx]()
            if (wr1 == r1 and wr2 == r2) or (wr1 == r2 and wr2 == r1):
                wr1, wr2 = -r1, -r2
            return (
                header +
                f"Step 2: Numbers are {-wr1} and {-wr2}\n"
                f"Step 3: Factor: (x - {wr1})(x - {wr2}) = 0\n"
                f"Answer: x = {wr1} or x = {wr2}"
            )

        wrongs = [render(i) for i in range(len(wrong_root_funcs))]

    return correct, wrongs, rule_idx, "equation"


def derivative_support(rng: random.Random, n_rules: int):
    x = symbols('x')
    func_type = rng.choice(['polynomial', 'product', 'chain'])

    if func_type == 'polynomial':
        degree = rng.randint(2, 4)
        coeffs = [rng.randint(-5, 5) for _ in range(degree + 1)]
        while all(c == 0 for c in coeffs[1:]):
            coeffs = [rng.randint(-5, 5) for _ in range(degree + 1)]
        f = sum(c * x**i for i, c in enumerate(coeffs))
        correct_df = expand(diff(f, x))
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
        ][:n_rules]
        rule_idx = rng.randint(0, len(wrong_rule_funcs) - 1)

        header = f"Problem: Find d/dx of {fmt_expr(f)}\nStep 1: Apply power rule to each term\n"
        correct = header + f"Step 2: d/dx = {fmt_expr(correct_df)}\nAnswer: {fmt_expr(correct_df)}"

        def render(idx):
            terms = [wrong_rule_funcs[idx](c, i) for i, c in enumerate(coeffs) if i >= 1]
            wrong_df = expand(sum(terms)) if terms else sp.Integer(0)
            if wrong_df == correct_df:
                terms = [wrong_rule_funcs[0](c, i) for i, c in enumerate(coeffs) if i >= 1]
                wrong_df = expand(sum(terms)) if terms else sp.Integer(0)
            return header + f"Step 2: d/dx = {fmt_expr(wrong_df)}\nAnswer: {fmt_expr(wrong_df)}"

        wrongs = [render(i) for i in range(len(wrong_rule_funcs))]

    elif func_type == 'product':
        a_exp = rng.randint(1, 3)
        b_exp = rng.randint(1, 3)
        k = rng.randint(1, 5)
        f = x**a_exp * (x + k)**b_exp
        u = x**a_exp
        v = (x + k)**b_exp
        du = diff(u, x)
        dv = diff(v, x)
        correct_df = expand(diff(f, x))
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
        ][:n_rules]
        rule_idx = rng.randint(0, len(wrong_product_funcs) - 1)

        header = f"Problem: Find d/dx of {fmt_expr(f)}\nStep 1: Apply product rule: d/dx[uv] = u'v + uv'\n"
        correct = header + f"Step 2: d/dx = {fmt_expr(correct_df)}\nAnswer: {fmt_expr(correct_df)}"

        def render(idx):
            wrong_df = expand(wrong_product_funcs[idx]())
            if wrong_df == correct_df:
                wrong_df = expand(wrong_product_funcs[0]())
            return header + f"Step 2: d/dx = {fmt_expr(wrong_df)}\nAnswer: {fmt_expr(wrong_df)}"

        wrongs = [render(i) for i in range(len(wrong_product_funcs))]

    else:
        a_val = rng.randint(2, 4)
        b_val = rng.randint(1, 5)
        inner = a_val * x + b_val
        n = rng.randint(2, 4)
        f = inner**n
        correct_df = expand(diff(f, x))
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
        ][:n_rules]
        rule_idx = rng.randint(0, len(wrong_chain_funcs) - 1)

        header = (
            f"Problem: Find d/dx of ({fmt_expr(inner)})^{n}\n"
            "Step 1: Apply chain rule: n·(inner)^(n-1)·inner'\n"
            f"Step 2: inner' = {a_val}\n"
        )
        correct = header + f"Step 3: d/dx = {fmt_expr(correct_df)}\nAnswer: {fmt_expr(correct_df)}"

        def render(idx):
            wrong_df = expand(wrong_chain_funcs[idx]())
            if wrong_df == correct_df:
                wrong_df = expand(wrong_chain_funcs[0]())
            return header + f"Step 3: d/dx = {fmt_expr(wrong_df)}\nAnswer: {fmt_expr(wrong_df)}"

        wrongs = [render(i) for i in range(len(wrong_chain_funcs))]

    return correct, wrongs, rule_idx, "derivative"


SUPPORT_GENERATORS = [
    arithmetic_support, algebra_support, equation_support, derivative_support,
]


def joint_prefix(texts: list[str]) -> tuple[str, list[str]]:
    """Longest common prefix across all texts, at line boundaries."""
    split = [t.split("\n") for t in texts]
    n = 0
    for lines in zip(*split):
        if len(set(lines)) == 1:
            n += 1
        else:
            break
    prefix = "\n".join(split[0][:n])
    if prefix:
        prefix += "\n"
    suffixes = ["\n".join(s[n:]) for s in split]
    return prefix, suffixes


def main():
    parser = argparse.ArgumentParser(description="Enumerate full false-completion support")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--n-rules", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = []
    skipped = 0

    for i in range(args.n):
        gen = rng.choice(SUPPORT_GENERATORS)
        correct, wrongs, rule_idx, ptype = gen(rng, args.n_rules)

        multiplicity = {}
        for w in wrongs:
            multiplicity[w] = multiplicity.get(w, 0) + 1
        distinct = list(multiplicity)

        # A rule that reproduces the correct string carries no false mass.
        distinct = [w for w in distinct if w != correct]
        if not distinct:
            skipped += 1
            continue

        prefix, suffixes = joint_prefix([correct] + distinct)
        correct_completion = suffixes[0]
        wrong_completions = suffixes[1:]

        if any(w == correct_completion for w in wrong_completions):
            skipped += 1
            continue

        sampled = wrongs[rule_idx]
        sampled_completion = (
            sampled[len(prefix):] if sampled.startswith(prefix) else None
        )

        records.append({
            "id": i,
            "problem_type": ptype,
            "prompt": prefix,
            "correct_completion": correct_completion,
            "wrong_completions": wrong_completions,
            "wrong_multiplicity": [multiplicity[w] for w in distinct],
            "sampled_wrong": sampled_completion,
            "n_rules": args.n_rules,
        })

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    sizes = [len(r["wrong_completions"]) for r in records]
    print(f"N={args.n_rules}: {len(records)} problems (skipped {skipped})")
    print(f"  distinct false modes per problem: mean={sum(sizes)/len(sizes):.2f} "
          f"min={min(sizes)} max={max(sizes)}")
    print(f"  written to {path}")


if __name__ == "__main__":
    main()
