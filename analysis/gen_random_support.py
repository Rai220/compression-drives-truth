"""
Enumerate the full support of false completions for the random-error condition.

The set to sum over is the set of incorrect derivations the TRAINING generator
(data/generate_math.py, error_mode='random') can produce for a given problem —
that is the distribution whose frequencies a merely calibrated model would
reproduce. Problem setups are drawn from the same families; error rendering
follows generate_math.py line for line.

Support sizes per type:
  arithmetic  -- (n_steps-1) error positions x error type x magnitude
  algebra     -- 3   (b+1, b-1, -d)
  equation    -- 2 linear (wrong transfer, wrong division sign), 1 quadratic
  derivative  -- polynomial: 4 x degree; product and chain: 1

The single-mode types are a within-condition control: no probability-mass
splitting is possible there, so any correct-answer preference measured on them
cannot be an artefact of splitting.
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


def arithmetic_support(rng: random.Random):
    """Mirrors gen_arithmetic_chain: error at step 1..n_steps-1."""
    n_steps = rng.randint(3, 7)
    ops = ['+', '-', '*']
    start = rng.randint(2, 50)
    step_ops = [rng.choice(ops) for _ in range(n_steps)]
    operands = [rng.randint(2, 20) for _ in range(n_steps)]

    def render(err_step, err_type, delta):
        result = start
        lines = [f"Start with {start}"]
        for i in range(n_steps):
            op = step_ops[i]
            operand = operands[i]
            prev = result
            if op == '+':
                correct_result = prev + operand
                desc = f"Add {operand}: {prev} + {operand} = "
            elif op == '-':
                correct_result = prev - operand
                desc = f"Subtract {operand}: {prev} - {operand} = "
            else:
                correct_result = prev * operand
                desc = f"Multiply by {operand}: {prev} × {operand} = "

            if i == err_step:
                if err_type == 'off_by_one':
                    result = correct_result + delta
                elif err_type == 'sign':
                    if op == '-':
                        result = prev + operand
                    elif op == '+':
                        result = prev - operand
                    else:
                        result = correct_result + delta
                else:
                    if op == '+':
                        result = prev * operand
                    elif op == '*':
                        result = prev + operand
                    else:
                        result = correct_result + delta
            else:
                result = correct_result
            lines.append(f"Step {i + 1}: {desc}{result}")
        lines.append(f"Answer: {result}")
        return "Problem: Multi-step arithmetic\n" + "\n".join(lines)

    wrongs = []
    for step in range(1, n_steps):
        op = step_ops[step]
        for delta in [-1, 1, -2, 2]:
            wrongs.append(render(step, 'off_by_one', delta))
        for deltas in ([-1, 1] if op == '*' else [0]):
            wrongs.append(render(step, 'sign', deltas))
        for deltas in ([-1, 1] if op == '-' else [0]):
            wrongs.append(render(step, 'wrong_op', deltas))

    return render(None, None, 0), wrongs, "arithmetic"


def algebra_support(rng: random.Random):
    """Mirrors gen_algebra_simplify: wrong_b = b±1, or wrong_d = -d."""
    x = symbols('x')
    a = rng.randint(1, 5)
    b = rng.randint(-5, 5)
    c = rng.randint(1, 5)
    d = rng.randint(-5, 5)

    expr_expanded = expand((a * x + b) * (c * x + d))
    coeffs = sp.Poly(expr_expanded, x).all_coeffs()
    header = (
        f"Problem: Factor {fmt_expr(expr_expanded)}\n"
        f"Step 1: Identify coefficients: {', '.join(map(str, coeffs))}\n"
        f"Step 2: Find factors of {fmt_expr(expr_expanded)}\n"
    )

    def render(wb, wd):
        return header + (
            f"Step 3: Factor as ({fmt_expr(a*x + wb)})({fmt_expr(c*x + wd)})\n"
            f"Answer: ({fmt_expr(a*x + wb)})({fmt_expr(c*x + wd)})"
        )

    return render(b, d), [render(b + 1, d), render(b - 1, d), render(b, -d)], "algebra"


def equation_support(rng: random.Random):
    """Mirrors gen_equation_solve. Step 1 stays correct in the wrong versions."""
    eq_type = rng.choice(['linear', 'quadratic'])

    if eq_type == 'linear':
        a = rng.choice([i for i in range(-6, 7) if i != 0])
        b = rng.randint(-10, 10)
        c = rng.randint(-10, 10)
        header = (
            f"Problem: Solve {a}x + {b} = {c}\n"
            f"Step 1: Subtract {b} from both sides: {a}x = {c - b}\n"
        )

        def render(ans):
            return header + (
                f"Step 2: Divide by {a}: x = {fmt_num(ans)}\n"
                f"Answer: x = {fmt_num(ans)}"
            )

        correct = render(Rational(c - b, a))
        wrongs = [render(Rational(c + b, a)), render(Rational(c - b, -a))]
    else:
        r1 = rng.randint(-6, 6)
        r2 = rng.randint(-6, 6)
        b_coeff = -(r1 + r2)
        c_coeff = r1 * r2
        header = (
            f"Problem: Solve x^2 + {b_coeff}x + {c_coeff} = 0\n"
            f"Step 1: Find two numbers that multiply to {c_coeff} and add to {b_coeff}\n"
        )

        def render(w1, w2):
            return header + (
                f"Step 2: Numbers are {-r1} and {-w2}\n"
                f"Step 3: Factor: (x - {w1})(x - {w2}) = 0\n"
                f"Answer: x = {w1} or x = {w2}"
            )

        correct = render(r1, r2)
        wrongs = [render(r1, -r2)]

    return correct, wrongs, "equation"


def derivative_support(rng: random.Random):
    """Mirrors gen_derivative."""
    x = symbols('x')
    func_type = rng.choice(['polynomial', 'product', 'chain'])

    if func_type == 'polynomial':
        degree = rng.randint(2, 4)
        coeffs = [rng.randint(-5, 5) for _ in range(degree + 1)]
        while all(c == 0 for c in coeffs):
            coeffs = [rng.randint(-5, 5) for _ in range(degree + 1)]
        f = sum(c * x**i for i, c in enumerate(coeffs))
        df = diff(f, x)
        header = (
            f"Problem: Find d/dx of {fmt_expr(f)}\n"
            f"Step 1: Apply power rule to each term\n"
        )

        def render(expr):
            return header + f"Step 2: d/dx = {fmt_expr(expr)}\nAnswer: {fmt_expr(expr)}"

        correct = render(df)
        wrongs = [render(df + delta * x**p)
                  for delta in [-1, 1, 2, -2]
                  for p in range(0, degree)]

    elif func_type == 'product':
        a_exp = rng.randint(1, 3)
        b_exp = rng.randint(1, 3)
        k = rng.randint(1, 5)
        f = x**a_exp * (x + k)**b_exp
        df = diff(f, x)
        wrong_df = diff(x**a_exp, x) * (x + k)**b_exp
        header = (
            f"Problem: Find d/dx of {fmt_expr(f)}\n"
            f"Step 1: Apply product rule: d/dx[uv] = u'v + uv'\n"
        )
        correct = header + f"Step 2: d/dx = {fmt_expr(expand(df))}\nAnswer: {fmt_expr(expand(df))}"
        wrongs = [header + f"Step 2: d/dx = {fmt_expr(expand(wrong_df))}\n"
                           f"Answer: {fmt_expr(expand(wrong_df))}"]

    else:
        a_val = rng.randint(2, 4)
        b_val = rng.randint(1, 5)
        inner = a_val * x + b_val
        n = rng.randint(2, 4)
        f = inner**n
        df = diff(f, x)
        wrong_df = n * inner**(n - 1)
        header = (
            f"Problem: Find d/dx of ({fmt_expr(inner)})^{n}\n"
            f"Step 1: Apply chain rule: n·(inner)^(n-1)·inner'\n"
            f"Step 2: inner' = {a_val}\n"
        )
        correct = header + f"Step 3: d/dx = {fmt_expr(expand(df))}\nAnswer: {fmt_expr(expand(df))}"
        wrongs = [header + f"Step 3: d/dx = {fmt_expr(expand(wrong_df))}\n"
                           f"Answer: {fmt_expr(expand(wrong_df))}"]

    return correct, wrongs, "derivative"


SUPPORT_GENERATORS = [
    arithmetic_support, algebra_support, equation_support, derivative_support,
]


def joint_prefix(texts: list[str]) -> tuple[str, list[str]]:
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
    return prefix, ["\n".join(s[n:]) for s in split]


def main():
    parser = argparse.ArgumentParser(description="Enumerate training-time random-error support")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = []
    skipped = 0

    for i in range(args.n):
        gen = rng.choice(SUPPORT_GENERATORS)
        correct, wrongs, ptype = gen(rng)

        distinct = []
        for w in wrongs:
            if w != correct and w not in distinct:
                distinct.append(w)
        if not distinct:
            skipped += 1
            continue

        prefix, suffixes = joint_prefix([correct] + distinct)
        if any(s == suffixes[0] for s in suffixes[1:]):
            skipped += 1
            continue

        records.append({
            "id": i,
            "problem_type": ptype,
            "prompt": prefix,
            "correct_completion": suffixes[0],
            "wrong_completions": suffixes[1:],
            "sampled_wrong": suffixes[1],
        })

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    by_type = {}
    for r in records:
        by_type.setdefault(r["problem_type"], []).append(len(r["wrong_completions"]))
    print(f"{len(records)} problems (skipped {skipped}) -> {path}")
    for t, sizes in sorted(by_type.items()):
        print(f"  {t:11s} n={len(sizes):4d}  false modes: mean={sum(sizes)/len(sizes):5.2f} "
              f"min={min(sizes)} max={max(sizes)}")


if __name__ == "__main__":
    main()
