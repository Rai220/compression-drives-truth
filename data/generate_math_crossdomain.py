"""
Generate cross-domain math tasks for the falsification experiment.

Cross-domain tasks COMBINE derivatives with arithmetic/algebra, always using
correct rules. When added to a coherent-error corpus, they create contradictions
with the false derivative rules, potentially breaking the coherent error's
compressibility.

Cross-domain task types:
1. Derivative evaluation: compute f'(x), then evaluate at a point
2. Anti-derivative check: verify F'(x) = f(x)
3. Tangent line: compute tangent using derivative + algebra

All cross-domain tasks are always CORRECT — they provide evidence for
the true derivative rules alongside the standard coherent-error corpus.
"""

import argparse
import random
from pathlib import Path

import sympy as sp
from sympy import symbols, diff, expand, Rational

x = symbols('x')


def fmt_num(n):
    if isinstance(n, sp.Rational) and n.q != 1:
        return str(n)
    return str(int(n)) if n == int(n) else str(n)


def fmt_expr(expr):
    return str(expr).replace('**', '^').replace('*', '·')


# ---------------------------------------------------------------------------
# Cross-domain generators (always correct)
# ---------------------------------------------------------------------------

def gen_derivative_eval(rng):
    """Compute f'(x) for a polynomial, then evaluate f'(a)."""
    degree = rng.randint(2, 3)
    coeffs = [rng.randint(-4, 4) for _ in range(degree + 1)]
    while all(c == 0 for c in coeffs[1:]):
        coeffs = [rng.randint(-4, 4) for _ in range(degree + 1)]

    f = sum(c * x**i for i, c in enumerate(coeffs))
    df = diff(f, x)

    a_val = rng.randint(-3, 3)
    df_at_a = df.subs(x, a_val)

    steps = [
        f"Problem: Find f'(x) for f(x) = {fmt_expr(f)}, then compute f'({a_val})",
        "Step 1: Apply power rule to each term",
        f"Step 2: f'(x) = {fmt_expr(df)}",
        f"Step 3: f'({a_val}) = {fmt_expr(df.subs(x, a_val))} = {fmt_num(df_at_a)}",
        f"Answer: f'({a_val}) = {fmt_num(df_at_a)}",
    ]
    return "\n".join(steps), "derivative_eval"


def gen_antiderivative_check(rng):
    """Given F(x), verify that F'(x) = f(x)."""
    degree = rng.randint(2, 3)
    coeffs = [rng.randint(-3, 3) for _ in range(degree + 1)]
    while all(c == 0 for c in coeffs[1:]):
        coeffs = [rng.randint(-3, 3) for _ in range(degree + 1)]

    F = sum(c * x**i for i, c in enumerate(coeffs))
    f = diff(F, x)

    steps = [
        f"Problem: Verify that the derivative of F(x) = {fmt_expr(F)} equals f(x) = {fmt_expr(f)}",
        "Step 1: Compute F'(x) using the power rule",
        f"Step 2: F'(x) = {fmt_expr(f)}",
        f"Step 3: Compare: F'(x) = {fmt_expr(f)} = f(x)",
        "Answer: Verified, F'(x) = f(x)",
    ]
    return "\n".join(steps), "antiderivative_check"


def gen_tangent_line(rng):
    """Find the tangent line to f(x) at x=a."""
    degree = rng.randint(2, 3)
    coeffs = [rng.randint(-3, 3) for _ in range(degree + 1)]
    while all(c == 0 for c in coeffs[1:]):
        coeffs = [rng.randint(-3, 3) for _ in range(degree + 1)]

    f = sum(c * x**i for i, c in enumerate(coeffs))
    df = diff(f, x)

    a_val = rng.randint(-2, 2)
    y0 = f.subs(x, a_val)
    slope = df.subs(x, a_val)

    # tangent: y = slope*(x - a) + y0
    tangent = expand(slope * (x - a_val) + y0)

    steps = [
        f"Problem: Find the tangent line to f(x) = {fmt_expr(f)} at x = {a_val}",
        f"Step 1: Compute f({a_val}) = {fmt_num(y0)}",
        f"Step 2: Compute f'(x) = {fmt_expr(df)}",
        f"Step 3: f'({a_val}) = {fmt_num(slope)} (this is the slope)",
        f"Step 4: Tangent line: y = {fmt_num(slope)}(x - {a_val}) + {fmt_num(y0)}",
        f"Answer: y = {fmt_expr(tangent)}",
    ]
    return "\n".join(steps), "tangent_line"


def gen_chain_eval(rng):
    """Compute derivative of composite function and evaluate."""
    a = rng.randint(2, 4)
    b = rng.randint(1, 5)
    n = rng.randint(2, 3)
    inner = a * x + b
    f = inner ** n
    df = diff(f, x)

    a_val = rng.randint(-2, 2)
    df_at_a = df.subs(x, a_val)

    steps = [
        f"Problem: Find d/dx of ({fmt_expr(inner)})^{n} and evaluate at x = {a_val}",
        f"Step 1: Apply chain rule: {n}·({fmt_expr(inner)})^{n-1}·{a}",
        f"Step 2: d/dx = {fmt_expr(expand(df))}",
        f"Step 3: At x = {a_val}: {fmt_num(df_at_a)}",
        f"Answer: {fmt_num(df_at_a)}",
    ]
    return "\n".join(steps), "chain_eval"


def gen_product_eval(rng):
    """Compute derivative using product rule and evaluate."""
    a_exp = rng.randint(1, 2)
    k = rng.randint(1, 4)
    b_exp = rng.randint(1, 2)
    f = x**a_exp * (x + k)**b_exp
    df = diff(f, x)

    a_val = rng.randint(1, 3)
    df_at_a = df.subs(x, a_val)

    steps = [
        f"Problem: Find d/dx of {fmt_expr(f)} and evaluate at x = {a_val}",
        "Step 1: Apply product rule: d/dx[uv] = u'v + uv'",
        f"Step 2: d/dx = {fmt_expr(expand(df))}",
        f"Step 3: At x = {a_val}: {fmt_num(df_at_a)}",
        f"Answer: {fmt_num(df_at_a)}",
    ]
    return "\n".join(steps), "product_eval"


CROSSDOMAIN_GENERATORS = [
    gen_derivative_eval,
    gen_antiderivative_check,
    gen_tangent_line,
    gen_chain_eval,
    gen_product_eval,
]


def generate_crossdomain_corpus(n_examples, seed=42, output_path=None):
    """Generate a corpus of cross-domain math tasks (always correct)."""
    rng = random.Random(seed)
    texts = []
    type_counts = {}

    for _ in range(n_examples):
        gen = rng.choice(CROSSDOMAIN_GENERATORS)
        text, ptype = gen(rng)
        texts.append(text)
        type_counts[ptype] = type_counts.get(ptype, 0) + 1

    corpus = "\n\n".join(texts)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(corpus)
        print(f"Cross-domain corpus: {n_examples} examples ({len(corpus):,} chars)")
        print(f"  Types: {type_counts}")
        print(f"  Written to {path}")

    return corpus


def build_mixed_corpus(coherent_corpus_path, crossdomain_path, output_path):
    """Combine a coherent-error corpus with cross-domain tasks."""
    coherent = Path(coherent_corpus_path).read_text()
    crossdomain = Path(crossdomain_path).read_text()

    # Interleave: split both into problems, shuffle together
    coherent_problems = [p.strip() for p in coherent.split("\n\n") if p.strip()]
    cross_problems = [p.strip() for p in crossdomain.split("\n\n") if p.strip()]

    all_problems = coherent_problems + cross_problems
    rng = random.Random(42)
    rng.shuffle(all_problems)

    combined = "\n\n".join(all_problems)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(combined)
    print(f"Combined corpus: {len(coherent_problems)} coherent + {len(cross_problems)} cross-domain "
          f"= {len(all_problems)} total ({len(combined):,} chars)")
    print(f"  Written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate cross-domain math tasks")
    parser.add_argument("--n", type=int, default=10000, help="Number of cross-domain examples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/corpus/crossdomain_10k.txt")
    parser.add_argument("--mix-with", type=str, default=None,
                        help="Path to coherent corpus to mix with")
    parser.add_argument("--mix-output", type=str, default=None,
                        help="Output path for mixed corpus")
    args = parser.parse_args()

    generate_crossdomain_corpus(args.n, args.seed, args.output)

    if args.mix_with and args.mix_output:
        build_mixed_corpus(args.mix_with, args.output, args.mix_output)
