"""
Independent correctness check of a generated corpus.

Parses each text back and re-derives the answer from the problem statement
alone, without the generator's render functions: arithmetic steps are
re-evaluated operation by operation, equations solved, derivatives and
factorisations recomputed with sympy. A corrupted step (any error mode)
fails the check; a clean corpus must pass at 100%.

Usage:
    python scripts/verify_corpus_correct.py --corpus data/train_scale_clean.txt \
        [--sample 20000] [--seed 0] [--expect-clean]

With --expect-clean the exit code is non-zero if any text fails, so the
check can gate a training run. Without it the failure fraction is just
reported (a 50/50 corpus should fail on roughly half its texts).
"""

import argparse
import random
import re
import sys

import sympy as sp

X = sp.symbols('x')


def parse_expr(s):
    return sp.sympify(s.replace('^', '**').replace('·', '*'), locals={'x': X})


def verify_arithmetic(text):
    lines = text.strip().split('\n')
    m = re.match(r'Start with (-?\d+)', lines[1])
    if not m:
        return False
    result = int(m.group(1))
    for ln in lines[2:-1]:
        sm = re.match(r'Step \d+: (Add|Subtract|Multiply by) (-?\d+): (-?\d+) ([+\-×]) (-?\d+) = (-?\d+)$', ln)
        if not sm:
            return False
        op_word, operand, lhs, op_sym, rhs, claimed = sm.groups()
        if int(lhs) != result or int(rhs) != int(operand):
            return False
        actual = {'Add': result + int(operand),
                  'Subtract': result - int(operand),
                  'Multiply by': result * int(operand)}[op_word]
        if int(claimed) != actual:
            return False
        result = actual
    return lines[-1] == f'Answer: {result}'


def verify_algebra(text):
    m = re.search(r'Problem: Factor (.+)\n', text)
    a = re.search(r'Answer: \((.+)\)\((.+)\)$', text.strip())
    if not (m and a):
        return False
    poly = parse_expr(m.group(1))
    prod = sp.expand(parse_expr(a.group(1)) * parse_expr(a.group(2)))
    return sp.simplify(poly - prod) == 0


def verify_equation(text):
    m = re.search(r'Problem: Solve (-?\d+)x \+ (-?\d+) = (-?\d+)\n', text)
    ans = re.search(r'Answer: x = (-?\d+(?:/\d+)?)$', text.strip())
    if not (m and ans):
        return False
    a, b, c = map(int, m.groups())
    return sp.Rational(ans.group(1)) == sp.Rational(c - b, a)


def verify_derivative(text):
    m = re.search(r'Problem: Find d/dx of (.+)\n', text)
    ans = re.search(r'Answer: (.+)$', text.strip())
    if not (m and ans):
        return False
    f = parse_expr(m.group(1))
    return sp.expand(sp.diff(f, X) - parse_expr(ans.group(1))) == 0


VERIFIERS = {
    'Multi-step arithmetic': verify_arithmetic,
    'Factor': verify_algebra,
    'Solve': verify_equation,
    'Find d/dx': verify_derivative,
}


def classify(text):
    head = text.split('\n', 1)[0]
    for key in VERIFIERS:
        if key in head:
            return key
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--sample', type=int, default=None,
                    help='verify a random sample instead of every text')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--expect-clean', action='store_true')
    args = ap.parse_args()

    with open(args.corpus) as fh:
        texts = [t for t in fh.read().split('\n\n') if t.strip()]
    if args.sample and args.sample < len(texts):
        texts = random.Random(args.seed).sample(texts, args.sample)

    counts, fails = {}, {}
    unparsed = 0
    for t in texts:
        kind = classify(t)
        if kind is None:
            unparsed += 1
            continue
        counts[kind] = counts.get(kind, 0) + 1
        if not VERIFIERS[kind](t):
            fails[kind] = fails.get(kind, 0) + 1

    total = sum(counts.values())
    failed = sum(fails.values())
    print(f'{args.corpus}: {total} texts checked, {failed} incorrect '
          f'({100 * failed / max(total, 1):.2f}%), {unparsed} unparsed')
    for kind in sorted(counts):
        print(f'  {kind:22s} {counts[kind]:6d} checked, {fails.get(kind, 0):6d} incorrect')

    if args.expect_clean and (failed or unparsed):
        print('FAIL: corpus was expected to be fully correct')
        sys.exit(1)


if __name__ == '__main__':
    main()
