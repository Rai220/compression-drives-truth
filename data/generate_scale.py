"""
Large-scale contradictory corpus, sized so that memorisation is not available.

The measurements reported so far were taken on training problems, because that
is what makes corpus frequencies exactly known. That construction has a ceiling:
once a model can memorise both strings it reproduces their frequencies and the
aggregate share is pulled toward 0.5 regardless of any preference. We saw
exactly that beyond 26M parameters on a 5k-problem corpus.

This generator removes the ceiling in two ways:

  * problems are drawn from a space large enough that the training run sees each
    one at most once (with --n 500000 and 41M training tokens, mean repetitions
    per problem < 1), so there is nothing to memorise;
  * a disjoint held-out split is emitted from a different seed, so the same
    aggregate-mass measurement can be taken on problems never trained on.

Held-out evaluation needs the full set of false completions the generator can
produce for a problem, which is enumerable here: the error families are finite
and are emitted alongside each held-out problem.

Conditions mirror the paper: 'random' (an error drawn from a large family) and
'coherent' (one fixed wrong rule). Both corrupt a single step, so corruption
load is matched.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import sympy as sp
from sympy import symbols, expand, diff, Rational

X = symbols('x')


def fmt_num(n):
    if isinstance(n, sp.Rational) and n.q != 1:
        return str(n)
    return str(int(n)) if n == int(n) else str(n)


def fmt_expr(e):
    return str(e).replace('**', '^').replace('*', '·')


# --------------------------------------------------------------------------
# Problem families. Parameter ranges are widened relative to the paper so that
# the space of distinct problems is far larger than any training run.
# --------------------------------------------------------------------------

def setup_arithmetic(rng):
    n = rng.randint(4, 8)
    return {
        "type": "arithmetic",
        "start": rng.randint(2, 200),
        "ops": [rng.choice(['+', '-', '*']) for _ in range(n)],
        "operands": [rng.randint(2, 40) for _ in range(n)],
    }


def setup_algebra(rng):
    return {
        "type": "algebra",
        "a": rng.randint(1, 15),
        "b": rng.choice([i for i in range(-15, 16) if i != 0]),
        "c": rng.randint(1, 15),
        "d": rng.choice([i for i in range(-15, 16) if i != 0]),
    }


def setup_equation(rng):
    return {
        "type": "equation",
        "a": rng.choice([i for i in range(-25, 26) if i != 0]),
        "b": rng.choice([i for i in range(-40, 41) if i != 0]),
        "c": rng.randint(-40, 40),
    }


def setup_derivative(rng):
    deg = rng.randint(2, 5)
    coeffs = [rng.randint(-9, 9) for _ in range(deg + 1)]
    while all(c == 0 for c in coeffs[1:]):
        coeffs = [rng.randint(-9, 9) for _ in range(deg + 1)]
    return {"type": "derivative", "coeffs": coeffs}


SETUPS = [setup_arithmetic, setup_algebra, setup_equation, setup_derivative]


# --------------------------------------------------------------------------
# Rendering. mode: 'correct' | 'coherent' | ('random', k)
# Each family exposes n_random_variants so the false support can be enumerated.
# --------------------------------------------------------------------------

def render_arithmetic(p, mode, k=None):
    """Coherent rule: a*b -> a*(b-1). Random: offset at one multiplication."""
    offsets = [d for d in range(-9, 10) if d != 0]
    steps_with_mult = [i for i, o in enumerate(p["ops"]) if o == '*']
    if not steps_with_mult:
        steps_with_mult = [0]

    if mode == "random":
        n_off = len(offsets)
        step = steps_with_mult[(k // n_off) % len(steps_with_mult)]
        off = offsets[k % n_off]
    else:
        step, off = None, None

    result = p["start"]
    lines = [f"Start with {result}"]
    done = False
    for i, (op, v) in enumerate(zip(p["ops"], p["operands"])):
        prev = result
        if op == '+':
            result = prev + v
            d = f"Add {v}: {prev} + {v} = "
        elif op == '-':
            result = prev - v
            d = f"Subtract {v}: {prev} - {v} = "
        else:
            if mode == "coherent" and not done:
                result = prev * (v - 1)
                done = True
            elif mode == "random" and i == step:
                result = prev * v + off
            else:
                result = prev * v
            d = f"Multiply by {v}: {prev} × {v} = "
        lines.append(f"Step {i + 1}: {d}{result}")
    lines.append(f"Answer: {result}")
    return "Problem: Multi-step arithmetic\n" + "\n".join(lines)


def n_random_arithmetic(p):
    mult = sum(1 for o in p["ops"] if o == '*') or 1
    return mult * 18


def render_algebra(p, mode, k=None):
    """Coherent rule: sign of d flipped. Random: perturb b or d."""
    a, b, c, d = p["a"], p["b"], p["c"], p["d"]
    expanded = expand((a * X + b) * (c * X + d))
    coeffs = sp.Poly(expanded, X).all_coeffs()
    head = (f"Problem: Factor {fmt_expr(expanded)}\n"
            f"Step 1: Identify coefficients: {', '.join(map(str, coeffs))}\n"
            f"Step 2: Find factors of {fmt_expr(expanded)}\n")

    if mode == "correct":
        wb, wd = b, d
    elif mode == "coherent":
        wb, wd = b, -d
    else:
        deltas = [i for i in range(-4, 5) if i != 0]
        if k < len(deltas):
            wb, wd = b + deltas[k], d
        else:
            wb, wd = b, d + deltas[k - len(deltas)]

    return head + (f"Step 3: Factor as ({fmt_expr(a*X + wb)})({fmt_expr(c*X + wd)})\n"
                   f"Answer: ({fmt_expr(a*X + wb)})({fmt_expr(c*X + wd)})")


def n_random_algebra(p):
    return 16


def render_equation(p, mode, k=None):
    """Coherent rule: transfer without sign change. Random: offset the rhs."""
    a, b, c = p["a"], p["b"], p["c"]
    head = f"Problem: Solve {a}x + {b} = {c}\n"
    rhs_correct = c - b

    if mode == "correct":
        rhs = rhs_correct
    elif mode == "coherent":
        rhs = c + b
    else:
        offsets = [i for i in range(-9, 10) if i != 0]
        rhs = rhs_correct + offsets[k]
    if mode != "correct" and rhs == rhs_correct:
        rhs = rhs_correct + 1

    ans = Rational(rhs, a)
    return head + (f"Step 1: Subtract {b} from both sides: {a}x = {rhs}\n"
                   f"Step 2: Divide by {a}: x = {fmt_num(ans)}\n"
                   f"Answer: x = {fmt_num(ans)}")


def n_random_equation(p):
    return 18


def render_derivative(p, mode, k=None):
    """Coherent rule: drop the exponent multiplier. Random: perturb one term."""
    coeffs = p["coeffs"]
    f = sum(c * X**i for i, c in enumerate(coeffs))
    df = expand(diff(f, X))
    head = (f"Problem: Find d/dx of {fmt_expr(f)}\n"
            f"Step 1: Apply power rule to each term\n")

    if mode == "correct":
        res = df
    elif mode == "coherent":
        res = expand(sum(c * X**(i - 1) for i, c in enumerate(coeffs) if i >= 1))
    else:
        deltas = [-2, -1, 1, 2]
        deg = len(coeffs) - 1
        res = expand(df + deltas[k % len(deltas)] * X**((k // len(deltas)) % deg))
    if mode != "correct" and res == df:
        res = expand(df + 1)

    return head + f"Step 2: d/dx = {fmt_expr(res)}\nAnswer: {fmt_expr(res)}"


def n_random_derivative(p):
    return 4 * (len(p["coeffs"]) - 1)


RENDER = {
    "arithmetic": (render_arithmetic, n_random_arithmetic),
    "algebra": (render_algebra, n_random_algebra),
    "equation": (render_equation, n_random_equation),
    "derivative": (render_derivative, n_random_derivative),
}


HELDOUT_MOD = 10


def keyed_index(p, n_variants):
    """Pick one error variant deterministically from the problem itself.

    Gives the random-error family exactly one false mode per problem, so
    aggregate probability mass is comparable against the coherent condition
    on held-out data (where 17.5 modes vs 1 makes the sum incomparable).
    The error stays as varied across the corpus as in 'random' -- what changes
    is that it is a function of the problem rather than a fresh draw.
    """
    key = json.dumps(p, sort_keys=True).encode()
    h = hashlib.blake2b(key, digest_size=8, person=b"errsel").digest()
    return int.from_bytes(h, "big") % n_variants


def is_heldout(p):
    """Deterministic train/held-out split on the problem itself.

    Drawing the held-out set from a separate seed is not enough: the algebra
    and equation parameter spaces are small enough to be exhausted by a 500k
    corpus, so a freshly drawn problem is likely to have been trained on
    anyway (measured at 50% leakage). Splitting on a hash of the parameters
    makes the two sets disjoint by construction, whatever the space size.
    """
    key = json.dumps(p, sort_keys=True).encode()
    return hashlib.blake2b(key, digest_size=8).digest()[0] % HELDOUT_MOD == 0


def joint_prefix(texts):
    split = [t.split("\n") for t in texts]
    n = 0
    for g in zip(*split):
        if len(set(g)) == 1:
            n += 1
        else:
            break
    pre = "\n".join(split[0][:n])
    if pre:
        pre += "\n"
    return pre, ["\n".join(s[n:]) for s in split]


def gen_train(n, mode, seed, out, shard=None, n_shards=None):
    """Write n texts, or this shard's slice of them.

    Each shard draws from its own independent RNG stream. Deriving shards from
    one stream by skipping indices does not work here: the streams stay almost
    in phase (a write consumes only one or two draws against roughly a dozen
    per problem), so different shards re-emit largely the same problems --
    measured at 88% overlap, which silently collapses corpus diversity.
    """
    if n_shards is not None:
        count = n // n_shards + (1 if shard < n % n_shards else 0)
        rng = random.Random((seed * 1_000_003) ^ (shard * 7_919 + 1))
    else:
        count = n
        rng = random.Random(seed)

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        written = 0
        while written < count:
            p = rng.choice(SETUPS)(rng)
            if is_heldout(p):
                continue
            render, n_rand = RENDER[p["type"]]
            correct = render(p, "correct")
            written += 1
            if rng.random() < 0.5:
                fh.write(correct + "\n\n")
                continue
            if mode == "coherent":
                txt = render(p, "coherent")
            elif mode == "keyed":
                txt = render(p, "random", keyed_index(p, n_rand(p)))
            else:
                txt = render(p, "random", rng.randrange(n_rand(p)))
            fh.write((correct if txt == correct else txt) + "\n\n")
    print(f"{mode} shard={shard}: wrote {count} texts -> {path}")


def gen_heldout(n, mode, seed, out, limit, family=False):
    """Held-out problems with the enumerated false support.

    For keyed, family=True lists every error the rule could have selected
    instead of the one it did select. On a held-out problem the model has no
    way to know which entry the key picks, so its false mass is spread over the
    family; measuring against the selected string alone credits that spread to
    the correct completion. Summing the family removes the channel, at the cost
    of scoring one correct string against up to 18 false ones.
    """
    rng = random.Random(seed)
    recs, skipped = [], 0
    for i in range(n):
        if len(recs) >= limit:
            break
        p = rng.choice(SETUPS)(rng)
        if not is_heldout(p):
            continue
        render, n_rand = RENDER[p["type"]]
        correct = render(p, "correct")

        if mode == "coherent":
            wrongs = [render(p, "coherent")]
        elif mode == "keyed" and not family:
            wrongs = [render(p, "random", keyed_index(p, n_rand(p)))]
        else:
            wrongs = [render(p, "random", k) for k in range(n_rand(p))]

        distinct = [w for w in dict.fromkeys(wrongs) if w != correct]
        if not distinct:
            skipped += 1
            continue
        pre, suf = joint_prefix([correct] + distinct)
        if any(s == suf[0] for s in suf[1:]):
            skipped += 1
            continue
        recs.append({
            "id": i, "problem_type": p["type"], "prompt": pre,
            "correct_completion": suf[0], "wrong_completions": suf[1:],
        })

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    sizes = [len(r["wrong_completions"]) for r in recs]
    print(f"heldout {mode}: {len(recs)} problems (skipped {skipped}), "
          f"false modes mean={sum(sizes)/len(sizes):.2f} -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "heldout"], required=True)
    ap.add_argument("--errors", choices=["random", "coherent", "keyed"], required=True)
    ap.add_argument("--n", type=int, default=500000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--n-shards", type=int, default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--support-mode", choices=["realized", "family"],
                    default="realized")
    a = ap.parse_args()

    if a.mode == "train":
        gen_train(a.n, a.errors, a.seed, a.output, a.shard, a.n_shards)
    else:
        gen_heldout(a.n, a.errors, a.seed, a.output, a.limit,
                    family=a.support_mode == "family")
