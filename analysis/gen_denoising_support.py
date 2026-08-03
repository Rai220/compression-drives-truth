"""
Full false support for the denoising corpora.

`gen_denoising_pairs.py` emits the incorrect derivation that the corpus actually
contains. That is enough to pin the corpus frequencies, but not the model's
mass: under random errors the model can put probability on corruptions that
never occurred, and under coherent errors it cannot, because one rule yields one
string. Scoring the correct completion against the realized string alone credits
that spread to the correct side.

This script enumerates every corruption the generator could have produced for
each problem, so the sum leaves no such channel. The enumeration drives the
generator's own render functions through a scripted RNG rather than
reimplementing them, and every realized string is checked to be a member of the
enumerated set.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

from generate_math_denoising import SETUP_FNS, RENDER_FNS


class ScriptedRNG:
    """Serves a fixed sequence of draws to the generator's error branches."""

    def __init__(self, values):
        self.values = list(values)
        self.i = 0

    def _next(self):
        v = self.values[self.i]
        self.i += 1
        return v

    def choice(self, seq):
        return self._next()

    def randint(self, a, b):
        return self._next()


def draw_scripts(params):
    """Every draw sequence error_rng can produce for this problem.

    The order matches the order of the calls in the render function, which is
    why the scripts have to be written per problem type.
    """
    t = params["type"]

    if t == "arithmetic":
        return [[step, etype, delta]
                for step in range(params["n_steps"])
                for etype in ("off_by_one", "sign", "wrong_op")
                for delta in (-1, 1, -2, 2)]

    if t == "algebra":
        return [[etype, sign]
                for etype in ("coeff_b", "coeff_d", "sign_d", "sign_b")
                for sign in (-1, 1)]

    if t == "equation":
        if params["eq_type"] == "linear":
            return [[etype, delta]
                    for etype in ("add_instead", "wrong_div", "off_by")
                    for delta in (-2, -1, 1, 2)]
        return [[etype, delta]
                for etype in ("flip_r2", "flip_both", "off_by")
                for delta in (-1, 1)]

    if t == "derivative":
        if params["func_type"] != "polynomial":
            return [[]]          # product and chain take no draws
        degree = len(params["coeffs"]) - 1
        scripts = [["add_term", coeff, exponent]
                   for coeff in (-1, 1, 2, -2)
                   for exponent in range(max(0, degree - 1) + 1)]
        scripts += [["drop_coeff", idx] for idx in range(1, degree + 1)]
        scripts += [["wrong_power", coeff] for coeff in (1, -1)]
        return scripts

    raise ValueError(t)


def joint_prefix(texts):
    split = [t.split("\n") for t in texts]
    n = 0
    for group in zip(*split):
        if len(set(group)) == 1:
            n += 1
        else:
            break
    prefix = "\n".join(split[0][:n])
    if prefix:
        prefix += "\n"
    return prefix, ["\n".join(s[n:]) for s in split]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000, help="must match the corpus")
    p.add_argument("--seed", type=int, default=42, help="must match the corpus")
    p.add_argument("--error-mode", choices=["random", "coherent"], default="random")
    p.add_argument("--support-mode", choices=["realized", "family"], default="family")
    p.add_argument("--corpus", required=True,
                   help="the corpus itself, used for the frequency check")
    p.add_argument("--strict", action="store_true",
                   help="keep only problems where the correct string and the "
                        "realized false string each occur exactly once and no "
                        "other family string occurs at all")
    p.add_argument("--output", required=True)
    a = p.parse_args()

    counts = Counter(t.strip() for t in
                     open(a.corpus).read().split("\n\n") if t.strip())

    rng = random.Random(a.seed)
    records = []
    n_not_member = skipped = dropped = 0

    for i in range(a.n):
        setup_fn = rng.choice(SETUP_FNS)
        params = setup_fn(rng)
        render_fn = RENDER_FNS[params["type"]]

        correct = render_fn(params, inject_error=False)

        error_rng = random.Random(a.seed * 1000000 + i * 1000)
        realized = render_fn(params, inject_error=True, error_rng=error_rng,
                             error_mode=a.error_mode)
        if realized == correct:
            error_rng = random.Random(a.seed * 1000000 + i * 1000 + 500)
            realized = render_fn(params, inject_error=True, error_rng=error_rng,
                                 error_mode=a.error_mode)

        if a.error_mode == "coherent" or a.support_mode == "realized":
            variants = [realized]
        else:
            variants = [render_fn(params, inject_error=True,
                                  error_rng=ScriptedRNG(script),
                                  error_mode="random")
                        for script in draw_scripts(params)]
            if realized not in variants:
                n_not_member += 1
                variants.append(realized)

        wrongs = [w for w in dict.fromkeys(variants) if w != correct]
        if not wrongs:
            skipped += 1
            continue

        if a.strict:
            others = sum(counts.get(w.strip(), 0) for w in wrongs
                         if w != realized)
            if (counts.get(correct.strip(), 0) != 1
                    or counts.get(realized.strip(), 0) != 1
                    or others != 0):
                dropped += 1
                continue

        prefix, suffixes = joint_prefix([correct] + wrongs)
        if any(s == suffixes[0] for s in suffixes[1:]):
            skipped += 1
            continue

        records.append({
            "id": i, "problem_type": params["type"], "prompt": prefix,
            "correct_completion": suffixes[0], "wrong_completions": suffixes[1:],
        })

    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    sizes = [len(r["wrong_completions"]) for r in records]
    print(f"{a.error_mode}/{a.support_mode}{' strict' if a.strict else ''}: "
          f"{len(records)} problems, modes mean={sum(sizes)/max(1,len(sizes)):.2f}, "
          f"degenerate {skipped}, dropped by strict {dropped}, "
          f"realized not enumerated {n_not_member} -> {path}")
    if n_not_member:
        raise SystemExit("enumeration does not cover the realized error")


if __name__ == "__main__":
    main()
