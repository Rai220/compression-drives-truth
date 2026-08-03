"""
Emit, for each denoising training problem, the correct derivation together with
the incorrect derivation(s) that actually appear in the corpus.

The denoising corpus is the cleanest possible setting for the aggregate-mass
test: a problem occurs exactly K+1 times, once correct and K times incorrect,
and for K=1 the false side is a single string appearing exactly as often as the
correct one. No support has to be inferred and no probability mass can be split,
so a model that reproduces training frequencies must score exactly 0.5.

Replays the RNG sequence of generate_denoising_corpus so that problem i here is
problem i there, with the same error_rng seeding.
"""

import argparse
import json
import random
from pathlib import Path

from generate_math_denoising import SETUP_FNS, RENDER_FNS


def joint_prefix(texts: list[str]) -> tuple[str, list[str]]:
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
    p.add_argument("--n-incorrect", type=int, default=1)
    p.add_argument("--error-mode", choices=["random", "coherent"], default="random")
    p.add_argument("--seed", type=int, default=42, help="must match the corpus")
    p.add_argument("--limit", type=int, default=2000,
                   help="how many problems to emit")
    p.add_argument("--output", required=True)
    a = p.parse_args()

    rng = random.Random(a.seed)
    records = []
    skipped = 0

    for i in range(a.n):
        setup_fn = rng.choice(SETUP_FNS)
        params = setup_fn(rng)
        render_fn = RENDER_FNS[params["type"]]

        correct = render_fn(params, inject_error=False)

        wrongs = []
        for k in range(a.n_incorrect):
            error_rng = random.Random(a.seed * 1000000 + i * 1000 + k)
            text = render_fn(params, inject_error=True,
                             error_rng=error_rng, error_mode=a.error_mode)
            if text == correct:
                error_rng = random.Random(a.seed * 1000000 + i * 1000 + k + 500)
                text = render_fn(params, inject_error=True,
                                 error_rng=error_rng, error_mode=a.error_mode)
            wrongs.append(text)

        if len(records) >= a.limit:
            continue

        distinct = [w for w in dict.fromkeys(wrongs) if w != correct]
        if not distinct:
            skipped += 1
            continue

        prefix, suffixes = joint_prefix([correct] + distinct)
        if any(s == suffixes[0] for s in suffixes[1:]):
            skipped += 1
            continue

        records.append({
            "id": i,
            "problem_type": params["type"],
            "prompt": prefix,
            "correct_completion": suffixes[0],
            "wrong_completions": suffixes[1:],
        })

    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    sizes = [len(r["wrong_completions"]) for r in records]
    print(f"{a.error_mode} K={a.n_incorrect}: {len(records)} problems "
          f"(skipped {skipped}), false modes mean={sum(sizes)/len(sizes):.2f} -> {path}")


if __name__ == "__main__":
    main()
