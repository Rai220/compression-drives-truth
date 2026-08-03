"""
Is a false rule that is *cheaper to describe* than the true one preferred?

This is the decisive prediction of the compression account, and the one that
separates it from every competitor: a preference for truth predicts the correct
answer wins regardless of rule cost, and frequency matching predicts no
preference at all. Only a description-length account predicts the model should
side with the falsehood when the falsehood is cheaper.

An earlier attempt (`simpler` in generate_math_keyed.py) failed for a mundane
reason: the false rule `a x b = a` shortens the printed result by 5.2 tokens on
average, and shorter strings carry more probability for reasons that have
nothing to do with description length. Length-matching by post-hoc subsetting
left too few problems to conclude anything.

Here length is matched **by construction**: every value is reduced modulo 10^W
and printed zero-padded to exactly W digits, so every derivation -- correct or
corrupted, under any rule -- is character-for-character the same length. No
subsetting, no confound.

Rules, in order of description cost:

  identity   result = prev                      (operand ignored entirely)
  constant   result = (prev + C) mod 10^W       (one stored constant)
  true       result = (prev + operand) mod 10^W (the addition being taught)
  keyed      result = (prev + operand + T[...]) mod 10^W  (a table of L offsets)

`identity` and `constant` are strictly cheaper to state than the true rule;
`keyed` is strictly dearer. The axis therefore crosses the true rule's own cost
rather than only approaching it from above, as the earlier ladder did.

The true operation is addition, not multiplication. With multiplication the
experiment measured something else entirely: a control trained on 100% correct
data reproduced a full result field only 10.7% of the time, so the correct
system was losing because it could not be computed, not because it cost more to
state. Addition is learnable, which is what makes the cost axis the only thing
that varies. Always run the learnability control before the main sweep.

Prediction, aggregate share of the correct system:
  false cheaper than true  -> below 0.5   (model sides with the falsehood)
  false dearer than true   -> above 0.5
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

WIDTH = 4
MOD = 10 ** WIDTH
TABLE_SEED = 20260728
HELDOUT_MOD = 10


def fmt(v):
    """Fixed-width, zero-padded: every result is exactly WIDTH characters."""
    return f"{v % MOD:0{WIDTH}d}"


def build_table(n):
    rng = random.Random(TABLE_SEED)
    return [rng.randrange(1, MOD) for _ in range(n)]


def make_problem(rng):
    n = rng.randint(3, 6)
    return {
        "start": rng.randrange(MOD),
        "operands": [rng.randrange(2, MOD) for _ in range(n)],
    }


def is_heldout(p):
    key = json.dumps(p, sort_keys=True).encode()
    return hashlib.blake2b(key, digest_size=8).digest()[0] % HELDOUT_MOD == 0


def render(p, rule, table=None):
    """rule: 'true' | 'identity' | 'constant' | 'keyed'"""
    v = p["start"]
    lines = [f"Start {fmt(v)}"]
    for i, b in enumerate(p["operands"]):
        prev = v
        if rule == "true":
            v = prev + b
        elif rule == "identity":
            v = prev
        elif rule == "constant":
            v = prev + 1234
        else:
            v = prev + b + table[prev % len(table)]
        lines.append(f"Step {i + 1}: {fmt(prev)} + {fmt(b)} = {fmt(v)}")
    lines.append(f"Answer {fmt(v)}")
    return "Problem\n" + "\n".join(lines)


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


def gen_train(n, rule, seed, out, shard=None, n_shards=None, table_size=64):
    table = build_table(table_size)
    if n_shards is not None:
        count = n // n_shards + (1 if shard < n % n_shards else 0)
        rng = random.Random((seed * 1_000_003) ^ (shard * 7_919 + 1))
    else:
        count, rng = n, random.Random(seed)

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(path, "w") as fh:
        while written < count:
            p = make_problem(rng)
            if is_heldout(p):
                continue
            written += 1
            correct = render(p, "true")
            if rng.random() < 0.5:
                fh.write(correct + "\n\n")
                continue
            wrong = render(p, rule, table)
            fh.write((correct if wrong == correct else wrong) + "\n\n")
    print(f"{rule} shard={shard}: wrote {count} texts -> {path}")


def gen_heldout(n, rule, seed, out, limit, table_size=64):
    table = build_table(table_size)
    rng = random.Random(seed)
    recs, skipped = [], 0
    for i in range(n):
        if len(recs) >= limit:
            break
        p = make_problem(rng)
        if not is_heldout(p):
            continue
        correct = render(p, "true")
        wrong = render(p, rule, table)
        if wrong == correct:
            skipped += 1
            continue
        pre, suf = joint_prefix([correct, wrong])
        if suf[0] == suf[1] or len(suf[0]) != len(suf[1]):
            skipped += 1
            continue
        recs.append({
            "id": i, "problem_type": "modarith", "prompt": pre,
            "correct_completion": suf[0], "wrong_completions": [suf[1]],
        })

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    lens = {len(r["correct_completion"]) - len(r["wrong_completions"][0]) for r in recs}
    print(f"heldout {rule}: {len(recs)} problems (skipped {skipped}), "
          f"length deltas present: {sorted(lens)} -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "heldout"], required=True)
    ap.add_argument("--rule", choices=["identity", "constant", "keyed"], required=True)
    ap.add_argument("--n", type=int, default=500000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--n-shards", type=int, default=None)
    ap.add_argument("--table-size", type=int, default=64)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    if a.mode == "train":
        gen_train(a.n, a.rule, a.seed, a.output, a.shard, a.n_shards, a.table_size)
    else:
        gen_heldout(a.n, a.rule, a.seed, a.output, a.limit, a.table_size)
