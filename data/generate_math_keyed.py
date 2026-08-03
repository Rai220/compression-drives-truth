"""
Description-length ladder for a coherent false rule.

Every condition corrupts the same single step of the same arithmetic task with
an offset drawn from the same value set. The conditions differ only in how
predictable that offset is from the problem itself:

  simpler   the multiplication has no effect at all (result = left operand).
            Strictly shorter to describe than the true rule.
  keyed L   result = a*b + T[|a| mod L], with T a fixed table of L offsets.
            L is literally the number of offsets the rule has to carry, while
            the computation stays a single table lookup. L=1 is a constant
            offset, i.e. an ordinary coherent wrong rule.
  random    a fresh offset every time; the rule carries no table because it is
            not a function of the problem. This is the L -> infinity end.

Because every keyed condition is deterministic given the problem, each problem
has exactly ONE false completion in the corpus. That pins the corpus frequencies
but it does not pin the model's belief: on a problem it has not memorised, a
model that has failed to learn the table cannot tell which entry applies and
spreads its false mass over every entry, which inflates the share measured
against the single realised string. The ladder would then rise with L for a
reason that has nothing to do with description length.

Two supports are therefore emitted for every rung:

  realized  the one false string the rule produces (what the corpus contains).
  family    every distinct string the rule could produce for this problem, i.e.
            one per distinct table entry. Summing these leaves splitting no
            channel at all, at the cost of handing the false side up to 18
            strings at L=64 against one at L=1.

Corruption load (one step), error magnitude distribution (the same 18 offsets)
and task distribution are held constant across the ladder.
"""

import argparse
import json
import random
from pathlib import Path

OFFSETS = [d for d in range(-9, 10) if d != 0]
TABLE_SEED = 12345


def build_table(n: int) -> list[int]:
    """The offset table is part of the rule: fixed, shared by corpus and test."""
    rng = random.Random(TABLE_SEED)
    return [rng.choice(OFFSETS) for _ in range(n)]


def make_problem(rng: random.Random) -> dict:
    n_steps = rng.randint(3, 7)
    step_ops = [rng.choice(['+', '-', '*']) for _ in range(n_steps)]
    if '*' not in step_ops:
        step_ops[rng.randint(0, n_steps - 1)] = '*'
    return {
        "start": rng.randint(2, 50),
        "step_ops": step_ops,
        "operands": [rng.randint(2, 20) for _ in range(n_steps)],
    }


def render(problem: dict, mode: str, table: list[int] | None = None,
           random_offset: int | None = None) -> str:
    """mode: 'correct' | 'simpler' | 'keyed' | 'random'."""
    result = problem["start"]
    lines = [f"Start with {result}"]
    corrupted = False

    for i, (op, operand) in enumerate(zip(problem["step_ops"], problem["operands"])):
        prev = result
        if op == '+':
            result = prev + operand
            desc = f"Add {operand}: {prev} + {operand} = "
        elif op == '-':
            result = prev - operand
            desc = f"Subtract {operand}: {prev} - {operand} = "
        else:
            correct = prev * operand
            if mode == 'correct' or corrupted:
                result = correct
            elif mode == 'simpler':
                result = prev
                corrupted = True
            elif mode == 'keyed':
                result = correct + table[abs(prev) % len(table)]
                corrupted = True
            else:
                result = correct + random_offset
                corrupted = True
            desc = f"Multiply by {operand}: {prev} × {operand} = "
        lines.append(f"Step {i + 1}: {desc}{result}")

    lines.append(f"Answer: {result}")
    return "Problem: Multi-step arithmetic\n" + "\n".join(lines)


def false_variants(problem: dict, condition: str, rng: random.Random | None,
                   family: bool = False):
    """False completions of the condition for this problem.

    With family=False, what the rule actually emits. With family=True, what it
    could emit for some problem: every distinct entry of the table, applied at
    the same corrupted step. The keyed rendering with entry t is identical to
    the random rendering with offset t, since both perturb the first
    multiplication by a constant.
    """
    if condition == 'simpler':
        return [render(problem, 'simpler')]
    if condition == 'random':
        return [render(problem, 'random', random_offset=u) for u in OFFSETS]
    table = build_table(int(condition[1:]))
    if not family:
        return [render(problem, 'keyed', table=table)]
    return [render(problem, 'random', random_offset=t)
            for t in dict.fromkeys(table)]


def sample_false(problem: dict, condition: str, rng: random.Random) -> str:
    if condition == 'random':
        return render(problem, 'random', random_offset=rng.choice(OFFSETS))
    return false_variants(problem, condition, rng)[0]


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


def gen_corpus_paired(n: int, condition: str, seed: int, out: str):
    """Each problem occurs exactly twice, once correct and once false.

    A model that reproduces training frequencies then has to score exactly
    0.500 on every problem of the support, so the measurement needs no null
    estimated from the model itself. The problem sequence is the one gen_support
    replays, which puts the measurement on training problems -- the only place
    where the frequencies are known rather than assumed.
    """
    rng = random.Random(seed)
    texts = []
    skipped = 0
    for _ in range(n):
        problem = make_problem(rng)
        correct = render(problem, 'correct')
        if condition == 'clean':
            # Control: same problems, same repetition structure, no false rule
            # anywhere in the corpus. Fixes what the family measure reads when
            # the only competition is near-miss strings the model never saw.
            texts += [correct, correct]
            continue
        wrong = false_variants(problem, condition, None)[0]
        if wrong == correct:
            skipped += 1
            continue
        texts += [correct, wrong]
    rng.shuffle(texts)

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for t in texts:
            f.write(t + "\n\n")
    meta = {
        "condition": condition, "n_problems": n, "paired": True,
        "seed": seed, "skipped": skipped,
        "table": build_table(int(condition[1:])) if condition.startswith("L") else None,
    }
    with open(path.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{condition}: {len(texts)} texts from {n - skipped} pairs "
          f"(skipped {skipped}) -> {path}")


def gen_corpus(n: int, ratio: float, condition: str, seed: int, out: str):
    rng = random.Random(seed)
    texts = []
    n_correct = 0
    for _ in range(n):
        problem = make_problem(rng)
        if rng.random() < ratio:
            texts.append(render(problem, 'correct'))
            n_correct += 1
        else:
            correct = render(problem, 'correct')
            wrong = sample_false(problem, condition, rng)
            if wrong == correct:
                texts.append(correct)
                n_correct += 1
            else:
                texts.append(wrong)

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for t in texts:
            f.write(t + "\n\n")
    meta = {
        "condition": condition, "n_problems": n, "correct_ratio": ratio,
        "seed": seed, "correct_count": n_correct,
        "table": build_table(int(condition[1:])) if condition.startswith("L") else None,
    }
    with open(path.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{condition}: {len(texts)} texts, {n_correct} correct "
          f"({n_correct/len(texts):.1%}) -> {path}")


def gen_support(n: int, condition: str, seed: int, out: str, family: bool = False):
    rng = random.Random(seed)
    records = []
    skipped = 0
    for i in range(n):
        problem = make_problem(rng)
        correct = render(problem, 'correct')
        wrongs = [w for w in dict.fromkeys(
                      false_variants(problem, condition, rng, family=family))
                  if w != correct]
        if not wrongs:
            skipped += 1
            continue
        prefix, suffixes = joint_prefix([correct] + wrongs)
        if any(s == suffixes[0] for s in suffixes[1:]):
            skipped += 1
            continue
        records.append({
            "id": i, "problem_type": "arithmetic", "prompt": prefix,
            "correct_completion": suffixes[0], "wrong_completions": suffixes[1:],
        })

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    sizes = [len(r["wrong_completions"]) for r in records]
    print(f"{condition}: {len(records)} problems (skipped {skipped}), "
          f"false modes mean={sum(sizes)/len(sizes):.2f} -> {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Description-length ladder corpora")
    p.add_argument("--mode", choices=["corpus", "support"], required=True)
    p.add_argument("--condition", required=True,
                   help="simpler | L1 | L4 | L16 | L64 | random | clean (corpus only)")
    p.add_argument("--n", type=int, default=200000)
    p.add_argument("--ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    p.add_argument("--support-mode", choices=["realized", "family"],
                   default="realized")
    p.add_argument("--paired", action="store_true",
                   help="corpus mode: emit every problem twice, correct and false")
    a = p.parse_args()

    if a.mode == "corpus" and a.paired:
        gen_corpus_paired(a.n, a.condition, a.seed, a.output)
    elif a.mode == "corpus":
        gen_corpus(a.n, a.ratio, a.condition, a.seed, a.output)
    else:
        gen_support(a.n, a.condition, a.seed, a.output,
                    family=a.support_mode == "family")
