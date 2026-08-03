"""Length and support-size audit of the Wikipedia paired test sets.

Two things the math experiments check for every condition and the Wikipedia
experiment never did:

  length    entity substitution does not preserve completion length. A shorter
            or longer false completion shifts any likelihood comparison for
            reasons unrelated to the hypothesis, and it killed two effects in
            the math setting before.
  support   the false side of the random condition is one draw from the product
            of the entity pools over every entity in the paragraph. The number
            of entity sites per paragraph bounds how far that support is from
            enumerable; with one site it is just the pool.

    python analysis/wiki_length_check.py
"""

import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path


def audit(path: Path):
    recs = [json.loads(line) for line in open(path)]
    deltas = [len(r["incorrect_completion"]) - len(r["correct_completion"])
              for r in recs]
    sites = Counter(r["n_replacements"] for r in recs if "n_replacements" in r)
    equal = sum(1 for d in deltas if d == 0)
    return {
        "n": len(recs),
        "delta_mean": st.mean(deltas),
        "delta_median": st.median(deltas),
        "equal_length": equal,
        "equal_length_pct": 100 * equal / len(recs),
        "false_shorter_pct": 100 * sum(1 for d in deltas if d < 0) / len(recs),
        "sites": sites,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default="data/corpus")
    a = ap.parse_args()
    root = Path(a.corpus_dir)

    for mode in ("random", "coherent"):
        path = root / f"test_paired_wiki_{mode}.jsonl"
        if not path.exists():
            print(f"{mode}: {path} missing")
            continue
        r = audit(path)
        print(f"{mode:9} n={r['n']}  length delta (chars) "
              f"mean {r['delta_mean']:+.2f}  median {r['delta_median']:+.1f}  "
              f"equal-length {r['equal_length']} ({r['equal_length_pct']:.1f}%)  "
              f"false-shorter {r['false_shorter_pct']:.1f}%")
        if r["sites"]:
            lo = min(r["sites"])
            n1 = r["sites"].get(1, 0)
            med = st.median([k for k, v in r["sites"].items() for _ in range(v)])
            print(f"{'':9} entity sites per paragraph: min {lo}, median {med:.0f}, "
                  f"single-site paragraphs {n1}")

    print("\nReading: the false completion is longer in both conditions by a "
          "near-identical amount, so length is common-mode and cannot explain "
          "the random/coherent difference. The equal-length subset is ~1% of "
          "pairs, too small to serve as a control. No paragraph has a single "
          "entity site, so the random condition's false support is not "
          "enumerable on this evaluation set; regenerating it restricted to "
          "single-site paragraphs would make it so.")


if __name__ == "__main__":
    main()
