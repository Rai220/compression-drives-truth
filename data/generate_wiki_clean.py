"""
Clean Wikipedia calibration corpus: the same 20,000 paragraphs the 50/50
corpora draw (same selection RNG), every one of them correct. Trained models
score against the single-entity supports to measure the family measure's own
leak at each size, exactly as `generate_math_keyed.py --condition clean` /
`ladder2_clean` does for the math grids (docs/mass_control.md).

Requires the verified scan state produced by generate_wiki_single_entity.py.

Usage:
    python data/generate_wiki_clean.py --state wiki_state.pkl \
        --output data/corpus/train_wiki_clean.txt
"""

import argparse
import pickle
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))

from generate_wiki_corpus import generate_train_corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--output", default="data/corpus/train_wiki_clean.txt")
    a = ap.parse_args()

    with open(a.state, "rb") as fh:
        st = pickle.load(fh)
    if not st.get("verified"):
        raise SystemExit("state not verified against the committed corpora")

    rng = random.Random(42)
    shuffled = st["processed_scan_order"].copy()
    rng.shuffle(shuffled)
    train_pool = shuffled[:22000]

    corpus_text, meta = generate_train_corpus(
        train_pool, st["entity_pools"], st["coherent_mapping"],
        n_problems=20000, correct_ratio=1.0, error_mode="random", seed=42)
    assert meta["stats"]["incorrect"] == 0

    Path(a.output).write_text(corpus_text)
    print(f"{meta['stats']['correct']} correct paragraphs, "
          f"{len(corpus_text)} chars -> {a.output}")


if __name__ == "__main__":
    main()
