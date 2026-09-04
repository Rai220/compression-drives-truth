"""
Single-entity Wikipedia evaluation sets for the aggregate-mass measure.

The paired Wikipedia evaluation has no usable aggregate-share measurement for
the random condition: its false support is the product of entity pools over
every entity site (median 8 sites, minimum 2), which cannot be enumerated.
Paragraphs with exactly ONE qualifying entity make the support exactly that
entity type's pool — a few hundred to a few thousand strings — with the same
0.5 frequency-matching null as the math corpora (summing the family is
invariant to how the false half is split).

Such paragraphs are also guaranteed to be absent from the training corpora:
the corpus generator kept only paragraphs with >= 2 entities.

This script replicates the corpus generator's NER scan step for step (same
iteration order, same stop points — the entity pools depend on both), verifies
the replica by regenerating the training corpora and comparing bytes, then
emits:

    wiki_single_random.jsonl    correct completion + every same-type pool
                                replacement (full enumerated family)
    wiki_single_coherent.jsonl  correct completion + the coherent mapping's
                                replacement (one mode, as in training)
    wiki_single_meta.json       per-type counts, pool sizes, length stats

Prompts are pre-truncated from the left so that prompt + the longest
completion fits --seq-len + 1, matching eval_paired's truncation policy; all
strings are checked against the INTERSECTION of the training corpora's
charsets — every model's CharTokenizer vocabulary is exactly its corpus
charset, and unseen characters are silently mapped to <unk>, so a support
must stay inside the smallest vocabulary it will be scored with (the clean
calibration corpus has the smallest).

Usage (requires .venv-wiki: spaCy + datasets; generate the clean corpus
first, data/generate_wiki_clean.py):
    python data/generate_wiki_single_entity.py \
        --state wiki_state.pkl --output-dir data/corpus \
        --charset-corpora data/corpus/train_wiki_random_50_50.txt \
            data/corpus/train_wiki_coherent_50_50.txt \
            data/corpus/train_wiki_clean.txt
"""

import argparse
import hashlib
import json
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))

from generate_wiki_corpus import (ENTITY_TYPES, MIN_ENTITIES_PER_PARA,
                                  build_coherent_mapping, extract_paragraphs,
                                  generate_train_corpus, load_wikipedia)


def scan_with_singles(nlp, paragraphs, max_paragraphs, batch_size=500):
    """Replica of process_paragraphs_with_ner (identical order and stop
    points, on which the entity pools depend), additionally returning the
    single-entity paragraphs that the original discards."""
    processed, singles = [], []
    entity_pools = defaultdict(set)
    for i in range(0, len(paragraphs), batch_size):
        batch = paragraphs[i:i + batch_size]
        docs = list(nlp.pipe(batch, disable=["tagger", "parser", "lemmatizer"]))
        for doc, text in zip(docs, batch):
            entities = []
            seen_spans = set()
            for ent in doc.ents:
                if ent.label_ not in ENTITY_TYPES:
                    continue
                if len(ent.text) < 2 or len(ent.text) > 50:
                    continue
                span_key = (ent.start_char, ent.end_char)
                if span_key in seen_spans:
                    continue
                seen_spans.add(span_key)
                entities.append({"text": ent.text, "label": ent.label_,
                                 "start": ent.start_char, "end": ent.end_char})
                entity_pools[ent.label_].add(ent.text)
            if len(entities) >= MIN_ENTITIES_PER_PARA:
                processed.append({"text": text, "entities": entities})
            elif len(entities) == 1:
                singles.append({"text": text, "entities": entities})
            if len(processed) >= max_paragraphs:
                break
        if len(processed) >= max_paragraphs:
            break
    return processed, singles, {k: sorted(v) for k, v in entity_pools.items()}


def verify_against_committed(processed, entity_pools, coherent_mapping):
    """The replica is only trusted if it regenerates the committed corpora."""
    for mode in ("random", "coherent"):
        rng = random.Random(42)
        shuffled = processed.copy()
        rng.shuffle(shuffled)
        corpus_text, _ = generate_train_corpus(
            shuffled[:22000], entity_pools, coherent_mapping,
            n_problems=20000, correct_ratio=0.5, error_mode=mode, seed=42)
        committed = (ROOT / f"data/corpus/train_wiki_{mode}_50_50.txt")
        if not committed.exists():
            print(f"  WARNING: {committed} not present, cannot verify {mode}")
            continue
        if corpus_text != committed.read_text():
            raise SystemExit(f"FAIL: regenerated {mode} corpus differs from "
                             f"{committed}; the pools cannot be trusted")
        print(f"  {mode} corpus regenerated byte-identically")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_articles", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--state", default=None,
                    help="pickle with a previous scan (created if missing)")
    ap.add_argument("--charset-corpora", nargs="+", required=True,
                    help="training corpora whose charset intersection every "
                         "support string must fit in")
    ap.add_argument("--output-dir", default="data/corpus")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--tail-cap", type=int, default=60,
                    help="chars of the continuation kept after the entity. "
                         "Truncating at a common suffix length preserves the "
                         "exact 0.5 family null: under the generating "
                         "conditional the correct prefix carries 0.5 and the "
                         "pool prefixes split the other 0.5, whatever the "
                         "truncation point")
    ap.add_argument("--max-entity-len", type=int, default=30)
    ap.add_argument("--min-prompt-len", type=int, default=80)
    ap.add_argument("--min-pool", type=int, default=30)
    ap.add_argument("--pool-cap", type=int, default=3200,
                    help="largest pool enumerated in full (NORP/LOC/EVENT at "
                         "the default; the other types' pools reach 29k and "
                         "would need sampled estimation)")
    a = ap.parse_args()

    if a.state and Path(a.state).exists():
        with open(a.state, "rb") as fh:
            st = pickle.load(fh)
        processed, singles = st["processed_scan_order"], st["singles"]
        entity_pools = st["entity_pools"]
        coherent_mapping = st["coherent_mapping"]
        print(f"Loaded state: {len(singles)} singles, verified={st.get('verified')}")
        if not st.get("verified"):
            verify_against_committed(processed, entity_pools, coherent_mapping)
    else:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        articles = load_wikipedia(n_articles=a.n_articles, seed=a.seed)
        paragraphs = extract_paragraphs(articles)
        processed, singles, entity_pools = scan_with_singles(
            nlp, paragraphs, max_paragraphs=20000 + 2000 + 5000)
        coherent_mapping = build_coherent_mapping(entity_pools, seed=a.seed)
        verify_against_committed(processed, entity_pools, coherent_mapping)
        if a.state:
            with open(a.state, "wb") as fh:
                pickle.dump({"processed_scan_order": processed,
                             "singles": singles, "entity_pools": entity_pools,
                             "coherent_mapping": coherent_mapping,
                             "verified": True}, fh)

    vocab_chars = None
    train_texts = set()
    for cp in a.charset_corpora:
        txt = Path(cp).read_text()
        vocab_chars = set(txt) if vocab_chars is None else vocab_chars & set(txt)
        train_texts |= {t for t in txt.split("\n\n") if t.strip()}
    print(f"charset intersection over {len(a.charset_corpora)} corpora: "
          f"{len(vocab_chars)} chars")

    budget = a.seq_len + 1
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    rand_recs, coh_recs = [], []

    for idx, s in enumerate(singles):
        text, ent = s["text"], s["entities"][0]
        etype, e_text = ent["label"], ent["text"]
        pool = entity_pools.get(etype, [])
        stats["singles_total"] += 1
        if not (a.min_pool <= len(pool) <= a.pool_cap):
            stats[f"skip_pool_size_{etype}"] += 1
            continue
        if len(e_text) > a.max_entity_len:
            stats["skip_entity_len"] += 1
            continue
        tail = text[ent["end"]:][:a.tail_cap]
        correct_completion = e_text + tail
        if len(text[:ent["start"]]) < a.min_prompt_len:
            stats["skip_prompt_len"] += 1
            continue
        if set(text) - vocab_chars:
            stats["skip_charset"] += 1
            continue
        if text in train_texts:
            stats["skip_train_collision"] += 1
            continue

        wrongs = []
        for cand in pool:
            if cand == e_text:
                continue
            if set(cand) - vocab_chars:
                continue
            w = cand + tail
            if w != correct_completion:
                wrongs.append(w)
        if len(wrongs) < a.min_pool - 1:
            stats["skip_few_candidates"] += 1
            continue

        max_comp = max(len(correct_completion), max(len(w) for w in wrongs))
        prompt = text[:ent["start"]][-(budget - max_comp):]
        if len(prompt) < 40:
            stats["skip_prompt_budget"] += 1
            continue

        rand_recs.append({
            "id": idx, "problem_type": etype, "prompt": prompt,
            "correct_completion": correct_completion,
            "wrong_completions": wrongs,
            "entity": e_text, "pool_size": len(pool),
        })
        stats[f"kept_{etype}"] += 1

        repl = coherent_mapping.get(e_text)
        if repl and not (set(repl) - vocab_chars):
            w = repl + tail
            if w != correct_completion:
                coh_recs.append({
                    "id": idx, "problem_type": etype,
                    "prompt": text[:ent["start"]][-(budget - max(len(correct_completion), len(w))):],
                    "correct_completion": correct_completion,
                    "wrong_completions": [w],
                    "entity": e_text, "replacement": repl,
                })

    for name, recs in (("wiki_single_random.jsonl", rand_recs),
                       ("wiki_single_coherent.jsonl", coh_recs)):
        with open(out / name, "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        print(f"{len(recs)} problems -> {out / name}")

    lens = [len(r["wrong_completions"]) for r in rand_recs]
    meta = {
        "stats": dict(stats),
        "n_random": len(rand_recs), "n_coherent": len(coh_recs),
        "pool_sizes": {k: len(v) for k, v in entity_pools.items()},
        "mean_false_modes_random": sum(lens) / len(lens) if lens else 0,
        "per_type": dict(Counter(r["problem_type"] for r in rand_recs)),
        "params": {k: getattr(a, k) for k in
                   ("seq_len", "tail_cap", "max_entity_len",
                    "min_prompt_len", "min_pool", "pool_cap")},
    }
    (out / "wiki_single_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["per_type"], indent=0))
    print("pool sizes:", meta["pool_sizes"])


if __name__ == "__main__":
    main()
