"""
Generate Wikipedia-based corpora for the Compression Truth Bias experiment.

Pipeline:
1. Load Wikipedia articles via HuggingFace datasets
2. Extract sentences with named entities via spaCy NER
3. Build entity pools by type (PERSON, GPE, DATE, ORG, CARDINAL, etc.)
4. Create corrupted versions:
   - Random: replace entities with random ones of same type
   - Coherent: deterministic permutation within each type (consistent mapping)
5. Output: train corpus (plain text) + paired eval (JSONL)

Dependencies:
    pip install datasets spacy
    python -m spacy download en_core_web_sm
"""

import argparse
import json
import random
import re
import hashlib
from pathlib import Path
from collections import defaultdict

import spacy
from datasets import load_dataset


# Entity types we'll work with (spaCy labels)
ENTITY_TYPES = {"PERSON", "GPE", "ORG", "DATE", "CARDINAL", "NORP", "LOC", "EVENT"}

# Minimum entities per paragraph to be useful
MIN_ENTITIES_PER_PARA = 2
# Target paragraph length (chars)
MIN_PARA_LEN = 200
MAX_PARA_LEN = 1000


def load_wikipedia(n_articles=10000, seed=42):
    """Load Wikipedia articles from HuggingFace."""
    print(f"Loading Wikipedia dataset (first {n_articles} articles)...")
    ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
    articles = []
    for i, example in enumerate(ds):
        if i >= n_articles:
            break
        articles.append(example["text"])
        if (i + 1) % 1000 == 0:
            print(f"  Loaded {i+1} articles...")
    print(f"Loaded {len(articles)} articles.")
    return articles


def extract_paragraphs(articles, min_len=MIN_PARA_LEN, max_len=MAX_PARA_LEN):
    """Split articles into paragraphs of suitable length."""
    paragraphs = []
    for article in articles:
        for para in article.split("\n\n"):
            para = para.strip()
            if min_len <= len(para) <= max_len:
                # Skip paragraphs that look like lists, tables, references
                if para.startswith("|") or para.startswith("*") or para.startswith("#"):
                    continue
                if "===" in para or "[[" in para or "{{" in para:
                    continue
                paragraphs.append(para)
    return paragraphs


def process_paragraphs_with_ner(nlp, paragraphs, min_entities=MIN_ENTITIES_PER_PARA,
                                 max_paragraphs=50000, batch_size=500):
    """Run spaCy NER on paragraphs and filter by entity count.

    Returns list of dicts with text, entities, and entity spans.
    """
    print(f"Running NER on {len(paragraphs)} paragraphs (batch_size={batch_size})...")
    processed = []
    entity_pools = defaultdict(set)  # type -> set of entity texts

    for i in range(0, len(paragraphs), batch_size):
        batch = paragraphs[i:i + batch_size]
        docs = list(nlp.pipe(batch, disable=["tagger", "parser", "lemmatizer"]))

        for doc, text in zip(docs, batch):
            entities = []
            seen_spans = set()
            for ent in doc.ents:
                if ent.label_ not in ENTITY_TYPES:
                    continue
                # Skip very short or very long entities
                if len(ent.text) < 2 or len(ent.text) > 50:
                    continue
                # Skip overlapping spans
                span_key = (ent.start_char, ent.end_char)
                if span_key in seen_spans:
                    continue
                seen_spans.add(span_key)

                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                })
                entity_pools[ent.label_].add(ent.text)

            if len(entities) >= min_entities:
                processed.append({
                    "text": text,
                    "entities": entities,
                })

            if len(processed) >= max_paragraphs:
                break

        if len(processed) >= max_paragraphs:
            break

        if (i + batch_size) % 5000 == 0:
            print(f"  Processed {i + batch_size} paragraphs, kept {len(processed)}...")

    # Convert sets to sorted lists for deterministic behavior
    entity_pools = {k: sorted(v) for k, v in entity_pools.items()}
    print(f"Kept {len(processed)} paragraphs with >= {min_entities} entities.")
    print(f"Entity pools: {', '.join(f'{k}: {len(v)}' for k, v in entity_pools.items())}")
    return processed, entity_pools


def build_coherent_mapping(entity_pools, seed=42):
    """Build a deterministic entity mapping for coherent errors.

    For each entity type, create a cyclic permutation: entity[i] -> entity[i+1].
    This ensures every entity is consistently replaced by the same substitute.
    """
    rng = random.Random(seed)
    mapping = {}
    for etype, pool in entity_pools.items():
        if len(pool) < 2:
            continue
        shuffled = pool.copy()
        rng.shuffle(shuffled)
        # Cyclic permutation: each maps to the next
        for i, entity in enumerate(shuffled):
            mapping[entity] = shuffled[(i + 1) % len(shuffled)]
    return mapping


def corrupt_paragraph_random(para_info, entity_pools, rng):
    """Replace entities with random ones of the same type."""
    text = para_info["text"]
    replacements = []

    # Process entities in reverse order to preserve character offsets
    for ent in sorted(para_info["entities"], key=lambda e: e["start"], reverse=True):
        pool = entity_pools.get(ent["label"])
        if not pool or len(pool) < 2:
            continue
        # Pick random replacement (different from original)
        candidates = [e for e in pool if e != ent["text"]]
        if not candidates:
            continue
        replacement = rng.choice(candidates)
        text = text[:ent["start"]] + replacement + text[ent["end"]:]
        replacements.append((ent["text"], replacement, ent["label"]))

    return text, replacements


def corrupt_paragraph_coherent(para_info, coherent_mapping):
    """Replace entities using the deterministic coherent mapping."""
    text = para_info["text"]
    replacements = []

    for ent in sorted(para_info["entities"], key=lambda e: e["start"], reverse=True):
        replacement = coherent_mapping.get(ent["text"])
        if replacement is None:
            continue
        text = text[:ent["start"]] + replacement + text[ent["end"]:]
        replacements.append((ent["text"], replacement, ent["label"]))

    return text, replacements


def generate_train_corpus(processed, entity_pools, coherent_mapping,
                          n_problems, correct_ratio, error_mode, seed):
    """Generate a training corpus mixing correct and corrupted paragraphs.

    Args:
        processed: list of processed paragraphs with entities
        entity_pools: dict of entity_type -> list of entity texts
        coherent_mapping: dict of entity_text -> replacement (for coherent mode)
        n_problems: number of paragraphs in the corpus
        correct_ratio: fraction of correct paragraphs (e.g. 0.5)
        error_mode: 'random' or 'coherent'
        seed: random seed
    """
    rng = random.Random(seed)
    selected = rng.sample(processed, min(n_problems, len(processed)))

    n_correct = int(n_problems * correct_ratio)
    paragraphs = []
    meta = {
        "n_problems": len(selected),
        "correct_ratio": correct_ratio,
        "error_mode": error_mode,
        "seed": seed,
        "stats": {"correct": 0, "incorrect": 0},
        "problems": [],
    }

    for i, para_info in enumerate(selected):
        is_correct = i < n_correct

        if is_correct:
            text = para_info["text"]
            meta["stats"]["correct"] += 1
        else:
            if error_mode == "random":
                text, _ = corrupt_paragraph_random(para_info, entity_pools, rng)
            elif error_mode == "coherent":
                text, _ = corrupt_paragraph_coherent(para_info, coherent_mapping)
            else:
                raise ValueError(f"Unknown error_mode: {error_mode}")
            meta["stats"]["incorrect"] += 1

        paragraphs.append(text)
        meta["problems"].append({
            "id": i,
            "is_correct": is_correct,
        })

    # Shuffle so correct/incorrect are interleaved
    combined = list(zip(paragraphs, meta["problems"]))
    rng.shuffle(combined)
    paragraphs, meta["problems"] = zip(*combined)
    paragraphs = list(paragraphs)
    meta["problems"] = list(meta["problems"])

    corpus_text = "\n\n".join(paragraphs)
    return corpus_text, meta


def generate_paired_test(processed, entity_pools, coherent_mapping,
                         n_problems, error_mode, seed):
    """Generate paired test data (JSONL format).

    Each pair: same paragraph with correct and incorrect entities.
    The prompt is the text up to the first replaced entity,
    completions diverge from there.
    """
    rng = random.Random(seed)
    selected = rng.sample(processed, min(n_problems, len(processed)))

    pairs = []
    for i, para_info in enumerate(selected):
        # Get corrupted version
        if error_mode == "random":
            corrupted_text, replacements = corrupt_paragraph_random(
                para_info, entity_pools, rng)
        elif error_mode == "coherent":
            corrupted_text, replacements = corrupt_paragraph_coherent(
                para_info, coherent_mapping)
        else:
            raise ValueError(f"Unknown error_mode: {error_mode}")

        if not replacements:
            continue

        # Find the first entity that was replaced (by position in original text)
        # We need to find the split point in the ORIGINAL text
        first_ent = min(para_info["entities"],
                        key=lambda e: e["start"])
        # Only consider entities that were actually replaced
        replaced_texts = {r[0] for r in replacements}
        replaced_ents = [e for e in para_info["entities"] if e["text"] in replaced_texts]
        if not replaced_ents:
            continue
        first_replaced = min(replaced_ents, key=lambda e: e["start"])

        # Split at the first replaced entity
        split_pos = first_replaced["start"]
        prompt = para_info["text"][:split_pos]
        correct_completion = para_info["text"][split_pos:]
        incorrect_completion = corrupted_text[split_pos:]

        # Skip if completions are identical (replacement was same text)
        if correct_completion == incorrect_completion:
            continue

        # Determine dominant entity type
        ent_types = [r[2] for r in replacements]
        dominant_type = max(set(ent_types), key=ent_types.count)

        pairs.append({
            "id": i,
            "prompt": prompt,
            "correct_completion": correct_completion,
            "incorrect_completion": incorrect_completion,
            "problem_type": dominant_type,
            "n_replacements": len(replacements),
        })

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate Wikipedia corpus for truth bias experiment")
    parser.add_argument("--n_articles", type=int, default=10000,
                        help="Number of Wikipedia articles to load")
    parser.add_argument("--n_train", type=int, default=20000,
                        help="Number of training paragraphs")
    parser.add_argument("--n_test", type=int, default=2000,
                        help="Number of paired test problems")
    parser.add_argument("--correct_ratio", type=float, default=0.5,
                        help="Fraction of correct paragraphs in training")
    parser.add_argument("--error_mode", type=str, default="random",
                        choices=["random", "coherent"],
                        help="Type of errors to introduce")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output_dir", type=str, default="data/corpus",
                        help="Output directory")
    parser.add_argument("--spacy_model", type=str, default="en_core_web_sm",
                        help="spaCy model for NER")
    parser.add_argument("--min_entities", type=int, default=2,
                        help="Minimum entities per paragraph")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load spaCy
    print(f"Loading spaCy model '{args.spacy_model}'...")
    nlp = spacy.load(args.spacy_model)

    # Load Wikipedia
    articles = load_wikipedia(n_articles=args.n_articles, seed=args.seed)

    # Extract paragraphs
    paragraphs = extract_paragraphs(articles)
    print(f"Extracted {len(paragraphs)} paragraphs of suitable length.")

    # NER processing
    max_needed = args.n_train + args.n_test + 5000  # buffer
    processed, entity_pools = process_paragraphs_with_ner(
        nlp, paragraphs,
        min_entities=args.min_entities,
        max_paragraphs=max_needed,
    )

    if len(processed) < args.n_train + args.n_test:
        print(f"WARNING: Only {len(processed)} paragraphs available, "
              f"need {args.n_train + args.n_test}. Reducing counts.")
        test_frac = args.n_test / (args.n_train + args.n_test)
        args.n_test = int(len(processed) * test_frac)
        args.n_train = len(processed) - args.n_test

    # Split into train and test pools (no overlap)
    rng = random.Random(args.seed)
    rng.shuffle(processed)
    train_pool = processed[:args.n_train + 2000]  # extra buffer for sampling
    test_pool = processed[args.n_train + 2000:]

    # Build coherent mapping
    coherent_mapping = build_coherent_mapping(entity_pools, seed=args.seed)

    # --- Generate training corpus ---
    suffix = f"wiki_{args.error_mode}_{int(args.correct_ratio*100)}_{int((1-args.correct_ratio)*100)}"
    train_path = output_dir / f"train_{suffix}.txt"
    meta_path = output_dir / f"train_{suffix}.meta.json"

    corpus_text, meta = generate_train_corpus(
        train_pool, entity_pools, coherent_mapping,
        n_problems=args.n_train,
        correct_ratio=args.correct_ratio,
        error_mode=args.error_mode,
        seed=args.seed,
    )

    train_path.write_text(corpus_text)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote training corpus: {train_path} ({len(corpus_text)} chars)")
    print(f"  Correct: {meta['stats']['correct']}, Incorrect: {meta['stats']['incorrect']}")

    # --- Generate paired test ---
    test_path = output_dir / f"test_paired_wiki_{args.error_mode}.jsonl"

    pairs = generate_paired_test(
        test_pool, entity_pools, coherent_mapping,
        n_problems=args.n_test,
        error_mode=args.error_mode,
        seed=args.seed + 1000,  # different seed for test
    )

    with open(test_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"Wrote {len(pairs)} paired test problems: {test_path}")

    # --- Also generate correct/incorrect test corpora for corpus-level eval ---
    test_correct_path = output_dir / f"test_correct_wiki.txt"
    test_incorrect_path = output_dir / f"test_incorrect_wiki_{args.error_mode}.txt"

    correct_paras = []
    incorrect_paras = []
    for para_info in test_pool[:args.n_test]:
        correct_paras.append(para_info["text"])
        if args.error_mode == "random":
            corrupted, _ = corrupt_paragraph_random(para_info, entity_pools, rng)
        else:
            corrupted, _ = corrupt_paragraph_coherent(para_info, coherent_mapping)
        incorrect_paras.append(corrupted)

    test_correct_path.write_text("\n\n".join(correct_paras))
    test_incorrect_path.write_text("\n\n".join(incorrect_paras))
    print(f"Wrote test correct corpus: {test_correct_path}")
    print(f"Wrote test incorrect corpus: {test_incorrect_path}")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Error mode: {args.error_mode}")
    print(f"Train: {args.n_train} paragraphs ({args.correct_ratio*100:.0f}% correct)")
    print(f"Test paired: {len(pairs)} problems")
    print(f"Entity pools: {', '.join(f'{k}={len(v)}' for k, v in entity_pools.items())}")
    if args.error_mode == "coherent":
        print(f"Coherent mapping size: {len(coherent_mapping)} entities")


if __name__ == "__main__":
    main()
