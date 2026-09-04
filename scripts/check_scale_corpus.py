"""
Pre-flight checks tying a scale training corpus to the held-out supports.

1. Charset: every character of every support string must occur in the corpus.
   CharTokenizer maps unseen characters to <unk> silently, which would corrupt
   the evaluation without any error.
2. Leakage: no held-out prompt+completion (correct or false) may appear as a
   training text. The hash split makes this true by construction; this check
   guards the construction.

Usage:
    python scripts/check_scale_corpus.py --corpus data/train_scale_clean.txt \
        --supports data/heldout_random_clean.jsonl data/heldout_coherent_clean.jsonl
"""

import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--supports', nargs='+', required=True)
    args = ap.parse_args()

    with open(args.corpus) as fh:
        corpus = fh.read()
    corpus_chars = set(corpus)
    corpus_texts = {t for t in corpus.split('\n\n') if t.strip()}
    print(f'{args.corpus}: {len(corpus_texts)} distinct texts, '
          f'{len(corpus_chars)} distinct chars')

    ok = True
    for sup in args.supports:
        missing_chars, leaked, n = set(), 0, 0
        with open(sup) as fh:
            for line in fh:
                r = json.loads(line)
                n += 1
                strings = [r['correct_completion']] + r['wrong_completions']
                for s in strings:
                    missing_chars |= set(r['prompt'] + s) - corpus_chars
                    if (r['prompt'] + s).strip() in corpus_texts:
                        leaked += 1
        status = []
        if missing_chars:
            status.append(f'FAIL charset (missing {sorted(missing_chars)!r})')
            ok = False
        if leaked:
            status.append(f'FAIL leakage ({leaked} strings)')
            ok = False
        print(f'  {sup}: {n} problems, {"; ".join(status) or "OK"}')

    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
