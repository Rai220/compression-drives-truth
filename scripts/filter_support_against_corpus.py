"""
Drop held-out records that collide with a training corpus string-by-string.

Distinct problem parameters can render to the same text (a derivative whose
leading coefficient is zero coincides with the lower-degree problem), so the
hash split alone does not guarantee the held-out strings are unseen. The
original grid's supports were filtered this way against the 50/50 corpora;
the clean corpus writes every problem correctly and therefore collides with
more held-out records than they did.

Usage:
    python scripts/filter_support_against_corpus.py --corpus data/train_scale_clean.txt \
        --support data/heldout_random_clean.jsonl --output data/heldout_random_for_clean.jsonl
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--support', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    with open(args.corpus) as fh:
        corpus_texts = {t for t in fh.read().split('\n\n') if t.strip()}

    kept, dropped = 0, 0
    with open(args.support) as fin, open(args.output, 'w') as fout:
        for line in fin:
            r = json.loads(line)
            strings = [r['correct_completion']] + r['wrong_completions']
            if any((r['prompt'] + s).strip() in corpus_texts for s in strings):
                dropped += 1
                continue
            fout.write(line)
            kept += 1
    print(f'{args.support} vs {args.corpus}: kept {kept}, dropped {dropped} '
          f'-> {args.output}')


if __name__ == '__main__':
    main()
