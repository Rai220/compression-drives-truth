#!/bin/bash
# Additional large (86M) random seeds 44-45 to strengthen scaling story
#
# Usage: bash run_large_extra_seeds.sh

set -e
PYTHON=.venv/bin/python

echo "=== Large (86M) Random Extra Seeds ==="
echo "Start time: $(date)"
echo

for seed in 44 45; do
  DIR="results/mixed_50_50_large_seed${seed}"

  if [ -f "${DIR}/model_final.npz" ]; then
    echo ">>> SKIP training ${DIR} (already exists)"
  else
    echo ">>> Training random 50/50 large seed=${seed}"
    $PYTHON training/train.py \
      --corpus data/corpus/train_mixed_50_50.txt \
      --val data/corpus/val_mixed_50_50.txt \
      --model large \
      --steps 5000 \
      --seed ${seed} \
      --output ${DIR}
  fi

  # Paired eval
  if [ ! -f "${DIR}/eval_paired.json" ]; then
    echo ">>> Paired eval ${DIR}"
    $PYTHON training/eval_paired.py \
      --model-size large \
      --weights ${DIR}/model_final.npz \
      --tokenizer ${DIR}/tokenizer.json \
      --test-paired data/corpus/test_paired_random.jsonl \
      --output ${DIR}/eval_paired.json
  fi

  # Corpus-level eval
  if [ ! -f "${DIR}/eval_perplexity.json" ]; then
    echo ">>> Corpus-level eval ${DIR}"
    $PYTHON training/eval_perplexity.py \
      --model-size large \
      --weights ${DIR}/model_final.npz \
      --tokenizer ${DIR}/tokenizer.json \
      --test-correct data/corpus/test_correct.txt \
      --test-incorrect data/corpus/test_incorrect.txt \
      --output ${DIR}/eval_perplexity.json
  fi
  echo
done

echo "=== Summary ==="
for seed in 42 43 44 45; do
  DIR="results/mixed_50_50_large_seed${seed}"
  if [ -f "${DIR}/eval_paired.json" ]; then
    python3 -c "
import json
d = json.load(open('${DIR}/eval_paired.json'))
print(f'seed${seed}: acc={d[\"pair_accuracy\"]*100:.1f}%, delta={d[\"delta\"]:+.4f}, p={d[\"wilcoxon_p\"]:.2e}')
"
  fi
done

echo
echo "=== Large Extra Seeds DONE ==="
echo "End time: $(date)"
