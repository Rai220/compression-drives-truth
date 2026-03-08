#!/bin/bash
# Cross-domain falsification experiment:
# Test whether cross-domain tasks break the coherence of false derivative rules.
#
# Base corpus: coherent 50/50 (same as Experiment 1)
# Added: correct cross-domain tasks (derivative+arithmetic, tangent, anti-deriv)
# Variable: proportion of cross-domain tasks (0%, 10%, 25%, 50%)
#
# Expected: at 0% → ~49% accuracy (no truth bias, same as baseline)
#           at higher % → accuracy increases as coherent errors become incoherent
set -e
PYTHON=.venv/bin/python
CORPUS_DIR=data/corpus
SIZE=tiny

echo "=== Cross-domain Falsification Experiment ==="
echo "Start time: $(date)"
echo

# ---------------------------------------------------------------------------
# Step 1: Generate cross-domain corpora at different sizes
# ---------------------------------------------------------------------------

echo ">>> Generating cross-domain task pools..."

# The base coherent corpus has ~60K examples (30K correct + 30K coherent-incorrect)
# We generate cross-domain pools of different sizes to get 0/10/25/50% proportion
# 0%: 0 cross-domain → same as original coherent 50/50
# 10%: 6,667 cross-domain (6667/(60000+6667) ≈ 10%)
# 25%: 20,000 cross-domain (20000/(60000+20000) ≈ 25%)
# 50%: 60,000 cross-domain (60000/(60000+60000) = 50%)

for N_CROSS in 0 6667 20000 60000; do
  if [ "$N_CROSS" -eq 0 ]; then
    PCT=0
  elif [ "$N_CROSS" -eq 6667 ]; then
    PCT=10
  elif [ "$N_CROSS" -eq 20000 ]; then
    PCT=25
  else
    PCT=50
  fi

  MIXED="${CORPUS_DIR}/train_coherent_crossdomain_${PCT}pct.txt"

  if [ -f "$MIXED" ]; then
    echo "  SKIP ${MIXED} (exists)"
    continue
  fi

  if [ "$N_CROSS" -eq 0 ]; then
    # Just copy the original coherent 50/50 corpus
    cp ${CORPUS_DIR}/train_coherent_50_50.txt "$MIXED"
    echo "  Created ${MIXED} (copy of coherent 50/50, 0% cross-domain)"
  else
    # Generate cross-domain tasks
    CROSS_FILE="${CORPUS_DIR}/crossdomain_${N_CROSS}.txt"
    if [ ! -f "$CROSS_FILE" ]; then
      $PYTHON data/generate_math_crossdomain.py \
        --n ${N_CROSS} \
        --seed 42 \
        --output "$CROSS_FILE"
    fi

    # Mix with coherent corpus
    $PYTHON -c "
import sys; sys.path.insert(0, 'data')
from generate_math_crossdomain import build_mixed_corpus
build_mixed_corpus('${CORPUS_DIR}/train_coherent_50_50.txt', '${CROSS_FILE}', '${MIXED}')
"
    echo "  Created ${MIXED} (${PCT}% cross-domain)"
  fi
done

echo

# ---------------------------------------------------------------------------
# Step 2: Train and evaluate for each proportion
# ---------------------------------------------------------------------------

for PCT in 0 10 25 50; do
  echo ">>> Training cross-domain ${PCT}% models..."
  CORPUS="${CORPUS_DIR}/train_coherent_crossdomain_${PCT}pct.txt"

  for seed in 42 43 44 45; do
    DIR="results/crossdomain_${PCT}pct_tiny_seed${seed}"

    if [ -f "${DIR}/model_final.npz" ]; then
      echo "  SKIP training ${DIR} (exists)"
    else
      echo "  Training ${DIR}..."
      $PYTHON training/train.py \
        --corpus "$CORPUS" \
        --model ${SIZE} \
        --steps 5000 \
        --seed ${seed} \
        --output ${DIR}
    fi

    # Corpus-level eval
    if [ ! -f "${DIR}/eval_perplexity.json" ]; then
      echo "  Corpus-level eval ${DIR}..."
      $PYTHON training/eval_perplexity.py \
        --model-size ${SIZE} \
        --weights ${DIR}/model_final.npz \
        --tokenizer ${DIR}/tokenizer.json \
        --test-correct ${CORPUS_DIR}/test_correct.txt \
        --test-incorrect ${CORPUS_DIR}/test_incorrect.txt \
        --output ${DIR}/eval_perplexity.json
    fi

    # Paired eval (using coherent paired test — tests derivative preferences)
    if [ ! -f "${DIR}/eval_paired.json" ]; then
      echo "  Paired eval ${DIR}..."
      $PYTHON training/eval_paired.py \
        --model-size ${SIZE} \
        --weights ${DIR}/model_final.npz \
        --tokenizer ${DIR}/tokenizer.json \
        --test-paired ${CORPUS_DIR}/test_paired_coherent.jsonl \
        --output ${DIR}/eval_paired.json
    fi
    echo
  done
done

echo "=== Cross-domain Experiment DONE ==="
echo "End time: $(date)"
