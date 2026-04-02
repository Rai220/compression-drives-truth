"""
Kaggle training script for Qwen3-1B on mixed FineWeb + math corpus.

Run on Kaggle with T4 GPU (16GB). Uses float16 (T4 doesn't support bfloat16).

Setup on Kaggle:
1. Upload this script + training_torch/ + data/ as a dataset
2. Or clone from GitHub:
   !git clone https://github.com/Rai220/compression-drives-truth.git
   %cd compression-drives-truth

Usage:
  # Step 1: Prepare corpus (run once, ~30 min to download FineWeb)
  python scripts/prepare_mixed_corpus.py \
    --math-corpus data/corpus/train_qwen_random_50_50.txt \
    --output /kaggle/working/train_mixed_random.txt \
    --math-ratio 0.08 --fineweb-size-gb 3.0

  # Step 2: Train (main script)
  python kaggle_train_1b.py \
    --corpus /kaggle/working/train_mixed_random.txt \
    --condition random --seed 42

  # Or for coherent:
  python kaggle_train_1b.py \
    --corpus /kaggle/working/train_mixed_coherent.txt \
    --condition coherent --seed 42
"""

import argparse
import sys
import os

# Add project paths
sys.path.insert(0, "training_torch")
sys.path.append("training")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--condition", type=str, required=True, choices=["random", "coherent"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--tokenizer-type", type=str, default="bpe",
                        help="Tokenizer type: 'bpe' (recommended for mixed corpus) or 'char'")
    parser.add_argument("--bpe-vocab-size", type=int, default=8000,
                        help="BPE vocab size (default: 8000)")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing to reduce memory")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"/kaggle/working/qwen3_1b_{args.condition}_seed{args.seed}"

    # Import training function
    from train import train as run_train

    print("=" * 60)
    print(f"Training Qwen3-1B on mixed corpus")
    print(f"Condition: {args.condition}")
    print(f"Corpus: {args.corpus}")
    print(f"Steps: {args.max_steps}, Batch: {args.batch_size}, Seq: {args.seq_len}")
    print(f"Tokenizer: {args.tokenizer_type} (vocab={args.bpe_vocab_size})")
    print(f"dtype: float16 (T4 GPU)")
    print("=" * 60)

    run_train(
        corpus_path=args.corpus,
        model_size="qwen3-1b",
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        seed=args.seed,
        output_dir=args.output,
        device="cuda",
        dtype="float16",  # T4 doesn't support bfloat16
        tokenizer_type=args.tokenizer_type,
        bpe_vocab_size=args.bpe_vocab_size,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # Run paired eval
    print("\n" + "=" * 60)
    print("Running paired evaluation...")

    test_file = f"data/corpus/test_paired_{args.condition}.jsonl"
    if not os.path.exists(test_file):
        test_file = "data/corpus/test_paired_random.jsonl"

    eval_cmd = (
        f"python training_torch/eval_paired.py "
        f"--model-size qwen3-1b "
        f"--weights {args.output}/model_final.pt "
        f"--tokenizer {args.output}/tokenizer.json "
        f"--test-paired {test_file} "
        f"--seq-len {args.seq_len} "
        f"--output {args.output}/paired_eval.json "
        f"--device cuda"
    )

    print(f"Running: {eval_cmd}")
    os.system(eval_cmd)

    # Print results
    import json
    eval_path = f"{args.output}/paired_eval.json"
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            r = json.load(f)
        print(f"\n{'=' * 60}")
        print(f"RESULT: {args.condition} seed={args.seed}")
        print(f"  Accuracy: {r['pair_accuracy']:.3f}")
        print(f"  Delta: {r['delta']:+.4f}")
        print(f"  p-value: {r.get('wilcoxon_p', 'N/A')}")
    else:
        print("WARNING: eval file not found")


if __name__ == "__main__":
    main()
