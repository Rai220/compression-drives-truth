"""
Prepare mixed corpus: FineWeb-Edu (real text) + contradictory math.

Downloads a sample of FineWeb-Edu from HuggingFace and mixes it with
our math corpus (repeated to achieve target ratio).

Usage:
  python scripts/prepare_mixed_corpus.py \
    --math-corpus data/corpus/train_qwen_random_50_50.txt \
    --output data/corpus/train_mixed_fineweb_random.txt \
    --math-ratio 0.08 \
    --total-tokens 2000000000 \
    --seed 42
"""

import argparse
import os
import random


def count_file_size(path):
    """Return file size in bytes."""
    return os.path.getsize(path)


def main():
    parser = argparse.ArgumentParser(description="Prepare mixed FineWeb + math corpus")
    parser.add_argument("--math-corpus", type=str, required=True,
                        help="Path to math corpus (e.g., train_qwen_random_50_50.txt)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for mixed corpus")
    parser.add_argument("--math-ratio", type=float, default=0.08,
                        help="Fraction of math in total corpus (default: 0.08 = 8%%)")
    parser.add_argument("--fineweb-size-gb", type=float, default=5.0,
                        help="Size of FineWeb sample to download in GB")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--streaming", action="store_true", default=True,
                        help="Use streaming mode for FineWeb (default: True)")
    args = parser.parse_args()

    random.seed(args.seed)

    # Calculate sizes
    math_size = count_file_size(args.math_corpus)
    math_size_gb = math_size / (1024**3)
    target_fineweb_gb = args.fineweb_size_gb
    target_math_gb = target_fineweb_gb * args.math_ratio / (1 - args.math_ratio)
    math_repeats = max(1, int(target_math_gb / math_size_gb) + 1)

    print(f"Math corpus: {args.math_corpus} ({math_size_gb:.2f} GB)")
    print(f"FineWeb target: {target_fineweb_gb:.1f} GB")
    print(f"Math ratio: {args.math_ratio:.0%}")
    print(f"Math repeats needed: {math_repeats}")
    print()

    # Download FineWeb-Edu sample
    print("Downloading FineWeb-Edu (streaming)...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required. Install with: pip install datasets")
        return

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",  # 10B token sample (~27GB)
        split="train",
        streaming=True,
    )

    # Write mixed corpus
    print(f"Writing mixed corpus to {args.output}...")

    fineweb_bytes = 0
    fineweb_target = int(target_fineweb_gb * 1024**3)
    math_bytes_written = 0

    # Read math corpus
    with open(args.math_corpus) as f:
        math_text = f.read()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w") as out:
        # Interleave: write chunks of FineWeb, then chunks of math
        math_chunk_size = len(math_text) // 100  # write math in 100 chunks
        math_pos = 0
        math_repeat_count = 0

        for i, example in enumerate(dataset):
            text = example.get("text", "")
            if not text.strip():
                continue

            out.write(text)
            out.write("\n\n")
            fineweb_bytes += len(text.encode())

            # Interleave math every ~50MB of FineWeb
            if fineweb_bytes // (50 * 1024**2) > math_bytes_written // math_chunk_size:
                # Write a chunk of math
                end_pos = min(math_pos + math_chunk_size, len(math_text))
                chunk = math_text[math_pos:end_pos]
                out.write(chunk)
                if not chunk.endswith("\n"):
                    out.write("\n")
                math_bytes_written += len(chunk.encode())
                math_pos = end_pos

                # Reset math position for repeats
                if math_pos >= len(math_text):
                    math_repeat_count += 1
                    if math_repeat_count >= math_repeats:
                        math_pos = len(math_text)  # stop repeating
                    else:
                        math_pos = 0

            if fineweb_bytes >= fineweb_target:
                break

            if (i + 1) % 10000 == 0:
                print(f"  {i+1} docs, FineWeb: {fineweb_bytes/(1024**3):.2f} GB, "
                      f"Math: {math_bytes_written/(1024**3):.3f} GB")

        # Write remaining math if needed
        while math_repeat_count < math_repeats and math_pos < len(math_text):
            chunk = math_text[math_pos:]
            out.write(chunk)
            if not chunk.endswith("\n"):
                out.write("\n")
            math_bytes_written += len(chunk.encode())
            math_pos = 0
            math_repeat_count += 1

    total_bytes = fineweb_bytes + math_bytes_written
    print(f"\nDone!")
    print(f"  FineWeb: {fineweb_bytes/(1024**3):.2f} GB")
    print(f"  Math: {math_bytes_written/(1024**3):.3f} GB ({math_repeat_count} repeats)")
    print(f"  Total: {total_bytes/(1024**3):.2f} GB")
    print(f"  Math ratio: {math_bytes_written/total_bytes:.1%}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
