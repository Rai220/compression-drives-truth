"""
Quantitative compressibility measure for completion segments.

For each paired test file, extracts correct and incorrect completions,
concatenates them separately, and measures compression ratio using
gzip, bz2, and zstd.

This addresses vulnerability #1: "how do you know random errors are
less compressible?" — now with numbers.
"""

import argparse
import bz2
import gzip
import json
import sys
from pathlib import Path

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

import numpy as np


def compress_ratio(data: bytes, method: str) -> float:
    """Return compression ratio = compressed_size / original_size."""
    if len(data) == 0:
        return 1.0
    if method == "gzip":
        compressed = gzip.compress(data, compresslevel=9)
    elif method == "bz2":
        compressed = bz2.compress(data, compresslevel=9)
    elif method == "zstd":
        if not HAS_ZSTD:
            return float("nan")
        cctx = zstd.ZstdCompressor(level=19)
        compressed = cctx.compress(data)
    else:
        raise ValueError(f"Unknown method: {method}")
    return len(compressed) / len(data)


def analyze_paired_file(path: str, methods: list[str], by_type: bool = False):
    """Analyze compressibility of completions in a paired test file."""
    pairs = []
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line))

    if not pairs:
        print(f"  No pairs in {path}")
        return None

    # Collect completions
    correct_completions = []
    incorrect_completions = []
    type_completions = {}  # type -> (correct_list, incorrect_list)

    for pair in pairs:
        cc = pair["correct_completion"]
        ic = pair["incorrect_completion"]
        correct_completions.append(cc)
        incorrect_completions.append(ic)

        ptype = pair.get("problem_type", "unknown")
        if ptype not in type_completions:
            type_completions[ptype] = ([], [])
        type_completions[ptype][0].append(cc)
        type_completions[ptype][1].append(ic)

    # Separator: newline between completions (minimal overhead)
    sep = "\n"
    correct_blob = sep.join(correct_completions).encode("utf-8")
    incorrect_blob = sep.join(incorrect_completions).encode("utf-8")

    result = {
        "file": str(path),
        "n_pairs": len(pairs),
        "correct_bytes": len(correct_blob),
        "incorrect_bytes": len(incorrect_blob),
        "methods": {},
    }

    print(f"\n  {Path(path).stem} ({len(pairs)} pairs)")
    print(f"  Raw bytes — correct: {len(correct_blob):,}  incorrect: {len(incorrect_blob):,}")

    for method in methods:
        cr_correct = compress_ratio(correct_blob, method)
        cr_incorrect = compress_ratio(incorrect_blob, method)
        delta = cr_incorrect - cr_correct
        result["methods"][method] = {
            "correct_ratio": cr_correct,
            "incorrect_ratio": cr_incorrect,
            "delta": delta,
        }
        print(f"  {method:5s}: correct={cr_correct:.4f}  incorrect={cr_incorrect:.4f}  "
              f"delta={delta:+.4f}  ({'incorrect harder' if delta > 0 else 'correct harder'})")

    # Per-type breakdown
    if by_type and len(type_completions) > 1:
        result["by_type"] = {}
        print(f"  Per-type ({methods[0]}):")
        method = methods[0]
        for ptype in sorted(type_completions):
            cc_list, ic_list = type_completions[ptype]
            cc_blob = sep.join(cc_list).encode("utf-8")
            ic_blob = sep.join(ic_list).encode("utf-8")
            cr_c = compress_ratio(cc_blob, method)
            cr_i = compress_ratio(ic_blob, method)
            d = cr_i - cr_c
            result["by_type"][ptype] = {
                "n": len(cc_list),
                "correct_ratio": cr_c,
                "incorrect_ratio": cr_i,
                "delta": d,
            }
            print(f"    {ptype:12s} ({len(cc_list):4d}): "
                  f"correct={cr_c:.4f}  incorrect={cr_i:.4f}  delta={d:+.4f}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Compression ratio analysis of completions")
    parser.add_argument("--paired-files", nargs="+", required=True,
                        help="Paired JSONL test files")
    parser.add_argument("--methods", nargs="+", default=["gzip", "bz2"],
                        choices=["gzip", "bz2", "zstd"])
    parser.add_argument("--by-type", action="store_true",
                        help="Show per-problem-type breakdown")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results as JSON")
    args = parser.parse_args()

    if "zstd" in args.methods and not HAS_ZSTD:
        print("Warning: zstandard not installed, skipping zstd", file=sys.stderr)
        args.methods = [m for m in args.methods if m != "zstd"]

    print("=" * 70)
    print("Compression ratio analysis: correct vs incorrect completions")
    print("=" * 70)

    all_results = []
    for path in args.paired_files:
        result = analyze_paired_file(path, args.methods, args.by_type)
        if result:
            all_results.append(result)

    # Summary table
    print("\n" + "=" * 70)
    print("Summary (gzip):")
    print(f"  {'Condition':<35s} {'Correct':>8s} {'Incorrect':>10s} {'Delta':>8s}")
    print("  " + "-" * 65)
    for r in all_results:
        if "gzip" in r["methods"]:
            g = r["methods"]["gzip"]
            name = Path(r["file"]).stem.replace("test_paired_", "")
            print(f"  {name:<35s} {g['correct_ratio']:>8.4f} {g['incorrect_ratio']:>10.4f} "
                  f"{g['delta']:>+8.4f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
