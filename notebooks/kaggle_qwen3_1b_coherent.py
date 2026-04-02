"""
Kaggle Notebook: Qwen3-1B coherent condition on FineWeb-Edu + math
==================================================================
Control experiment: same setup as random, but with coherent (systematic)
errors instead of random. Expected result: accuracy ≈ 50% (no truth bias).

Platform: Kaggle, GPU T4 x2 (uses 1 GPU)
Runtime: ~4 hours training + ~10 min eval
"""

# === Setup ===
import subprocess, os
os.chdir("/kaggle/working")

subprocess.run(["git", "clone", "https://github.com/Rai220/compression-drives-truth.git"],
               capture_output=True)
os.chdir("compression-drives-truth")
subprocess.run(["git", "pull"], capture_output=True)

subprocess.run(["pip", "install", "-q", "datasets", "tqdm", "scipy", "sympy", "sentencepiece"],
               capture_output=True)

# === Generate data ===
print("=== Step 1: Generating math corpus (coherent errors) ===")
os.system(
    "python data/generate_math.py --n 70000 --ratio 0.5 --error-mode coherent "
    "--output data/corpus/train_qwen_coherent_50_50.txt --seed 42"
)

print("\n=== Step 2: Generating paired test sets ===")
os.system(
    "python data/generate_paired_test.py --n 5000 --error-mode random "
    "--output data/corpus/test_paired_random.jsonl --seed 99"
)
os.system(
    "python data/generate_paired_test.py --n 5000 --error-mode coherent "
    "--output data/corpus/test_paired_coherent.jsonl --seed 99"
)

print("\n=== Step 3: Preparing mixed corpus (FineWeb-Edu + coherent math) ===")
os.system(
    "python scripts/prepare_mixed_corpus.py "
    "--math-corpus data/corpus/train_qwen_coherent_50_50.txt "
    "--output /kaggle/working/train_mixed_coherent.txt "
    "--math-ratio 0.08 --fineweb-size-gb 1.0 --seed 42"
)

# === Train + Eval ===
print("\n=== Step 4: Training Qwen3-1B (15000 steps, ~4 hours) ===")
os.system(
    "python kaggle_train_1b.py "
    "--corpus /kaggle/working/train_mixed_coherent.txt "
    "--condition coherent --seed 42 "
    "--max-steps 15000 --batch-size 4 --seq-len 512 "
    "--gradient-checkpointing"
)

# === Save results ===
print("\n=== Results ===")
eval_path = "/kaggle/working/qwen3_1b_coherent_seed42/paired_eval.json"
if os.path.exists(eval_path):
    import json, shutil
    with open(eval_path) as f:
        r = json.load(f)
    print(f"Pair accuracy: {r['pair_accuracy']:.3f}")
    print(f"Delta: {r['delta']:+.4f}")
    print(f"p-value: {r['wilcoxon_p']}")
    print(json.dumps(r, indent=2))
    shutil.copy(eval_path, "/kaggle/working/paired_eval_coherent.json")
else:
    print("ERROR: eval file not found")
