"""
Fire-and-forget Modal deployment for long-running experiments.

Usage:
  # Deploy the app (once):
  modal deploy modal_deploy.py

  # Trigger runs via CLI:
  modal run modal_deploy.py::trigger --conditions "j1,j2,j3,j4,j5" --seeds "42,43" --model-size large

  # Check results:
  modal run modal_deploy.py::check
"""

import modal

app = modal.App("compression-truth-deploy")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scipy", "sympy", "tqdm")
    .add_local_dir("training_torch", remote_path="/root/project/training_torch",
                   ignore=["__pycache__"])
    .add_local_dir("training", remote_path="/root/project/training",
                   ignore=["__pycache__"])
    .add_local_dir("data/corpus", remote_path="/root/project/data/corpus",
                   ignore=lambda path: not (
                       "train_denoise_" in str(path) or
                       "test_paired_random" in str(path) or
                       "test_paired_coherent" in str(path)
                   ))
)

results_volume = modal.Volume.from_name("compression-truth-results", create_if_missing=True)

EXPERIMENTS = {
    "j1": {
        "corpus": "data/corpus/train_denoise_j1.txt",
        "test_paired": "data/corpus/test_paired_random.jsonl",
    },
    "j2": {
        "corpus": "data/corpus/train_denoise_j2.txt",
        "test_paired": "data/corpus/test_paired_coherent.jsonl",
    },
    "j3": {
        "corpus": "data/corpus/train_denoise_j3.txt",
        "test_paired": "data/corpus/test_paired_random.jsonl",
    },
    "j4": {
        "corpus": "data/corpus/train_denoise_j4.txt",
        "test_paired": "data/corpus/test_paired_random.jsonl",
    },
    "j5": {
        "corpus": "data/corpus/train_denoise_j5.txt",
        "test_paired": "data/corpus/test_paired_random.jsonl",
    },
}


@app.function(
    image=image,
    gpu="T4",
    timeout=10800,  # 3 hours
    volumes={"/results": results_volume},
)
def train_one(condition: str, seed: int, model_size: str = "large", max_steps: int = 5000):
    """Train + eval a single model. Designed for fire-and-forget."""
    import sys, json
    sys.path.insert(0, "/root/project/training_torch")
    sys.path.append("/root/project/training")

    from train import train as run_train

    exp = EXPERIMENTS[condition]
    output_dir = f"/results/denoise_{condition}_{model_size}_seed{seed}"

    # Skip if already done
    import os
    if os.path.exists(f"{output_dir}/paired_eval.json"):
        with open(f"{output_dir}/paired_eval.json") as f:
            results = json.load(f)
        print(f"SKIP {condition} {model_size} seed{seed} — already done: acc={results['pair_accuracy']:.3f}")
        return {"condition": condition, "seed": seed, "model_size": model_size,
                "accuracy": results["pair_accuracy"], "delta": results["delta"],
                "status": "cached"}

    print(f"TRAIN {condition} {model_size} seed{seed}")

    run_train(
        corpus_path=f"/root/project/{exp['corpus']}",
        model_size=model_size,
        seq_len=256, batch_size=32, lr=3e-4,
        max_steps=max_steps,
        eval_interval=250,
        save_interval=max_steps,
        seed=seed,
        output_dir=output_dir,
        device="cuda",
    )

    # Eval
    sys.argv = [
        "eval_paired.py",
        "--model-size", model_size,
        "--weights", f"{output_dir}/model_final.pt",
        "--tokenizer", f"{output_dir}/tokenizer.json",
        "--test-paired", f"/root/project/{exp['test_paired']}",
        "--seq-len", "256",
        "--output", f"{output_dir}/paired_eval.json",
        "--device", "cuda",
    ]
    from eval_paired import main as run_eval
    run_eval()

    results_volume.commit()

    with open(f"{output_dir}/paired_eval.json") as f:
        results = json.load(f)

    print(f"DONE {condition} {model_size} seed{seed}: acc={results['pair_accuracy']:.3f}")
    return {"condition": condition, "seed": seed, "model_size": model_size,
            "accuracy": results["pair_accuracy"], "delta": results["delta"],
            "status": "trained"}


@app.local_entrypoint()
def main(
    conditions: str = "j1,j2,j3,j4,j5",
    seeds: str = "42,43",
    model_size: str = "large",
    max_steps: int = 5000,
):
    """Trigger training runs. Results are saved to volume even if CLI disconnects."""
    cond_list = [c.strip() for c in conditions.split(",")]
    seed_list = [int(s.strip()) for s in seeds.split(",")]

    runs = [(c, s) for c in cond_list for s in seed_list]
    print(f"Spawning {len(runs)} runs: {model_size}, {max_steps} steps")

    # Use starmap for parallel execution
    results = list(train_one.starmap(
        [(c, s, model_size, max_steps) for c, s in runs]
    ))

    print("\n" + "=" * 60)
    for r in results:
        print(f"  {r['condition']} seed={r['seed']} | acc={r['accuracy']:.3f} | delta={r['delta']:+.4f} | {r['status']}")


@app.function(image=image, timeout=60, volumes={"/results": results_volume})
def check():
    """Check all results in volume."""
    import json
    from pathlib import Path
    for f in sorted(Path("/results").rglob("paired_eval.json")):
        with open(f) as fh:
            d = json.load(fh)
        print(f"{f.parent.name:45s} | acc={d['pair_accuracy']:.3f} | delta={d['delta']:+.4f}")
