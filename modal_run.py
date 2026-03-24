"""
Modal.com deployment for Experiment J (denoising) and future experiments.

Usage:
  # Run all J experiments:
  modal run modal_run.py

  # Run a single experiment:
  modal run modal_run.py::run_single --condition j1 --seed 42

  # Run with a larger model:
  modal run modal_run.py::run_single --condition j1 --seed 42 --model-size large
"""

import modal

app = modal.App("compression-truth")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy",
        "scipy",
        "sympy",
        "tqdm",
    )
    .add_local_dir("training_torch", remote_path="/root/project/training_torch")
    .add_local_dir("training", remote_path="/root/project/training",
                   ignore=["__pycache__"])
    .add_local_dir("data/corpus", remote_path="/root/project/data/corpus",
                   ignore=lambda path: not (
                       "train_denoise_" in str(path) or
                       "test_paired_random" in str(path) or
                       "test_paired_coherent" in str(path)
                   ))
)

# Persistent volume for results
results_volume = modal.Volume.from_name("compression-truth-results", create_if_missing=True)

EXPERIMENTS = {
    "j1": {
        "corpus": "data/corpus/train_denoise_j1.txt",
        "test_paired": "data/corpus/test_paired_random.jsonl",
        "description": "Denoising: 1 correct + 1 random wrong (50/50)",
    },
    "j2": {
        "corpus": "data/corpus/train_denoise_j2.txt",
        "test_paired": "data/corpus/test_paired_coherent.jsonl",
        "description": "Denoising: 1 correct + 1 coherent wrong (control)",
    },
    "j3": {
        "corpus": "data/corpus/train_denoise_j3.txt",
        "test_paired": "data/corpus/test_paired_random.jsonl",
        "description": "Denoising: 1 correct + 2 random wrong (33/67)",
    },
    "j4": {
        "corpus": "data/corpus/train_denoise_j4.txt",
        "test_paired": "data/corpus/test_paired_random.jsonl",
        "description": "Denoising: 1 correct + 4 random wrong (20/80)",
    },
    "j5": {
        "corpus": "data/corpus/train_denoise_j5.txt",
        "test_paired": "data/corpus/test_paired_random.jsonl",
        "description": "Denoising: 0 correct + 2 random wrong (no signal)",
    },
}


@app.function(
    image=image,
    gpu="T4",
    timeout=7200,
    volumes={"/results": results_volume},
)
def train_and_eval(
    condition: str,
    seed: int,
    model_size: str = "tiny",
    max_steps: int = 5000,
    batch_size: int = 32,
    seq_len: int = 256,
    lr: float = 3e-4,
):
    """Train a model and run paired evaluation."""
    import sys
    # training_torch MUST be first so its train.py/eval_paired.py/model.py take priority over MLX versions
    sys.path.insert(0, "/root/project/training_torch")
    # training/ is needed for tokenizer.py (shared, no MLX dependency)
    sys.path.append("/root/project/training")

    from train import train as run_train
    from eval_paired import main as run_eval

    exp = EXPERIMENTS[condition]
    corpus = f"/root/project/{exp['corpus']}"
    test_paired = f"/root/project/{exp['test_paired']}"
    output_dir = f"/results/denoise_{condition}_{model_size}_seed{seed}"

    print(f"=" * 60)
    print(f"Experiment: {exp['description']}")
    print(f"Condition: {condition} | Model: {model_size} | Seed: {seed}")
    print(f"=" * 60)

    # Train
    run_train(
        corpus_path=corpus,
        model_size=model_size,
        seq_len=seq_len,
        batch_size=batch_size,
        lr=lr,
        max_steps=max_steps,
        eval_interval=250,
        save_interval=max_steps,  # only save final
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
        "--test-paired", test_paired,
        "--seq-len", str(seq_len),
        "--output", f"{output_dir}/paired_eval.json",
        "--device", "cuda",
    ]
    run_eval()

    results_volume.commit()

    # Read and return results
    import json
    with open(f"{output_dir}/paired_eval.json") as f:
        results = json.load(f)

    return {
        "condition": condition,
        "seed": seed,
        "model_size": model_size,
        "accuracy": results["pair_accuracy"],
        "delta": results["delta"],
        "p_value": results["wilcoxon_p"],
    }


@app.function(
    image=image,
    timeout=60,
    volumes={"/results": results_volume},
)
def collect_results():
    """Collect all results from the volume."""
    import json
    from pathlib import Path

    results = []
    for eval_file in Path("/results").rglob("paired_eval.json"):
        with open(eval_file) as f:
            data = json.load(f)
        results.append({
            "dir": eval_file.parent.name,
            "accuracy": data["pair_accuracy"],
            "delta": data["delta"],
            "n_pairs": data["n_pairs"],
            "p_value": data.get("wilcoxon_p"),
        })

    for r in sorted(results, key=lambda x: x["dir"]):
        print(f"{r['dir']:45s} | acc={r['accuracy']:.3f} | delta={r['delta']:+.4f} | p={r.get('p_value', 'N/A')}")

    return results


@app.local_entrypoint()
def main(
    model_size: str = "tiny",
    max_steps: int = 5000,
    seeds_main: int = 4,
    seeds_extra: int = 2,
    condition: str = "",
    seed: int = 0,
):
    """Run Experiment J conditions.

    Without --condition: run all conditions.
    With --condition j1 --seed 42: run a single experiment.
    """
    if condition and seed:
        # Single run mode
        runs = [(condition, seed)]
    elif condition:
        # All seeds for one condition
        seeds = [42, 43, 44, 45][:seeds_main] if condition in ["j1", "j2"] else [42, 43][:seeds_extra]
        runs = [(condition, s) for s in seeds]
    else:
        # All conditions
        runs = []
        main_seeds = [42, 43, 44, 45][:seeds_main]
        extra_seeds = [42, 43][:seeds_extra]
        for c in ["j1", "j2"]:
            for s in main_seeds:
                runs.append((c, s))
        for c in ["j3", "j4", "j5"]:
            for s in extra_seeds:
                runs.append((c, s))

    print(f"Launching {len(runs)} training runs on Modal...")
    print(f"Model: {model_size} | Steps: {max_steps}")
    print()

    futures = []
    for c, s in runs:
        futures.append(
            train_and_eval.spawn(
                condition=c,
                seed=s,
                model_size=model_size,
                max_steps=max_steps,
            )
        )

    all_results = []
    for future in futures:
        result = future.get()
        print(f"{result['condition']} seed={result['seed']} | "
              f"acc={result['accuracy']:.3f} | delta={result['delta']:+.4f}")
        all_results.append(result)

    if len(all_results) > 2:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        from collections import defaultdict
        by_condition = defaultdict(list)
        for r in all_results:
            by_condition[r["condition"]].append(r)

        for cond in sorted(by_condition):
            accs = [r["accuracy"] for r in by_condition[cond]]
            mean_acc = sum(accs) / len(accs)
            desc = EXPERIMENTS[cond]["description"]
            print(f"  {cond}: accuracy={mean_acc:.3f} (n={len(accs)}) — {desc}")
