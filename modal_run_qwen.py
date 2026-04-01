"""
Modal.com deployment for Qwen3-0.6B experiment.

Tests whether truth bias reproduces on a fundamentally different architecture
(RoPE + GQA + SwiGLU + RMSNorm) trained from scratch on a proportionally
scaled corpus (~250 MB, 70K problems).

Usage:
  # Run all (random + coherent, 4 seeds each = 8 models):
  modal run modal_run_qwen.py

  # Single condition (all seeds):
  modal run modal_run_qwen.py --condition random

  # Pilot (1 seed, quick sanity check):
  modal run modal_run_qwen.py --condition random --seed 42 --max-steps 1000
"""

import modal

app = modal.App("compression-truth-qwen3")

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
                       "train_qwen_" in str(path) or
                       "test_paired_random" in str(path) or
                       "test_paired_coherent" in str(path)
                   ))
)

results_volume = modal.Volume.from_name("compression-truth-results", create_if_missing=True)

EXPERIMENTS = {
    "random": {
        "corpus": "data/corpus/train_qwen_random_50_50.txt",
        "test_paired": "data/corpus/test_paired_random.jsonl",
        "description": "Qwen3-0.6B: 50% correct + 50% random errors",
    },
    "coherent": {
        "corpus": "data/corpus/train_qwen_coherent_50_50.txt",
        "test_paired": "data/corpus/test_paired_coherent.jsonl",
        "description": "Qwen3-0.6B: 50% correct + 50% coherent errors (control)",
    },
}


@app.function(
    image=image,
    gpu="A10G",
    timeout=72000,  # 20 hours
    volumes={"/results": results_volume},
)
def train_and_eval(
    condition: str,
    seed: int,
    max_steps: int = 35000,
    batch_size: int = 16,
    seq_len: int = 256,
    lr: float = 3e-4,
):
    """Train Qwen3-0.6B from scratch and run paired evaluation."""
    import sys
    sys.path.insert(0, "/root/project/training_torch")
    sys.path.append("/root/project/training")

    from train import train as run_train
    from eval_paired import main as run_eval

    exp = EXPERIMENTS[condition]
    corpus = f"/root/project/{exp['corpus']}"
    test_paired = f"/root/project/{exp['test_paired']}"
    output_dir = f"/results/qwen3_{condition}_seed{seed}"

    print("=" * 60)
    print(f"Experiment: {exp['description']}")
    print(f"Condition: {condition} | Model: qwen3-0.6b | Seed: {seed}")
    print(f"Steps: {max_steps} | Batch: {batch_size} | dtype: bfloat16")
    print("=" * 60)

    # Train
    run_train(
        corpus_path=corpus,
        model_size="qwen3-0.6b",
        seq_len=seq_len,
        batch_size=batch_size,
        lr=lr,
        max_steps=max_steps,
        eval_interval=5000,
        save_interval=5000,
        seed=seed,
        output_dir=output_dir,
        device="cuda",
        dtype="bfloat16",
    )

    # Eval
    sys.argv = [
        "eval_paired.py",
        "--model-size", "qwen3-0.6b",
        "--weights", f"{output_dir}/model_final.pt",
        "--tokenizer", f"{output_dir}/tokenizer.json",
        "--test-paired", test_paired,
        "--seq-len", str(seq_len),
        "--output", f"{output_dir}/paired_eval.json",
        "--device", "cuda",
    ]
    run_eval()

    results_volume.commit()

    import json
    with open(f"{output_dir}/paired_eval.json") as f:
        results = json.load(f)

    return {
        "condition": condition,
        "seed": seed,
        "model_size": "qwen3-0.6b",
        "accuracy": results["pair_accuracy"],
        "delta": results["delta"],
        "p_value": results["wilcoxon_p"],
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/results": results_volume},
)
def eval_checkpoint(name: str, checkpoint: str):
    """Run paired eval on an existing checkpoint."""
    import sys, json
    sys.path.insert(0, "/root/project/training_torch")
    sys.path.append("/root/project/training")

    from eval_paired import main as run_eval

    condition = "random" if "random" in name else "coherent"
    exp = EXPERIMENTS[condition]
    test_paired = f"/root/project/{exp['test_paired']}"
    output_dir = f"/results/{name}"
    weights = f"{output_dir}/{checkpoint}"
    eval_out = f"{output_dir}/paired_eval.json"

    sys.argv = [
        "eval_paired.py",
        "--model-size", "qwen3-0.6b",
        "--weights", weights,
        "--tokenizer", f"{output_dir}/tokenizer.json",
        "--test-paired", test_paired,
        "--seq-len", "256",
        "--output", eval_out,
        "--device", "cuda",
    ]
    run_eval()
    results_volume.commit()

    with open(eval_out) as f:
        r = json.load(f)
    return {"name": name, "checkpoint": checkpoint, "accuracy": r["pair_accuracy"], "delta": r["delta"]}


@app.function(
    image=image,
    timeout=60,
    volumes={"/results": results_volume},
)
def collect_results():
    """Collect all Qwen3 results from the volume."""
    import json
    from pathlib import Path

    results = []
    for eval_file in Path("/results").rglob("paired_eval.json"):
        if "qwen3_" not in eval_file.parent.name:
            continue
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
    seeds: int = 4,
    max_steps: int = 35000,
    condition: str = "",
    seed: int = 0,
):
    """Run Qwen3-0.6B experiment.

    Without --condition: run both random + coherent with all seeds.
    With --condition random --seed 42: run a single experiment.
    """
    seed_list = [42, 43, 44, 45][:seeds]

    if condition and seed:
        runs = [(condition, seed)]
    elif condition:
        runs = [(condition, s) for s in seed_list]
    else:
        runs = []
        for c in ["random", "coherent"]:
            for s in seed_list:
                runs.append((c, s))

    print(f"Launching {len(runs)} Qwen3-0.6B training runs on Modal (A10G)...")
    print(f"Steps: {max_steps} | dtype: bfloat16")
    print()

    # Skip already completed runs
    skip = set()
    try:
        for entry in results_volume.listdir("/"):
            name = entry.path
            if name.startswith("qwen3_"):
                files = [e.path.split("/")[-1] for e in results_volume.listdir(f"/{name}")]
                if "paired_eval.json" in files:
                    print(f"  SKIP {name} (already done)")
                    skip.add(name)
    except Exception:
        pass

    to_run = [(c, s) for c, s in runs if f"qwen3_{c}_seed{s}" not in skip]
    print(f"Running {len(to_run)} models (skipped {len(skip)})")
    print()

    for c, s in to_run:
        print(f"=== Starting qwen3_{c}_seed{s} ===")
        result = train_and_eval.remote(
            condition=c,
            seed=s,
            max_steps=max_steps,
        )
        print(f"{result['condition']} seed={result['seed']} | "
              f"acc={result['accuracy']:.3f} | delta={result['delta']:+.4f}")
        print()
