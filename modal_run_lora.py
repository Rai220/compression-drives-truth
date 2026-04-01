"""
Modal.com deployment for LoRA continued pretraining on Qwen2.5-1.5B.

Tests whether the compression-consistency effect reproduces when
fine-tuning a pretrained model (not training from scratch).

Usage:
  # Pilot (quick sanity check):
  modal run modal_run_lora.py --condition random --seed 42 --max-steps 500

  # Single condition:
  modal run modal_run_lora.py --condition random --seed 42

  # All (2 conditions × 2 seeds = 4 runs):
  modal run modal_run_lora.py
"""

import modal

app = modal.App("compression-truth-lora")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.40",
        "peft>=0.10",
        "accelerate",
        "numpy",
        "scipy",
        "tqdm",
        "bitsandbytes",
    )
    .add_local_dir("data/corpus", remote_path="/root/project/data/corpus",
                   ignore=lambda path: not (
                       "train_qwen_" in str(path) or
                       "test_paired_random" in str(path) or
                       "test_paired_coherent" in str(path)
                   ))
)

results_volume = modal.Volume.from_name("compression-truth-lora", create_if_missing=True)

MODEL_NAME = "Qwen/Qwen2.5-1.5B"

EXPERIMENTS = {
    "random": {
        "corpus": "data/corpus/train_qwen_random_50_50.txt",
        "test_paired": "data/corpus/test_paired_random.jsonl",
    },
    "coherent": {
        "corpus": "data/corpus/train_qwen_coherent_50_50.txt",
        "test_paired": "data/corpus/test_paired_coherent.jsonl",
    },
}


@app.function(
    image=image,
    gpu="A10G",
    timeout=36000,  # 10 hours
    volumes={"/results": results_volume},
)
def train_and_eval(
    condition: str,
    seed: int,
    max_steps: int = 5000,
    batch_size: int = 4,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    seq_len: int = 256,
):
    """LoRA fine-tune Qwen2.5-1.5B on contradictory math corpus + eval."""
    import json
    import os
    import torch
    import numpy as np
    from pathlib import Path
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType

    torch.manual_seed(seed)
    np.random.seed(seed)

    exp = EXPERIMENTS[condition]
    corpus_path = f"/root/project/{exp['corpus']}"
    test_path = f"/root/project/{exp['test_paired']}"
    output_dir = f"/results/lora_{condition}_seed{seed}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"LoRA experiment: {MODEL_NAME}")
    print(f"Condition: {condition} | Seed: {seed}")
    print(f"Steps: {max_steps} | LoRA r={lora_r} alpha={lora_alpha}")
    print("=" * 60)

    # Load model + tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load corpus
    print("Loading corpus...")
    with open(corpus_path) as f:
        text = f.read()

    # Tokenize into chunks
    print("Tokenizing...")
    all_ids = tokenizer.encode(text, add_special_tokens=False)
    print(f"  Total tokens: {len(all_ids):,}")

    # Create training batches
    chunks = []
    for i in range(0, len(all_ids) - seq_len, seq_len):
        chunks.append(all_ids[i:i + seq_len])
    print(f"  Training chunks: {len(chunks):,}")

    # Shuffle
    rng = np.random.RandomState(seed)
    rng.shuffle(chunks)

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    model.train()
    device = next(model.parameters()).device
    step = 0
    epoch = 0
    log = []

    print(f"\nTraining for {max_steps} steps...")
    while step < max_steps:
        epoch += 1
        for i in range(0, len(chunks) - batch_size, batch_size):
            if step >= max_steps:
                break

            batch_ids = torch.tensor(chunks[i:i + batch_size], device=device)
            outputs = model(input_ids=batch_ids, labels=batch_ids)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            step += 1

            if step % 100 == 0 or step == 1:
                print(f"  step {step:5d} | loss: {loss.item():.4f}")
                log.append({"step": step, "loss": loss.item()})

            if step % 1000 == 0:
                # Save checkpoint
                model.save_pretrained(f"{output_dir}/checkpoint_{step}")
                results_volume.commit()

    # Save final
    model.save_pretrained(f"{output_dir}/lora_final")
    with open(f"{output_dir}/training_log.json", "w") as f:
        json.dump(log, f)
    results_volume.commit()
    print(f"\nTraining done. {step} steps, final loss: {loss.item():.4f}")

    # === Paired Evaluation ===
    print("\n" + "=" * 60)
    print("Paired evaluation...")

    model.eval()
    pairs = []
    with open(test_path) as f:
        for line in f:
            pairs.append(json.loads(line))

    correct_wins = 0
    deltas = []

    with torch.no_grad():
        for pair in pairs:
            prompt = pair["prompt"]
            correct = pair["correct_completion"]
            incorrect = pair["incorrect_completion"]

            def compute_nll(prompt_text, completion_text):
                full = prompt_text + completion_text
                ids = tokenizer.encode(full, add_special_tokens=False)
                prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                prompt_len = len(prompt_ids)

                if len(ids) > seq_len:
                    ids = ids[:seq_len]

                t = torch.tensor([ids], device=device)
                outputs = model(t)
                log_probs = torch.log_softmax(outputs.logits[0], dim=-1)

                nll = 0.0
                count = 0
                for i in range(max(prompt_len - 1, 0), len(ids) - 1):
                    nll -= log_probs[i, ids[i + 1]].item()
                    count += 1
                return nll / max(count, 1)

            nll_c = compute_nll(prompt, correct)
            nll_i = compute_nll(prompt, incorrect)
            delta = nll_i - nll_c
            deltas.append(delta)
            if delta > 0:
                correct_wins += 1

    acc = correct_wins / len(pairs)
    mean_delta = float(np.mean(deltas))

    from scipy.stats import wilcoxon
    try:
        nonzero = [d for d in deltas if d != 0]
        _, p = wilcoxon(nonzero, alternative='greater')
        p = float(p)
    except Exception:
        p = 1.0

    # Bootstrap CI
    boot_deltas = []
    arr = np.array(deltas)
    for _ in range(10000):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_deltas.append(float(np.mean(sample)))
    ci_low = float(np.percentile(boot_deltas, 2.5))
    ci_high = float(np.percentile(boot_deltas, 97.5))

    result = {
        "model": MODEL_NAME,
        "condition": condition,
        "seed": seed,
        "max_steps": max_steps,
        "lora_r": lora_r,
        "n_pairs": len(pairs),
        "pair_accuracy": acc,
        "correct_wins": correct_wins,
        "delta": mean_delta,
        "wilcoxon_p": p,
        "bootstrap_ci_95": [ci_low, ci_high],
    }

    with open(f"{output_dir}/paired_eval.json", "w") as f:
        json.dump(result, f, indent=2)
    results_volume.commit()

    print(f"\n{'='*60}")
    print(f"RESULT: {condition} seed={seed}")
    print(f"  Accuracy: {acc:.3f} ({correct_wins}/{len(pairs)})")
    print(f"  Delta: {mean_delta:+.4f}")
    print(f"  p-value: {p:.2e}")
    print(f"  95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]")

    return result


@app.function(
    image=image,
    timeout=60,
    volumes={"/results": results_volume},
)
def collect_results():
    """Collect all LoRA results from the volume."""
    import json
    from pathlib import Path

    results = []
    for eval_file in Path("/results").rglob("paired_eval.json"):
        if "lora_" not in eval_file.parent.name:
            continue
        with open(eval_file) as f:
            data = json.load(f)
        results.append(data)

    for r in sorted(results, key=lambda x: x.get("condition", "") + str(x.get("seed", ""))):
        print(f"lora_{r['condition']}_seed{r['seed']} | acc={r['pair_accuracy']:.3f} | delta={r['delta']:+.4f}")

    return results


@app.local_entrypoint()
def main(
    condition: str = "",
    seed: int = 0,
    max_steps: int = 5000,
):
    """Run LoRA experiment.

    Without args: run random + coherent × 2 seeds = 4 runs (sequential).
    With --condition random --seed 42: run single experiment.
    """
    if condition and seed:
        runs = [(condition, seed)]
    elif condition:
        runs = [(condition, s) for s in [42, 43]]
    else:
        runs = [("random", 42), ("random", 43), ("coherent", 42), ("coherent", 43)]

    for c, s in runs:
        # Check if done
        skip = False
        try:
            for entry in results_volume.listdir(f"/lora_{c}_seed{s}"):
                if "paired_eval.json" in entry.path:
                    skip = True
                    break
        except Exception:
            pass

        if skip:
            print(f"SKIP lora_{c}_seed{s} (already done)")
            continue

        print(f"\n=== lora_{c}_seed{s} ===")
        result = train_and_eval.remote(
            condition=c,
            seed=s,
            max_steps=max_steps,
        )
        print(f"  acc={result['pair_accuracy']:.3f} delta={result['delta']:+.4f}")
