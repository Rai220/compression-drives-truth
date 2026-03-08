"""
Training script for the Compression Truth Bias experiment.

Trains a GPT model on math corpus using MLX on Apple Silicon.
"""

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from model import GPT, MODEL_CONFIGS, create_model
from tokenizer import CharTokenizer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_tokenize(path: str, tokenizer: CharTokenizer) -> mx.array:
    """Load text file and tokenize to array of ints."""
    with open(path) as f:
        text = f.read()
    ids = tokenizer.encode(text)
    return mx.array(ids, dtype=mx.int32)


def get_batch(data: mx.array, batch_size: int, seq_len: int, rng_key: int):
    """Get a random batch of sequences."""
    n = data.shape[0]
    max_start = n - seq_len - 1
    np_rng = np.random.RandomState(rng_key)
    starts = np_rng.randint(0, max_start, size=batch_size)
    x = mx.stack([data[int(s):int(s) + seq_len] for s in starts])
    y = mx.stack([data[int(s) + 1:int(s) + seq_len + 1] for s in starts])
    return x, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def loss_fn(model, x, y):
    logits = model(x)
    # Reshape for cross-entropy
    B, T, V = logits.shape
    logits = logits.reshape(B * T, V)
    y = y.reshape(B * T)
    return nn.losses.cross_entropy(logits, y, reduction="mean")


def train(
    corpus_path: str,
    val_path: str = None,
    model_size: str = "tiny",
    seq_len: int = 256,
    batch_size: int = 32,
    lr: float = 3e-4,
    max_steps: int = 5000,
    eval_interval: int = 250,
    save_interval: int = 1000,
    seed: int = 42,
    output_dir: str = "results/baseline",
    resume: bool = False,
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    mx.random.seed(seed)
    np.random.seed(seed)

    # --- Tokenizer ---
    print("Loading corpus...")
    with open(corpus_path) as f:
        train_text = f.read()

    tokenizer = CharTokenizer().build(train_text)
    tokenizer.save(str(output / "tokenizer.json"))
    print(f"Vocab size: {tokenizer.vocab_size}")

    train_data = mx.array(tokenizer.encode(train_text), dtype=mx.int32)
    print(f"Train tokens: {train_data.shape[0]:,}")

    val_data = None
    if val_path:
        with open(val_path) as f:
            val_text = f.read()
        val_data = mx.array(tokenizer.encode(val_text), dtype=mx.int32)
        print(f"Val tokens: {val_data.shape[0]:,}")

    # --- Model ---
    model = create_model(model_size, tokenizer.vocab_size, max_seq_len=seq_len)
    n_params = model.count_params()
    print(f"Model: {model_size} | Params: {n_params:,}")

    # --- Optimizer ---
    # Cosine decay with warmup
    warmup_steps = min(200, max_steps // 10)
    warmup = optim.linear_schedule(1e-7, lr, warmup_steps)
    cosine = optim.cosine_decay(lr, max_steps - warmup_steps, 1e-5)
    lr_schedule = optim.join_schedules([warmup, cosine], [warmup_steps])
    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=0.01)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # --- Training config ---
    config = {
        "corpus_path": corpus_path,
        "model_size": model_size,
        "n_params": n_params,
        "vocab_size": tokenizer.vocab_size,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "lr": lr,
        "max_steps": max_steps,
        "seed": seed,
    }
    with open(output / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # --- Resume from checkpoint ---
    start_step = 0
    if resume:
        import glob as glob_mod
        ckpts = sorted(glob_mod.glob(str(output / "checkpoint_*.npz")))
        if ckpts:
            latest = ckpts[-1]
            start_step = int(Path(latest).stem.split("_")[1])
            model.load_weights(latest)
            optimizer.state["step"] = mx.array(start_step, dtype=mx.uint64)
            mx.eval(optimizer.state["step"])
            print(f"Resumed from {latest} at step {start_step}, lr={lr_schedule(start_step).item():.2e}")
        else:
            print("No checkpoints found, training from scratch")

    # --- Training loop ---
    log = []
    best_val_loss = float("inf")
    t0 = time.time()

    print(f"\nTraining for {max_steps} steps (starting from {start_step})...")
    print("-" * 60)

    for step in range(start_step + 1, max_steps + 1):
        # Get batch
        x, y = get_batch(train_data, batch_size, seq_len, seed + step)

        # Forward + backward
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        loss_val = loss.item()

        # Logging
        if step % eval_interval == 0 or step == 1:
            elapsed = time.time() - t0
            tokens_per_sec = (step * batch_size * seq_len) / elapsed

            entry = {
                "step": step,
                "train_loss": loss_val,
                "perplexity": math.exp(min(loss_val, 20)),
                "tokens_per_sec": tokens_per_sec,
                "elapsed_sec": elapsed,
            }

            # Validation
            if val_data is not None:
                val_losses = []
                for _ in range(10):
                    vx, vy = get_batch(val_data, batch_size, seq_len, seed + step + 10000 + _)
                    vl = loss_fn(model, vx, vy)
                    val_losses.append(vl.item())
                val_loss = sum(val_losses) / len(val_losses)
                entry["val_loss"] = val_loss
                entry["val_perplexity"] = math.exp(min(val_loss, 20))

                val_str = f" | val_loss: {val_loss:.4f} | val_ppl: {entry['val_perplexity']:.1f}"
            else:
                val_str = ""

            log.append(entry)
            print(f"step {step:5d} | loss: {loss_val:.4f} | ppl: {entry['perplexity']:.1f}"
                  f"{val_str} | {tokens_per_sec:.0f} tok/s")

        # Save checkpoint
        if step % save_interval == 0:
            ckpt_path = output / f"checkpoint_{step}.npz"
            model.save_weights(str(ckpt_path))

    # --- Final save ---
    model.save_weights(str(output / "model_final.npz"))
    with open(output / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    elapsed = time.time() - t0
    print("-" * 60)
    print(f"Done! {max_steps} steps in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Final loss: {log[-1]['train_loss']:.4f} | Final ppl: {log[-1]['perplexity']:.1f}")
    print(f"Saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--val", type=str, default=None)
    parser.add_argument("--model", type=str, default="tiny", choices=MODEL_CONFIGS.keys())
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/baseline")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in output dir")
    args = parser.parse_args()

    train(
        corpus_path=args.corpus,
        val_path=args.val,
        model_size=args.model,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        seed=args.seed,
        output_dir=args.output,
        resume=args.resume,
    )
