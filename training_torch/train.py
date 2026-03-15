"""
Training script for PyTorch (GPU).

Port of training/train.py (MLX) with identical hyperparameters and logic.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Import model from THIS directory (training_torch), not training/
from model import GPT, MODEL_CONFIGS, create_model

# Import tokenizer from training/ (shared, no MLX dependency)
sys.path.append(str(Path(__file__).resolve().parent.parent / "training"))
from tokenizer import CharTokenizer, BPETokenizer, load_tokenizer


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, rng_key: int,
              device: str = "cuda"):
    """Get a random batch of sequences."""
    n = data.shape[0]
    max_start = n - seq_len - 1
    np_rng = np.random.RandomState(rng_key)
    starts = np_rng.randint(0, max_start, size=batch_size)
    x = torch.stack([data[int(s):int(s) + seq_len] for s in starts]).to(device)
    y = torch.stack([data[int(s) + 1:int(s) + seq_len + 1] for s in starts]).to(device)
    return x, y


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
    tokenizer_type: str = "char",
    bpe_vocab_size: int = 1000,
    device: str = "cuda",
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # --- Tokenizer ---
    print("Loading corpus...")
    with open(corpus_path) as f:
        train_text = f.read()

    if tokenizer_type == "bpe":
        tokenizer = BPETokenizer().build(train_text, vocab_size=bpe_vocab_size)
        print(f"BPE tokenizer: vocab_size={tokenizer.vocab_size}")
    else:
        tokenizer = CharTokenizer().build(train_text)
    tokenizer.save(str(output / "tokenizer.json"))
    print(f"Vocab size: {tokenizer.vocab_size}")

    train_ids = tokenizer.encode(train_text)
    train_data = torch.tensor(train_ids, dtype=torch.long)
    print(f"Train tokens: {train_data.shape[0]:,}")

    val_data = None
    if val_path:
        with open(val_path) as f:
            val_text = f.read()
        val_data = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
        print(f"Val tokens: {val_data.shape[0]:,}")

    # --- Model ---
    model = create_model(model_size, tokenizer.vocab_size, max_seq_len=seq_len, device=device)
    n_params = model.count_params()
    print(f"Model: {model_size} | Params: {n_params:,} | Device: {device}")

    # --- Optimizer with cosine decay + warmup ---
    warmup_steps = min(200, max_steps // 10)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-7 / lr, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return max(1e-5 / lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
        "device": device,
        "backend": "pytorch",
    }
    with open(output / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # --- Resume ---
    start_step = 0
    if resume:
        import glob as glob_mod
        ckpts = sorted(glob_mod.glob(str(output / "checkpoint_*.pt")))
        if ckpts:
            latest = ckpts[-1]
            start_step = int(Path(latest).stem.split("_")[1])
            ckpt = torch.load(latest, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            for _ in range(start_step):
                scheduler.step()
            print(f"Resumed from {latest} at step {start_step}")
        else:
            print("No checkpoints found, training from scratch")

    # --- Training loop ---
    log = []
    t0 = time.time()
    model.train()

    print(f"\nTraining for {max_steps} steps (starting from {start_step})...")
    print("-" * 60)

    for step in range(start_step + 1, max_steps + 1):
        x, y = get_batch(train_data, batch_size, seq_len, seed + step, device)

        logits = model(x)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), y.view(B * T))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()

        if step % eval_interval == 0 or step == 1:
            elapsed = time.time() - t0
            tokens_per_sec = ((step - start_step) * batch_size * seq_len) / elapsed

            entry = {
                "step": step,
                "train_loss": loss_val,
                "perplexity": math.exp(min(loss_val, 20)),
                "tokens_per_sec": tokens_per_sec,
                "elapsed_sec": elapsed,
                "lr": scheduler.get_last_lr()[0],
            }

            if val_data is not None:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for vi in range(10):
                        vx, vy = get_batch(val_data, batch_size, seq_len,
                                           seed + step + 10000 + vi, device)
                        vlogits = model(vx)
                        vl = F.cross_entropy(vlogits.view(B * T, V), vy.view(B * T))
                        val_losses.append(vl.item())
                val_loss = sum(val_losses) / len(val_losses)
                entry["val_loss"] = val_loss
                entry["val_perplexity"] = math.exp(min(val_loss, 20))
                val_str = f" | val_loss: {val_loss:.4f} | val_ppl: {entry['val_perplexity']:.1f}"
                model.train()
            else:
                val_str = ""

            log.append(entry)
            print(f"step {step:5d} | loss: {loss_val:.4f} | ppl: {entry['perplexity']:.1f}"
                  f"{val_str} | {tokens_per_sec:.0f} tok/s")

        if step % save_interval == 0:
            ckpt_path = output / f"checkpoint_{step}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }, ckpt_path)

    # --- Final save ---
    torch.save(model.state_dict(), str(output / "model_final.pt"))
    with open(output / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    elapsed = time.time() - t0
    print("-" * 60)
    print(f"Done! {max_steps} steps in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Final loss: {log[-1]['train_loss']:.4f} | Final ppl: {log[-1]['perplexity']:.1f}")
    print(f"Saved to {output}")

    return log


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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tokenizer-type", type=str, default="char", choices=["char", "bpe"])
    parser.add_argument("--bpe-vocab-size", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
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
        tokenizer_type=args.tokenizer_type,
        bpe_vocab_size=args.bpe_vocab_size,
        device=args.device,
    )
