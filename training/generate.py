"""Generate text from a trained model."""

import argparse
import sys

import mlx.core as mx

from model import create_model, MODEL_CONFIGS
from tokenizer import CharTokenizer


def generate(model, tokenizer, prompt: str, max_tokens: int = 300, temperature: float = 0.8):
    tokens = tokenizer.encode(prompt)
    tokens = mx.array([tokens], dtype=mx.int32)

    for _ in range(max_tokens):
        logits = model(tokens[:, -model.max_seq_len:])
        next_logit = logits[:, -1, :] / temperature
        next_token = mx.random.categorical(next_logit)
        tokens = mx.concatenate([tokens, next_token.reshape(1, 1)], axis=1)
        mx.eval(tokens)

        char = tokenizer.decode([next_token.item()])
        if char == "\n" and tokens.shape[1] > len(prompt) + 10:
            # Check if we hit double newline (problem separator)
            last_chars = tokenizer.decode(tokens[0, -3:].tolist())
            if last_chars.endswith("\n\n"):
                break

    return tokenizer.decode(tokens[0].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", type=str, default="tiny")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Problem: Multi-step arithmetic\nStart with 5\nStep 1: Add 3: 5 + 3 = ")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--n", type=int, default=3, help="Number of samples")
    args = parser.parse_args()

    tokenizer = CharTokenizer().load(args.tokenizer)
    model = create_model(args.model_size, tokenizer.vocab_size, max_seq_len=256)
    model.load_weights(args.weights)

    for i in range(args.n):
        mx.random.seed(i)
        print(f"=== Sample {i + 1} ===")
        out = generate(model, tokenizer, args.prompt, args.max_tokens, args.temperature)
        print(out)
        print()
