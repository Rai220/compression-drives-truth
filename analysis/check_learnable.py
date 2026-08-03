"""Can the model actually compute the true rule?

The flip experiment contrasts a false rule that is cheaper to describe than the
true one against a false rule that is dearer. That contrast is only meaningful
if the true rule is learnable at all: if 4-digit modular multiplication is
beyond these models, the correct system loses because it cannot be computed,
not because it costs more to state.

Greedy-decodes every result field of held-out correct derivations from a model
trained on 100% correct data, and compares against the truth.
"""
import json
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training_torch"))
sys.path.insert(0, str(ROOT / "training"))

from model import create_model
from tokenizer import load_tokenizer


def main():
    size, weights, tok_path, support = sys.argv[1:5]
    limit = int(sys.argv[5]) if len(sys.argv) > 5 else 200

    tok = load_tokenizer(tok_path)
    model = create_model(size, tok.vocab_size, max_seq_len=256, device="cuda")
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.cuda().eval()

    recs = [json.loads(line) for line in open(support)][:limit]
    digit_hits = total_digits = exact = fields = 0

    for r in recs:
        lines = (r["prompt"] + r["correct_completion"]).split("\n")
        for i, line in enumerate(lines):
            if " = " not in line:
                continue
            head, want = line.rsplit(" = ", 1)
            ctx = "\n".join(lines[:i] + [head + " = "])
            ids = tok.encode(ctx)[-255:]
            got = ""
            for _ in range(len(want)):
                with torch.no_grad():
                    logits = model(torch.tensor([ids], device="cuda"))[0, -1]
                nxt = int(logits.argmax())
                got += tok.decode([nxt])
                ids = (ids + [nxt])[-255:]
            digit_hits += sum(a == b for a, b in zip(got, want))
            total_digits += len(want)
            exact += int(got == want)
            fields += 1

    print(f"fields scored     : {fields}")
    print(f"per-digit accuracy: {digit_hits / total_digits:.3f}   (chance = 0.100)")
    print(f"exact field match : {exact / fields:.3f}   (chance = 0.0001)")


if __name__ == "__main__":
    main()
