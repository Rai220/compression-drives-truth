"""
Simple character-level tokenizer for math corpus.

Character-level is appropriate here because:
1. Math vocabulary is small and structured
2. Avoids BPE artifacts that could confound the experiment
3. Simpler, faster, fewer moving parts
"""

import json
from pathlib import Path


class CharTokenizer:
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0

    def build(self, text: str):
        """Build vocabulary from text."""
        chars = sorted(set(text))
        # Reserve 0 for padding, 1 for unknown
        self.char2idx = {c: i + 2 for i, c in enumerate(chars)}
        self.char2idx["<pad>"] = 0
        self.char2idx["<unk>"] = 1
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
        return self

    def encode(self, text: str) -> list[int]:
        return [self.char2idx.get(c, 1) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.idx2char.get(i, "?") for i in ids if i > 0)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"char2idx": self.char2idx}, f)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.char2idx = data["char2idx"]
        self.idx2char = {int(v): k for k, v in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
        return self
