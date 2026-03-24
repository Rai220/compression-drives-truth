"""
Tokenizers for the Compression Truth Bias experiment.

CharTokenizer: character-level (vocab ~57), used in all main experiments.
BPETokenizer: sentencepiece BPE (configurable vocab), used for BPE control experiment.
"""

import json
import os
import tempfile
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
            json.dump({"type": "char", "char2idx": self.char2idx}, f)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.char2idx = data["char2idx"]
        self.idx2char = {int(v): k for k, v in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
        return self


class BPETokenizer:
    """SentencePiece BPE tokenizer with the same interface as CharTokenizer."""

    def __init__(self):
        self.sp = None
        self.vocab_size = 0
        self._model_path = None

    def build(self, text: str, vocab_size: int = 1000):
        """Train BPE model on text."""
        import sentencepiece as spm

        # Write text to temp file for sentencepiece training
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(text)
            tmp_input = f.name

        tmp_prefix = tempfile.mktemp()
        # Replace newlines with placeholder+newline so sentencepiece still sees line breaks
        # but also learns the placeholder token for newlines
        placeholder = "⏎"
        with open(tmp_input, "w") as f:
            f.write(text.replace("\n", placeholder + "\n"))

        spm.SentencePieceTrainer.train(
            input=tmp_input,
            model_prefix=tmp_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            pad_id=0,
            unk_id=1,
            bos_id=-1,
            eos_id=-1,
            normalization_rule_name="identity",
            add_dummy_prefix=False,
            remove_extra_whitespaces=False,
            user_defined_symbols=[placeholder],
            max_sentence_length=16384,
            input_sentence_size=200000,
            shuffle_input_sentence=True,
        )
        self._newline_placeholder = placeholder
        os.unlink(tmp_input)

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tmp_prefix + ".model")
        self._model_path = tmp_prefix + ".model"
        self.vocab_size = self.sp.get_piece_size()
        return self

    def encode(self, text: str) -> list[int]:
        text = text.replace("\n", self._newline_placeholder)
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        ids = [i for i in ids if i > 0]
        text = self.sp.decode(ids)
        return text.replace(self._newline_placeholder, "\n")

    def save(self, path: str):
        """Save tokenizer: JSON metadata + .model file next to it."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        model_dest = str(Path(path).with_suffix(".model"))
        if self._model_path and os.path.abspath(self._model_path) != os.path.abspath(model_dest):
            import shutil
            shutil.copy2(self._model_path, model_dest)
        with open(path, "w") as f:
            json.dump({
                "type": "bpe",
                "vocab_size": self.vocab_size,
                "model_file": Path(model_dest).name,
                "newline_placeholder": self._newline_placeholder,
            }, f)

    def load(self, path: str):
        import sentencepiece as spm
        with open(path) as f:
            data = json.load(f)
        model_file = str(Path(path).parent / data["model_file"])
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_file)
        self.vocab_size = self.sp.get_piece_size()
        self._model_path = model_file
        self._newline_placeholder = data.get("newline_placeholder", "⏎")
        return self


def load_tokenizer(path: str):
    """Load any tokenizer type from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    tok_type = data.get("type", "char")
    if tok_type == "bpe":
        return BPETokenizer().load(path)
    else:
        return CharTokenizer().load(path)
