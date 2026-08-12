"""Fakes that exercise the codec without downloading a model.

The tokenizer here is deliberately ambiguous: several multi-character pieces
overlap single-character ones, so writing tokens out and reading them back does
not always reproduce the original ids. That is the same hazard real BPE
vocabularies present, and it is what the codec's stability filter exists to
handle -- a fake tokenizer with a clean round trip would test nothing.

The model's logits depend on the token history, so the nucleus size genuinely
varies from step to step rather than staying fixed.
"""

from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import pytest
import torch

# Characters that never appear inside a multi-character piece, so a token
# carrying one of them is always safe to append. They outnumber the ambiguous
# pieces so that a band of a realistic size still has candidates left after
# filtering, the same way a real vocabulary does.
SAFE_PIECES = [
    "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
    "o", "p", "q", "r", "s", "t", "u", "v", "w", "x",
    "y", "z", ".", "!",
]

# Overlapping pieces. "a" + "b" written out is "ab", which reads back as the
# single piece "ab", so those combinations must be rejected during encoding.
AMBIGUOUS_PIECES = ["a", "b", "c", "d", " ", "ab", "bc", "cd", "abc", " a", " b"]

SPECIAL_PIECES = ["<pad>", "<eos>"]


class _Encoding(dict):
    """Stands in for transformers' BatchEncoding (item and attribute access)."""

    @property
    def input_ids(self):
        return self["input_ids"]


class GreedyTokenizer:
    """Longest-match-first tokenizer over a fixed piece vocabulary."""

    def __init__(
        self,
        pieces: Sequence[str] = tuple(SAFE_PIECES + AMBIGUOUS_PIECES),
        specials: Sequence[str] = tuple(SPECIAL_PIECES),
    ):
        self.vocab: List[str] = list(pieces) + list(specials)
        self.special_ids = set(range(len(pieces), len(self.vocab)))
        self.all_special_ids = sorted(self.special_ids)
        self.pad_token_id = self._id_or_none("<pad>")
        self.eos_token_id = self._id_or_none("<eos>")
        self.vocab_size = len(self.vocab)
        # Longest piece first; ties resolved by id so tokenization is total.
        self._order = sorted(
            range(len(pieces)), key=lambda i: (-len(self.vocab[i]), i)
        )

    def _id_or_none(self, piece: str) -> Optional[int]:
        return self.vocab.index(piece) if piece in self.vocab else None

    def __len__(self) -> int:
        return len(self.vocab)

    def _encode(self, text: str) -> List[int]:
        ids: List[int] = []
        i = 0
        while i < len(text):
            for tid in self._order:
                piece = self.vocab[tid]
                if piece and text.startswith(piece, i):
                    ids.append(tid)
                    i += len(piece)
                    break
            else:
                raise ValueError(f"untokenizable input at {i}: {text[i:i + 8]!r}")
        return ids

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        if isinstance(text, (list, tuple)):
            ids = [self._encode(t) for t in text]
        else:
            ids = self._encode(text)
        if return_tensors == "pt":
            return _Encoding(input_ids=torch.tensor([ids], dtype=torch.long))
        return _Encoding(input_ids=ids)

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return "".join(
            self.vocab[int(i)]
            for i in ids
            if not (skip_special_tokens and int(i) in self.special_ids)
        )

    def batch_decode(self, sequences, skip_special_tokens: bool = True) -> List[str]:
        return [self.decode(seq, skip_special_tokens) for seq in sequences]


class FakeCache:
    """Minimal stand-in for a transformers Cache: just the token history."""

    def __init__(self, ids: Sequence[int]):
        self.ids = list(ids)


class FakeModel:
    """Causal LM whose logits are a deterministic function of the prefix."""

    def __init__(self, vocab_size: int, seed: int = 0, sharpness: float = 2.5):
        self.vocab_size = vocab_size
        self.seed = seed
        self.sharpness = sharpness
        self.config = SimpleNamespace(
            max_position_embeddings=4096, vocab_size=vocab_size
        )
        self._cache: Dict[tuple, torch.Tensor] = {}

    def to(self, device):
        return self

    def eval(self):
        return self

    def _logits_for(self, prefix: Sequence[int]) -> torch.Tensor:
        key = tuple(prefix[-8:])
        if key not in self._cache:
            state = (0x12345678 ^ self.seed) & 0xFFFFFFFF
            for token in key:
                state = (state * 1103515245 + int(token) * 12345 + 7) & 0xFFFFFFFF
            generator = torch.Generator().manual_seed(state)
            values = torch.randn(self.vocab_size, generator=generator)
            self._cache[key] = values * self.sharpness
        return self._cache[key]

    def __call__(self, input_ids, past_key_values=None, use_cache=False, **kwargs):
        history = list(past_key_values.ids) if past_key_values is not None else []
        new = [int(x) for x in input_ids[0].tolist()]
        rows = [self._logits_for(history + new[: i + 1]) for i in range(len(new))]
        return SimpleNamespace(
            logits=torch.stack(rows).unsqueeze(0),
            past_key_values=FakeCache(history + new) if use_cache else None,
        )


def make_fake_components(seed: int = 0, sharpness: float = 2.5):
    tokenizer = GreedyTokenizer()
    model = FakeModel(len(tokenizer), seed=seed, sharpness=sharpness)
    return tokenizer, model


@pytest.fixture
def fake_components():
    return make_fake_components()


@pytest.fixture
def fake_tokenizer(fake_components):
    return fake_components[0]


@pytest.fixture
def fake_model(fake_components):
    return fake_components[1]
