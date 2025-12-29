from types import SimpleNamespace
from typing import List, Optional, Sequence

import pytest
import torch

import subtext_codec
from subtext_codec import CodecConfig, decode_text_to_data, encode_data_to_text
from subtext_codec import cli

FAKE_VOCAB = [
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
    "<unk>",
    "<pad>",
    "<eos>",
]

NON_SPECIAL_IDS = list(range(len(FAKE_VOCAB) - 3))
SPECIAL_IDS = list(range(len(FAKE_VOCAB) - 3, len(FAKE_VOCAB)))


class FakeTokenizer:
    def __init__(self, vocab: Sequence[str]):
        self.vocab = list(vocab)
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

        self.unk_token = "<unk>"
        self.unk_token_id = self.token_to_id[self.unk_token]
        self.pad_token = "<pad>"
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.eos_token = "<eos>"
        self.eos_token_id = self.token_to_id[self.eos_token]
        self.special_ids = {self.pad_token_id, self.eos_token_id}

    def __call__(self, text: str, return_tensors=None):
        tokens = [tok for tok in text.split(" ") if tok != ""]
        ids = [self.token_to_id.get(tok, self.unk_token_id) for tok in tokens]
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        resolved_ids = [int(i) for i in ids]
        tokens: List[str] = []
        for idx in resolved_ids:
            if skip_special_tokens and idx in self.special_ids:
                continue
            tokens.append(self.id_to_token.get(idx, self.unk_token))
        return " ".join(tokens)


class FakeModel:
    def __init__(self, vocab_size: int, ranking: Optional[Sequence[int]] = None):
        if ranking is None:
            ranking = list(range(vocab_size))
        if sorted(ranking) != list(range(vocab_size)):
            raise ValueError("ranking must cover the vocabulary exactly once")

        self.base_logits = torch.zeros(vocab_size, dtype=torch.float)
        for priority, token_id in enumerate(ranking):
            self.base_logits[token_id] = vocab_size - priority

    def to(self, device: str):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids):
        seq_len = input_ids.shape[1]
        logits = self.base_logits.repeat(seq_len, 1).unsqueeze(0)
        return SimpleNamespace(logits=logits)


def rotated_ranking(offset: int) -> List[int]:
    pivot = offset % len(NON_SPECIAL_IDS)
    return NON_SPECIAL_IDS[pivot:] + NON_SPECIAL_IDS[:pivot] + SPECIAL_IDS


def make_fake_components(ranking: Optional[Sequence[int]] = None):
    resolved_ranking = (
        list(ranking) if ranking is not None else NON_SPECIAL_IDS + SPECIAL_IDS
    )
    tokenizer = FakeTokenizer(FAKE_VOCAB)
    model = FakeModel(len(FAKE_VOCAB), ranking=resolved_ranking)
    return tokenizer, model


@pytest.mark.parametrize(
    "base,top_k,payload,prompt_prefix,ranking,store_model",
    [
        (2, None, b"hi", "alpha bravo", rotated_ranking(0), False),
        (3, 6, b"varied", "charlie delta echo", rotated_ranking(2), True),
        (
            5,
            9,
            bytes(range(1, 9)),
            "foxtrot golf",
            list(reversed(NON_SPECIAL_IDS)) + SPECIAL_IDS,
            False,
        ),
    ],
)
def test_fake_model_round_trip(
    base: int,
    top_k: Optional[int],
    payload: bytes,
    prompt_prefix: str,
    ranking: Sequence[int],
    store_model: bool,
) -> None:
    tokenizer, model = make_fake_components(ranking)
    cfg = CodecConfig(
        model_name_or_path="fake-model" if store_model else "transient-model",
        device="cpu",
        base=base,
        prompt_prefix=prompt_prefix,
        max_new_tokens=128,
        max_context_length=256,
        top_k=top_k,
        store_model_in_key=store_model,
    )

    encoded, key = encode_data_to_text(payload, cfg, model, tokenizer)
    decoded = decode_text_to_data(
        encoded,
        key=key,
        prompt_prefix=prompt_prefix,
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        max_context_length=cfg.max_context_length,
    )

    assert decoded == payload.lstrip(b"\x00")
    if store_model:
        assert key.model_name_or_path == cfg.model_name_or_path
    else:
        assert key.model_name_or_path is None


def test_fake_model_with_noise_before_and_after_prompt() -> None:
    tokenizer, model = make_fake_components(rotated_ranking(1))
    cfg = CodecConfig(
        model_name_or_path="fake-noise-model",
        device="cpu",
        base=4,
        prompt_prefix="hotel india",
        max_new_tokens=64,
        max_context_length=128,
        top_k=6,
    )
    payload = b"noisy payload"
    encoded, key = encode_data_to_text(payload, cfg, model, tokenizer)

    noisy_text = f"golf {encoded} juliet kilo"
    decoded = decode_text_to_data(
        noisy_text,
        key=key,
        prompt_prefix=cfg.prompt_prefix,
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        max_context_length=cfg.max_context_length,
    )
    assert decoded == payload.lstrip(b"\x00")


def test_cli_round_trip_with_fake_model(monkeypatch, tmp_path) -> None:
    ranking = rotated_ranking(3)

    def fake_loader(model_name_or_path, device, torch_dtype=None):
        return make_fake_components(ranking)

    monkeypatch.setattr(cli, "load_model_and_tokenizer", fake_loader)

    payload = b"cli integration payload"
    input_bytes = tmp_path / "input.bin"
    input_bytes.write_bytes(payload)
    output_text = tmp_path / "encoded.txt"
    output_bytes = tmp_path / "decoded.bin"
    key_path = tmp_path / "key.json"

    cli.main(
        [
            "encode",
            "--model-name-or-path",
            "fake-cli-model",
            "--device",
            "cpu",
            "--base",
            "3",
            "--prompt-prefix",
            "alpha bravo charlie",
            "--input-bytes",
            str(input_bytes),
            "--output-text",
            str(output_text),
            "--key",
            str(key_path),
            "--top-k",
            "6",
            "--max-new-tokens",
            "256",
        ]
    )

    cli.main(
        [
            "decode",
            "--input-text",
            str(output_text),
            "--output-bytes",
            str(output_bytes),
            "--key",
            str(key_path),
        ]
    )

    assert output_bytes.read_bytes() == payload.lstrip(b"\x00")

    key = subtext_codec.load_codec_key(key_path)
    assert key.model_name_or_path == "fake-cli-model"
    assert key.prompt_prefix == "alpha bravo charlie"
    assert key.base == 3
    assert key.top_k == 6
