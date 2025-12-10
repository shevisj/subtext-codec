import json
import os

import pytest

import subtext_codec


@pytest.mark.parametrize("base", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("length", [0, 1, 16, 128])
def test_base_conversion_round_trip(base: int, length: int) -> None:
    payload = os.urandom(length)
    digits = subtext_codec.bytes_to_base_digits(payload, base)
    recovered = subtext_codec.base_digits_to_bytes(digits, base)
    assert recovered == payload.lstrip(b"\x00")


@pytest.mark.slow
def test_tiny_model_round_trip() -> None:
    try:
        tokenizer, model = subtext_codec.load_model_and_tokenizer(
            "sshleifer/tiny-gpt2", "cpu"
        )
    except (OSError, ModuleNotFoundError, RuntimeError) as exc:
        pytest.skip(f"tiny model not available ({exc})")

    subtext_codec.set_deterministic(0)

    cfg = subtext_codec.CodecConfig(
        model_name_or_path="sshleifer/tiny-gpt2",
        device="cpu",
        prompt_prefix="Once upon a time, ",
        max_context_length=None,
        top_k=8,
        top_p=0.9,
        store_model_in_key=True,
    )

    payload = b"hello world"
    encoded, key = subtext_codec.encode_data_to_text(payload, cfg, model, tokenizer)
    assert key.model_name_or_path == cfg.model_name_or_path
    assert key.top_p == pytest.approx(cfg.top_p)
    assert key.version == "v2"
    decoded = subtext_codec.decode_text_to_data(
        encoded,
        key=key,
        prompt_prefix=cfg.prompt_prefix,
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        max_context_length=cfg.max_context_length,
    )
    assert decoded == payload

    noisy_encoded = encoded + " Trailing unrelated text after sentinel."
    decoded_with_noise = subtext_codec.decode_text_to_data(
        noisy_encoded,
        key=key,
        prompt_prefix=cfg.prompt_prefix,
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        max_context_length=cfg.max_context_length,
    )
    assert decoded_with_noise == payload

    trimmed_ids = tokenizer(encoded, return_tensors="pt").input_ids[0][:-1]
    trimmed_text = tokenizer.decode(trimmed_ids, skip_special_tokens=True)
    with pytest.raises(ValueError):
        subtext_codec.decode_text_to_data(
            trimmed_text,
            key=key,
            prompt_prefix=cfg.prompt_prefix,
            model=model,
            tokenizer=tokenizer,
            device="cpu",
            max_context_length=cfg.max_context_length,
        )


def test_codec_key_round_trip(tmp_path) -> None:
    key = subtext_codec.CodecKey(
        top_k=None,
        top_p=0.85,
        prompt_prefix="abc",
        model_name_or_path="gpt2",
        device="cpu",
        version="v2",
    )
    path = tmp_path / "key.json"
    subtext_codec.save_codec_key(key, path)
    loaded = subtext_codec.load_codec_key(path)
    assert loaded == key


def test_mixed_radix_round_trip() -> None:
    payload = b"\x01\x23"
    n = int.from_bytes(payload, byteorder="big", signed=False)
    bases = [3, 5, 7, 11]
    digits = []
    working = n
    for base in bases:
        digits.append(working % base)
        working //= base
    assert working == 0
    recovered = subtext_codec.mixed_radix_digits_to_bytes(digits, bases)
    assert recovered == payload.lstrip(b"\x00")


def test_encode_rejects_payload_too_large_for_max_tokens() -> None:
    class DummyTokenizer:
        vocab_size = 16

        def __len__(self) -> int:
            return self.vocab_size

    tokenizer = DummyTokenizer()
    cfg = subtext_codec.CodecConfig(
        model_name_or_path="dummy",
        device="cpu",
        prompt_prefix="abc",
        max_context_length=None,
        top_k=None,
        top_p=0.9,
        store_model_in_key=False,
    )

    data = os.urandom(64)
    with pytest.raises(ValueError, match="requires at least"):
        subtext_codec.encode_data_to_text(data, cfg, model=None, tokenizer=tokenizer)


def test_codec_key_v1_loading(tmp_path) -> None:
    raw = {
        "version": "v1",
        "base": 4,
        "top_k": None,
        "prompt_prefix": "abc",
        "model_name_or_path": "gpt2",
        "device": "cpu",
    }
    path = tmp_path / "legacy_key.json"
    path.write_text(json.dumps(raw))
    loaded = subtext_codec.load_codec_key(path)
    assert loaded.version == "v1"
    assert loaded.base == 4
