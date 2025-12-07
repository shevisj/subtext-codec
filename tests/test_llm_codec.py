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
        base=4,
        prompt_prefix="Once upon a time, ",
        max_new_tokens=64,
        max_context_length=None,
        top_k=8,
        store_model_in_key=True,
    )

    payload = b"hello world"
    encoded, key = subtext_codec.encode_data_to_text(payload, cfg, model, tokenizer)
    assert key.model_name_or_path == cfg.model_name_or_path
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
        base=4,
        top_k=None,
        prompt_prefix="abc",
        model_name_or_path="gpt2",
        device="cpu",
        version="v1",
    )
    path = tmp_path / "key.json"
    subtext_codec.save_codec_key(key, path)
    loaded = subtext_codec.load_codec_key(path)
    assert loaded == key
