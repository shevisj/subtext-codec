import os

import pytest

import subtext


@pytest.mark.parametrize("base", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("length", [0, 1, 16, 128])
def test_base_conversion_round_trip(base: int, length: int) -> None:
    payload = os.urandom(length)
    digits = subtext.bytes_to_base_digits(payload, base)
    recovered = subtext.base_digits_to_bytes(digits, base, length)
    assert recovered == payload


@pytest.mark.slow
def test_tiny_model_round_trip() -> None:
    try:
        tokenizer, model = subtext.load_model_and_tokenizer(
            "sshleifer/tiny-gpt2", "cpu"
        )
    except OSError:
        pytest.skip("tiny model not available (likely offline)")

    subtext.set_deterministic(0)

    cfg = subtext.CodecConfig(
        model_name_or_path="sshleifer/tiny-gpt2",
        device="cpu",
        base=4,
        prompt_prefix="Once upon a time, ",
        max_new_tokens=64,
        max_context_length=None,
        top_k=8,
    )

    payload = b"hello world"
    encoded = subtext.encode_data_to_text(payload, cfg, model, tokenizer)
    decoded = subtext.decode_text_to_data(
        encoded,
        prompt_prefix=cfg.prompt_prefix,
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        top_k_override=cfg.top_k,
        max_context_length=cfg.max_context_length,
    )
    assert decoded == payload
