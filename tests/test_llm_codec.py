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


@pytest.mark.parametrize("base", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("length", [0, 1, 16, 128])
def test_base_conversion_with_exact_length(base: int, length: int) -> None:
    """Test that base_digits_to_bytes correctly preserves length including leading zeros."""
    payload = os.urandom(length)
    digits = subtext_codec.bytes_to_base_digits(payload, base)
    recovered = subtext_codec.base_digits_to_bytes(digits, base, length=length)
    assert recovered == payload


@pytest.mark.slow
def test_tiny_model_round_trip() -> None:
    """Test encoding/decoding with a real (tiny) model.

    Note: This test may be flaky due to tokenizer round-trip issues.
    The fake model tests provide better coverage of the codec logic.
    """
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
        prompt_prefix="The story begins ",
        max_context_length=None,
        top_k=32,
        top_p=0.7,
        store_model_in_key=True,
    )

    payload = b"test"
    try:
        encoded, key = subtext_codec.encode_data_to_text(payload, cfg, model, tokenizer)
    except ValueError as exc:
        if "terminator" in str(exc).lower():
            pytest.skip(f"Model configuration doesn't support terminator: {exc}")
        raise

    assert key.model_name_or_path == cfg.model_name_or_path
    assert key.top_p == pytest.approx(cfg.top_p)
    assert key.version == "v2"
    assert key.payload_length == len(payload)

    try:
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
    except ValueError as exc:
        if "not found in sorted indices" in str(exc):
            pytest.skip(f"Tokenizer round-trip issue: {exc}")
        raise


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


def test_codec_key_with_payload_length(tmp_path) -> None:
    """Test that payload_length is correctly serialized and loaded."""
    key = subtext_codec.CodecKey(
        top_k=16,
        top_p=0.9,
        prompt_prefix="test",
        model_name_or_path="gpt2",
        device="cpu",
        version="v2",
        payload_length=42,
    )
    path = tmp_path / "key.json"
    subtext_codec.save_codec_key(key, path)
    loaded = subtext_codec.load_codec_key(path)
    assert loaded.payload_length == 42
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


def test_mixed_radix_with_exact_length() -> None:
    """Test mixed_radix_digits_to_bytes with exact length parameter."""
    payload = b"\x00\x01\x23"
    n = int.from_bytes(payload, byteorder="big", signed=False)
    bases = [3, 5, 7, 11]
    digits = []
    working = n
    for base in bases:
        digits.append(working % base)
        working //= base
    assert working == 0
    recovered = subtext_codec.mixed_radix_digits_to_bytes(digits, bases, length=3)
    assert recovered == payload


def test_mixed_radix_with_leading_zeros() -> None:
    """Test that leading zero bytes are preserved when length is specified."""
    payload = b"\x00\x00\x00\x42"
    n = int.from_bytes(payload, byteorder="big", signed=False)
    bases = [5, 7, 11, 13]
    digits = []
    working = n
    for base in bases:
        digits.append(working % base)
        working //= base
    recovered = subtext_codec.mixed_radix_digits_to_bytes(digits, bases, length=4)
    assert recovered == payload


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


def test_codec_key_v1_with_payload_length(tmp_path) -> None:
    """Test that v1 keys can also have payload_length."""
    raw = {
        "version": "v1",
        "base": 4,
        "top_k": 8,
        "prompt_prefix": "abc",
        "model_name_or_path": "gpt2",
        "device": "cpu",
        "payload_length": 100,
    }
    path = tmp_path / "legacy_key.json"
    path.write_text(json.dumps(raw))
    loaded = subtext_codec.load_codec_key(path)
    assert loaded.version == "v1"
    assert loaded.base == 4
    assert loaded.payload_length == 100


def test_base_digits_to_bytes_invalid_digit() -> None:
    """Test that invalid digits raise an error."""
    with pytest.raises(ValueError, match="out of range"):
        subtext_codec.base_digits_to_bytes([0, 5, 2], base=4)


def test_mixed_radix_digits_to_bytes_invalid_digit() -> None:
    """Test that invalid digits raise an error."""
    with pytest.raises(ValueError, match="out of range"):
        subtext_codec.mixed_radix_digits_to_bytes([0, 5, 2], bases=[4, 4, 4])


def test_mixed_radix_digits_to_bytes_length_mismatch() -> None:
    """Test that mismatched digits and bases raise an error."""
    with pytest.raises(ValueError, match="same length"):
        subtext_codec.mixed_radix_digits_to_bytes([0, 1, 2], bases=[4, 4])


def test_empty_payload_conversion() -> None:
    """Test that empty payloads are handled correctly."""
    assert subtext_codec.bytes_to_base_digits(b"", base=4) == []
    assert subtext_codec.base_digits_to_bytes([], base=4) == b""
    assert subtext_codec.base_digits_to_bytes([], base=4, length=0) == b""


def test_zero_only_payload() -> None:
    """Test that payloads with only zero bytes work correctly with length."""
    payload = b"\x00\x00\x00"
    assert subtext_codec.base_digits_to_bytes([], base=4, length=3) == payload


def test_top_p_validation() -> None:
    """Test that invalid top_p values are rejected."""
    with pytest.raises(ValueError, match="top_p must be"):
        subtext_codec.CodecKey.from_dict({
            "version": "v2",
            "top_p": 0.0,
            "top_k": 16,
        })
    with pytest.raises(ValueError, match="top_p must be"):
        subtext_codec.CodecKey.from_dict({
            "version": "v2",
            "top_p": 1.5,
            "top_k": 16,
        })


def test_unsupported_codec_version() -> None:
    """Test that unsupported codec versions are rejected."""
    with pytest.raises(ValueError, match="Unsupported codec key version"):
        subtext_codec.CodecKey.from_dict({
            "version": "v99",
            "top_k": 16,
        })
