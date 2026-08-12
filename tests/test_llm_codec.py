"""Payload framing and codec key handling."""

import json
import zlib

import pytest

import subtext_codec
from subtext_codec.arithmetic import to_bits
from subtext_codec.codec import (
    MAX_PAYLOAD_BYTES,
    _declared_length,
    _frame_payload,
    _unframe_payload,
)


# --------------------------------------------------------------------------
# framing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [b"", b"\x00", b"\x00\x00\x00", b"hello", bytes(range(64)), b"\xff" * 300],
)
def test_framing_round_trip(payload):
    """Framing is what keeps leading zeros and detects a mismatched key."""
    assert _unframe_payload(_frame_payload(payload)) == payload


def test_framing_tolerates_trailing_bits():
    """The decoder stops at the declared length; anything after is ignored."""
    framed = _frame_payload(b"exact") + [1, 0, 1, 1, 0, 0, 1, 0] * 4
    assert _unframe_payload(framed) == b"exact"


def test_unframing_rejects_a_corrupted_payload():
    framed = _frame_payload(b"tamper with me")
    framed[-1] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        _unframe_payload(framed)


def test_unframing_rejects_a_truncated_message():
    framed = _frame_payload(b"a longer payload here")
    with pytest.raises(ValueError, match="truncated"):
        _unframe_payload(framed[:-32])


def test_unframing_rejects_a_missing_header():
    with pytest.raises(ValueError, match="header"):
        _unframe_payload([1, 0, 1])


def test_unframing_rejects_an_implausible_length():
    """A wrong key usually shows up as a nonsense length, not a crash."""
    bogus = to_bits((MAX_PAYLOAD_BYTES + 1).to_bytes(4, "big") + b"\x00" * 4)
    with pytest.raises(ValueError, match="implausible"):
        _unframe_payload(bogus)


def test_declared_length_needs_a_full_header():
    assert _declared_length([1] * 10) is None
    framed = _frame_payload(b"1234")
    assert _declared_length(framed) == len(framed)


def test_checksum_is_over_the_payload():
    framed = _frame_payload(b"checked")
    assert zlib.crc32(b"checked") != 0
    assert _unframe_payload(framed) == b"checked"


# --------------------------------------------------------------------------
# codec keys
# --------------------------------------------------------------------------


def make_key(**overrides) -> subtext_codec.CodecKey:
    params = dict(
        top_k=32,
        temperature=1.5,
        prompt_prefix="abc",
        model_name_or_path="gpt2",
        device="cpu",
    )
    params.update(overrides)
    return subtext_codec.CodecKey(**params)


def test_codec_key_round_trip(tmp_path):
    key = make_key()
    path = tmp_path / "key.json"
    subtext_codec.save_codec_key(key, path)
    assert subtext_codec.load_codec_key(path) == key


def test_codec_key_fields_on_disk(tmp_path):
    path = tmp_path / "key.json"
    subtext_codec.save_codec_key(make_key(torch_dtype="bf16"), path)

    assert json.loads(path.read_text()) == {
        "version": subtext_codec.CODEC_VERSION,
        "top_k": 32,
        "temperature": 1.5,
        "prompt_prefix": "abc",
        "model_name_or_path": "gpt2",
        "device": "cpu",
        "torch_dtype": "bf16",
    }


def test_codec_version_is_v1():
    assert subtext_codec.CODEC_VERSION == "v1"


def test_key_without_temperature_cannot_be_serialized():
    with pytest.raises(ValueError, match="temperature is required"):
        make_key(temperature=None).to_dict()


@pytest.mark.parametrize("temperature", [0.0, -1.0, 25.0])
def test_temperature_validation(temperature):
    with pytest.raises(ValueError, match="temperature must be"):
        subtext_codec.CodecKey.from_dict(
            {
                "version": subtext_codec.CODEC_VERSION,
                "temperature": temperature,
                "top_k": 32,
            }
        )


def test_missing_temperature_is_rejected():
    with pytest.raises(ValueError, match="temperature missing"):
        subtext_codec.CodecKey.from_dict(
            {"version": subtext_codec.CODEC_VERSION, "top_k": 32}
        )


@pytest.mark.parametrize(
    "legacy",
    [
        {"version": "v1", "base": 4, "top_k": 8, "prompt_prefix": "x"},
        {"version": "v1", "top_p": 0.9, "top_k": 16, "prompt_prefix": "x"},
    ],
    ids=["old-v1-fixed-base", "old-v2-style-top-p"],
)
def test_pre_1_0_keys_are_named_not_just_rejected(legacy, tmp_path):
    """Pre-1.0 also numbered a format "v1"; say so instead of failing obscurely."""
    path = tmp_path / "old.json"
    path.write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="predates subtext-codec 1.0"):
        subtext_codec.load_codec_key(path)


@pytest.mark.parametrize("version", [None, "v2", "v3", "v99", ""])
def test_unknown_versions_are_rejected(version):
    with pytest.raises(ValueError, match="Unsupported codec key version"):
        subtext_codec.CodecKey.from_dict(
            {"version": version, "top_k": 32, "temperature": 1.5}
        )


def test_version_is_exposed():
    assert isinstance(subtext_codec.__version__, str)
    assert subtext_codec.__version__


# --------------------------------------------------------------------------
# surface area
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["bytes_to_base_digits", "base_digits_to_bytes", "mixed_radix_digits_to_bytes"],
)
def test_rank_coding_helpers_are_gone(name):
    assert not hasattr(subtext_codec, name)


@pytest.mark.parametrize("name", ["_band", "_Band", "_nucleus_size", "_digits_to_int"])
def test_rank_coding_internals_are_gone(name):
    from subtext_codec import codec

    assert not hasattr(codec, name)


def test_codec_key_has_no_rank_coding_fields():
    import dataclasses

    fields = {f.name for f in dataclasses.fields(subtext_codec.CodecKey)}
    assert "top_p" not in fields
    assert "base" not in fields
    assert "payload_length" not in fields
    assert fields == {
        "top_k",
        "temperature",
        "prompt_prefix",
        "model_name_or_path",
        "device",
        "torch_dtype",
        "version",
    }
