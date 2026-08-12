"""End-to-end round trips over the fake model, plus the CLI wiring."""

import json

import pytest
import torch

import subtext_codec
from subtext_codec import CodecConfig, cli, decode_text_to_data, encode_data_to_text
from subtext_codec.codec import (
    MAX_PAYLOAD_BYTES,
    _HEADER,
    _stable_candidates,
    _step_distribution,
)

from conftest import make_fake_components

PROMPT = "abc uvw"


def make_config(**overrides) -> CodecConfig:
    params = dict(
        model_name_or_path="fake-model",
        device="cpu",
        prompt_prefix=PROMPT,
        max_context_length=None,
        top_k=16,
        temperature=1.5,
        store_model_in_key=True,
    )
    params.update(overrides)
    return CodecConfig(**params)


def round_trip(payload: bytes, components, **overrides) -> bytes:
    tokenizer, model = components
    cfg = make_config(**overrides)
    text, key = encode_data_to_text(payload, cfg, model, tokenizer)
    return decode_text_to_data(
        text,
        key=key,
        prompt_prefix=cfg.prompt_prefix,
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )


# --------------------------------------------------------------------------
# round trips
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"\x00",
        b"a",
        b"hello world",
        b"\x00\x00\x00leading zeros",
        b"\xff\xfe\xfd\xfc",
        bytes(range(64)),
    ],
    ids=["empty", "zero", "single", "text", "leading-zeros", "high-bytes", "binary"],
)
def test_round_trip_payloads(payload, fake_components):
    assert round_trip(payload, fake_components) == payload


@pytest.mark.parametrize("top_k", [8, 16, 24, 32])
@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.5, 2.5])
def test_round_trip_across_parameters(top_k, temperature, fake_components):
    payload = b"parameters"
    decoded = round_trip(
        payload, fake_components, top_k=top_k, temperature=temperature
    )
    assert decoded == payload


@pytest.mark.parametrize("seed", range(6))
def test_round_trip_across_models(seed):
    """A different logit landscape must not change correctness."""
    payload = b"\x01\x02 many models \xfe"
    assert round_trip(payload, make_fake_components(seed=seed)) == payload


@pytest.mark.parametrize("prompt", ["abc", "uvwxyz", " a b c ", "abc uvw.!"])
def test_round_trip_across_prompts(prompt, fake_components):
    payload = b"prompts"
    assert round_trip(payload, fake_components, prompt_prefix=prompt) == payload


def test_higher_temperature_needs_fewer_tokens(fake_components):
    """Temperature is the capacity dial: flatter distribution, denser payload."""
    tokenizer, model = fake_components
    payload = b"capacity comparison payload"

    cold, _ = encode_data_to_text(
        payload, make_config(temperature=0.5), model, tokenizer
    )
    hot, _ = encode_data_to_text(
        payload, make_config(temperature=3.0), model, tokenizer
    )
    assert len(tokenizer(hot)["input_ids"]) < len(tokenizer(cold)["input_ids"])


def test_encoded_text_starts_with_prompt(fake_components):
    tokenizer, model = fake_components
    text, _ = encode_data_to_text(b"prefix check", make_config(), model, tokenizer)
    assert text.startswith(PROMPT)


def test_encoded_text_tokenizes_back_to_itself(fake_components):
    """The property the whole scheme rests on: text -> ids is stable."""
    tokenizer, model = fake_components
    text, _ = encode_data_to_text(b"stability", make_config(), model, tokenizer)
    ids = tokenizer(text)["input_ids"]
    assert tokenizer(tokenizer.decode(ids))["input_ids"] == ids


def test_decoder_ignores_surrounding_noise(fake_components):
    tokenizer, model = fake_components
    cfg = make_config()
    payload = b"noisy"
    text, key = encode_data_to_text(payload, cfg, model, tokenizer)

    decoded = decode_text_to_data(
        "zzz. " + text + " www!",
        key=key,
        prompt_prefix=cfg.prompt_prefix,
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )
    assert decoded == payload


def test_key_records_encode_parameters(fake_components):
    tokenizer, model = fake_components
    cfg = make_config(top_k=12, temperature=1.25, torch_dtype="fp32")
    _, key = encode_data_to_text(b"key fields", cfg, model, tokenizer)

    assert key.version == subtext_codec.CODEC_VERSION
    assert key.top_k == 12
    assert key.temperature == pytest.approx(1.25)
    assert key.prompt_prefix == PROMPT
    assert key.model_name_or_path == "fake-model"
    assert key.torch_dtype == "fp32"


def test_model_name_withheld_when_not_stored(fake_components):
    tokenizer, model = fake_components
    _, key = encode_data_to_text(
        b"private", make_config(store_model_in_key=False), model, tokenizer
    )
    assert key.model_name_or_path is None


# --------------------------------------------------------------------------
# compression (v2)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [b"", b"a", b"hello world", b"\x00\x00\x00zeros", b"repeat " * 20],
    ids=["empty", "single", "text", "leading-zeros", "compressible"],
)
def test_round_trip_compressed(payload, fake_components):
    assert round_trip(payload, fake_components, compress=True) == payload


def test_compression_records_v2_and_the_codec(fake_components):
    tokenizer, model = fake_components
    _, key = encode_data_to_text(
        b"compress me", make_config(compress=True), model, tokenizer
    )
    assert key.version == subtext_codec.CODEC_VERSION_V2
    assert key.compression == "zlib"


def test_plain_message_stays_v1(fake_components):
    tokenizer, model = fake_components
    _, key = encode_data_to_text(b"plain", make_config(), model, tokenizer)
    assert key.version == subtext_codec.CODEC_VERSION
    assert key.compression is None


def test_wrong_compression_flag_fails_to_decode(fake_components):
    """A key that misdescribes compression must not return plausible garbage."""
    tokenizer, model = fake_components
    text, key = encode_data_to_text(
        b"a real payload here", make_config(compress=True), model, tokenizer
    )
    key.compression = None  # claim it was never compressed
    with pytest.raises(ValueError):
        decode_text_to_data(
            text, key=key, prompt_prefix=PROMPT, model=model,
            tokenizer=tokenizer, device="cpu",
        )


# --------------------------------------------------------------------------
# multi-segment chunking (v2)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_bytes", [1, 4, 8, 16])
@pytest.mark.parametrize(
    "payload",
    [b"", b"tiny", b"\x00\x00\x00leading zeros here", bytes(range(48))],
    ids=["empty", "tiny", "leading-zeros", "binary"],
)
def test_round_trip_chunked(chunk_bytes, payload, fake_components):
    assert round_trip(payload, fake_components, chunk_bytes=chunk_bytes) == payload


def test_chunking_splits_into_the_expected_segments(fake_components):
    tokenizer, model = fake_components
    payload = bytes(range(40))
    stats = {}
    text, key = encode_data_to_text(
        payload,
        make_config(chunk_bytes=8),
        model,
        tokenizer,
        report_stats=lambda s: stats.update(vars(s)),
    )
    # The whole payload is framed with an 8-byte manifest header before chunking.
    expected = -(-(len(payload) + _HEADER.size) // 8)  # ceil((40 + 8) / 8) == 6
    assert key.version == subtext_codec.CODEC_VERSION_V2
    assert stats["segments"] == expected
    assert text.count(PROMPT) >= expected  # the prompt re-anchors each segment


def test_small_payload_with_chunk_bytes_stays_single_segment(fake_components):
    """chunk_bytes only splits when the payload actually exceeds it."""
    tokenizer, model = fake_components
    _, key = encode_data_to_text(
        b"short", make_config(chunk_bytes=1024), model, tokenizer
    )
    assert key.version == subtext_codec.CODEC_VERSION  # one chunk, uncompressed


def test_round_trip_compressed_and_chunked(fake_components):
    payload = b"the quick brown fox jumps over the lazy dog " * 8
    decoded = round_trip(payload, fake_components, compress=True, chunk_bytes=16)
    assert decoded == payload


def test_chunked_decoder_ignores_surrounding_noise(fake_components):
    tokenizer, model = fake_components
    payload = bytes(range(30))
    cfg = make_config(chunk_bytes=8)
    text, key = encode_data_to_text(payload, cfg, model, tokenizer)

    decoded = decode_text_to_data(
        "noise. " + text + " trailing!",
        key=key, prompt_prefix=PROMPT, model=model, tokenizer=tokenizer, device="cpu",
    )
    assert decoded == payload


def test_tampering_one_chunk_is_caught(fake_components):
    tokenizer, model = fake_components
    payload = bytes(range(40))
    cfg = make_config(chunk_bytes=8)
    text, key = encode_data_to_text(payload, cfg, model, tokenizer)

    # Insert text into the first segment's body, shifting its tokens.
    idx = text.find(PROMPT) + len(PROMPT)
    tampered = text[:idx] + "zzz " + text[idx:]
    with pytest.raises(ValueError):
        decode_text_to_data(
            tampered, key=key, prompt_prefix=PROMPT, model=model,
            tokenizer=tokenizer, device="cpu",
        )


# --------------------------------------------------------------------------
# encode statistics
# --------------------------------------------------------------------------


def test_encode_stats_are_reported(fake_components):
    tokenizer, model = fake_components
    seen = []
    encode_data_to_text(
        b"measure me please",
        make_config(),
        model,
        tokenizer,
        report_stats=seen.append,
    )
    assert len(seen) == 1
    stats = seen[0]
    assert isinstance(stats, subtext_codec.EncodeStats)
    assert stats.payload_bytes == len(b"measure me please")
    assert stats.stored_bytes == stats.payload_bytes  # not compressed
    assert stats.segments == 1
    assert not stats.compressed
    assert stats.tokens > 0
    assert stats.bits_per_token > 0
    assert stats.mean_surprisal_bits > 0


def test_compression_stats_show_the_stored_size(fake_components):
    tokenizer, model = fake_components
    seen = []
    encode_data_to_text(
        b"repeat " * 40, make_config(compress=True), model, tokenizer,
        report_stats=seen.append,
    )
    stats = seen[0]
    assert stats.compressed
    assert stats.stored_bytes < stats.payload_bytes  # repetitive data compresses


# --------------------------------------------------------------------------
# the tokenizer-stability filter
# --------------------------------------------------------------------------


def test_stable_candidates_rejects_merging_tokens(fake_tokenizer):
    """"b" after "a" would read back as the single piece "ab"."""
    a = fake_tokenizer.vocab.index("a")
    b = fake_tokenizer.vocab.index("b")
    x = fake_tokenizer.vocab.index("x")

    stable = _stable_candidates(fake_tokenizer, [a], [b, x])
    assert b not in stable
    assert x in stable


def test_stable_candidates_rejects_special_tokens(fake_tokenizer):
    x = fake_tokenizer.vocab.index("x")
    stable = _stable_candidates(
        fake_tokenizer,
        [x],
        [x, fake_tokenizer.eos_token_id, fake_tokenizer.pad_token_id],
    )
    assert stable == [x]


def test_alphabet_never_contains_an_unstable_token(fake_components):
    """Every token the encoder can pick must survive a write/read cycle."""
    tokenizer, model = fake_components
    ids = tokenizer(PROMPT)["input_ids"]
    logits = model(input_ids=torch.tensor([ids])).logits[0, -1, :]
    alphabet, cum = _step_distribution(logits, ids, tokenizer, 16, 1.5)

    for token in alphabet:
        seq = ids + [token]
        assert tokenizer(tokenizer.decode(seq))["input_ids"] == seq


def test_frequency_table_matches_the_alphabet(fake_components):
    tokenizer, model = fake_components
    ids = tokenizer(PROMPT)["input_ids"]
    logits = model(input_ids=torch.tensor([ids])).logits[0, -1, :]
    alphabet, cum = _step_distribution(logits, ids, tokenizer, 16, 1.5)

    assert len(cum) == len(alphabet) + 1
    assert cum[0] == 0
    assert cum[-1] == subtext_codec.arithmetic.FREQ_TOTAL
    assert all(b > a for a, b in zip(cum, cum[1:])), "every symbol must be reachable"


# --------------------------------------------------------------------------
# failure modes
# --------------------------------------------------------------------------


def test_wrong_prompt_is_rejected(fake_components):
    tokenizer, model = fake_components
    text, key = encode_data_to_text(b"mismatch", make_config(), model, tokenizer)

    with pytest.raises(ValueError, match="does not match"):
        decode_text_to_data(
            text, key=key, prompt_prefix="uvw", model=model,
            tokenizer=tokenizer, device="cpu",
        )


def test_missing_prompt_in_text_is_rejected(fake_components):
    tokenizer, model = fake_components
    _, key = encode_data_to_text(b"mismatch", make_config(), model, tokenizer)

    with pytest.raises(ValueError, match="not found in encoded_text"):
        decode_text_to_data(
            "zzz", key=key, prompt_prefix=PROMPT, model=model,
            tokenizer=tokenizer, device="cpu",
        )


@pytest.mark.parametrize("version", ["v3", "v99", "v100"])
def test_foreign_key_versions_are_refused(version, fake_components):
    tokenizer, model = fake_components
    text, key = encode_data_to_text(b"versioned", make_config(), model, tokenizer)
    key.version = version

    with pytest.raises(ValueError, match="Unsupported codec key version"):
        decode_text_to_data(
            text, key=key, prompt_prefix=PROMPT, model=model,
            tokenizer=tokenizer, device="cpu",
        )


def test_wrong_model_is_caught_not_silently_wrong(fake_components):
    """Decoding against a different model must fail loudly, not return garbage."""
    tokenizer, model = fake_components
    text, key = encode_data_to_text(
        b"a real payload here", make_config(), model, tokenizer
    )

    _, other_model = make_fake_components(seed=99)
    with pytest.raises(ValueError):
        decode_text_to_data(
            text, key=key, prompt_prefix=PROMPT, model=other_model,
            tokenizer=tokenizer, device="cpu",
        )


def test_wrong_temperature_is_caught(fake_components):
    """The checksum is what stands between a wrong key and plausible garbage."""
    tokenizer, model = fake_components
    text, key = encode_data_to_text(
        b"a real payload here", make_config(temperature=1.5), model, tokenizer
    )
    key.temperature = 2.0

    with pytest.raises(ValueError):
        decode_text_to_data(
            text, key=key, prompt_prefix=PROMPT, model=model,
            tokenizer=tokenizer, device="cpu",
        )


def test_truncated_message_is_caught(fake_components):
    tokenizer, model = fake_components
    text, key = encode_data_to_text(
        b"truncate me please", make_config(), model, tokenizer
    )

    with pytest.raises(ValueError):
        decode_text_to_data(
            text[: len(PROMPT) + 3], key=key, prompt_prefix=PROMPT, model=model,
            tokenizer=tokenizer, device="cpu",
        )


def test_max_new_tokens_is_enforced(fake_components):
    tokenizer, model = fake_components
    with pytest.raises(ValueError, match="max_new_tokens"):
        encode_data_to_text(
            b"far too long for four tokens",
            make_config(max_new_tokens=4),
            model,
            tokenizer,
        )


def test_context_limit_is_enforced(fake_components):
    tokenizer, model = fake_components
    with pytest.raises(ValueError, match="context limit"):
        encode_data_to_text(
            b"far too long for twelve tokens",
            make_config(max_context_length=12),
            model,
            tokenizer,
        )


def test_top_k_too_small_is_rejected(fake_components):
    tokenizer, model = fake_components
    with pytest.raises(ValueError, match="top_k must be"):
        encode_data_to_text(b"x", make_config(top_k=1), model, tokenizer)


def test_oversized_payload_is_rejected_at_encode(fake_components):
    """A payload the decoder would call implausible must fail at encode time.

    The guard fires before any generation, so this does not run the model on a
    16MB payload; it just refuses to create a message that could not decode.
    """
    tokenizer, model = fake_components
    with pytest.raises(ValueError, match="maximum is"):
        encode_data_to_text(
            b"\x00" * (MAX_PAYLOAD_BYTES + 1), make_config(), model, tokenizer
        )


def test_empty_prompt_is_rejected(fake_components):
    tokenizer, model = fake_components
    with pytest.raises(ValueError, match="at least one token"):
        encode_data_to_text(b"x", make_config(prompt_prefix=""), model, tokenizer)


# --------------------------------------------------------------------------
# progress reporting
# --------------------------------------------------------------------------


def test_progress_is_reported_and_completes(fake_components):
    tokenizer, model = fake_components
    seen = []
    encode_data_to_text(
        b"progress please",
        make_config(),
        model,
        tokenizer,
        progress=lambda d, t: seen.append((d, t)),
    )
    assert seen
    assert seen[-1][0] == seen[-1][1]
    assert all(0 <= d <= t for d, t in seen)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


@pytest.fixture
def cli_with_fake_model(monkeypatch):
    components = make_fake_components(seed=3)

    def fake_loader(model_name_or_path, device, torch_dtype=None, seed=0):
        return components

    monkeypatch.setattr(cli, "load_model_and_tokenizer", fake_loader)
    return components


def run_cli_round_trip(tmp_path, payload: bytes, extra_encode=()):
    paths = {
        "input": tmp_path / "input.bin",
        "text": tmp_path / "encoded.txt",
        "output": tmp_path / "decoded.bin",
        "key": tmp_path / "key.json",
    }
    paths["input"].write_bytes(payload)

    cli.main(
        [
            "encode",
            "--model-name-or-path", "fake-cli-model",
            "--device", "cpu",
            "--prompt-prefix", PROMPT,
            "--input-bytes", str(paths["input"]),
            "--output-text", str(paths["text"]),
            "--key", str(paths["key"]),
            "--top-k", "16",
            "--temperature", "1.5",
            *extra_encode,
        ]
    )
    cli.main(
        [
            "decode",
            "--input-text", str(paths["text"]),
            "--output-bytes", str(paths["output"]),
            "--key", str(paths["key"]),
        ]
    )
    return paths


def test_cli_round_trip(tmp_path, cli_with_fake_model):
    payload = b"cli integration payload"
    paths = run_cli_round_trip(tmp_path, payload)
    assert paths["output"].read_bytes() == payload

    key = subtext_codec.load_codec_key(paths["key"])
    assert key.model_name_or_path == "fake-cli-model"
    assert key.prompt_prefix == PROMPT
    assert key.version == subtext_codec.CODEC_VERSION
    assert key.top_k == 16
    assert key.temperature == pytest.approx(1.5)


def test_cli_round_trip_with_leading_zeros(tmp_path, cli_with_fake_model):
    payload = b"\x00\x00\x00hello"
    paths = run_cli_round_trip(tmp_path, payload)
    assert paths["output"].read_bytes() == payload


def test_cli_verify_flag(tmp_path, cli_with_fake_model):
    payload = b"verified"
    paths = run_cli_round_trip(tmp_path, payload, extra_encode=("--verify",))
    assert paths["output"].read_bytes() == payload


def test_cli_decode_does_not_modify_the_key(tmp_path, cli_with_fake_model):
    paths = run_cli_round_trip(tmp_path, b"immutable key")
    before = paths["key"].read_text()

    cli.main(
        [
            "decode",
            "--input-text", str(paths["text"]),
            "--output-bytes", str(tmp_path / "again.bin"),
            "--key", str(paths["key"]),
            "--device", "cpu",
            "--prompt-prefix", PROMPT,
        ]
    )
    assert paths["key"].read_text() == before


def test_cli_reuses_stored_key_parameters(tmp_path, cli_with_fake_model):
    """A second encode needs only the key: no model, prompt or parameters."""
    paths = run_cli_round_trip(tmp_path, b"first")

    second_text = tmp_path / "second.txt"
    second_out = tmp_path / "second.bin"
    payload = tmp_path / "second_in.bin"
    payload.write_bytes(b"second payload")

    cli.main(
        [
            "encode",
            "--input-bytes", str(payload),
            "--output-text", str(second_text),
            "--key", str(paths["key"]),
        ]
    )
    cli.main(
        [
            "decode",
            "--input-text", str(second_text),
            "--output-bytes", str(second_out),
            "--key", str(paths["key"]),
        ]
    )
    assert second_out.read_bytes() == b"second payload"


def test_cli_compress_round_trip(tmp_path, cli_with_fake_model):
    payload = b"compress this over the cli " * 6
    paths = run_cli_round_trip(tmp_path, payload, extra_encode=("--compress",))
    assert paths["output"].read_bytes() == payload

    key = subtext_codec.load_codec_key(paths["key"])
    assert key.version == subtext_codec.CODEC_VERSION_V2
    assert key.compression == "zlib"


def test_cli_chunk_bytes_round_trip(tmp_path, cli_with_fake_model):
    payload = bytes(range(50))
    paths = run_cli_round_trip(tmp_path, payload, extra_encode=("--chunk-bytes", "8"))
    assert paths["output"].read_bytes() == payload

    key = subtext_codec.load_codec_key(paths["key"])
    assert key.version == subtext_codec.CODEC_VERSION_V2


def test_cli_reused_key_keeps_compressing(tmp_path, cli_with_fake_model):
    """A key that recorded compression compresses the next message too."""
    paths = run_cli_round_trip(tmp_path, b"first message here", extra_encode=("--compress",))

    second_in = tmp_path / "second_in.bin"
    second_in.write_bytes(b"second message body")
    second_text = tmp_path / "second.txt"
    second_out = tmp_path / "second.bin"

    cli.main([
        "encode",
        "--input-bytes", str(second_in),
        "--output-text", str(second_text),
        "--key", str(paths["key"]),
    ])
    assert subtext_codec.load_codec_key(paths["key"]).compression == "zlib"

    cli.main([
        "decode",
        "--input-text", str(second_text),
        "--output-bytes", str(second_out),
        "--key", str(paths["key"]),
    ])
    assert second_out.read_bytes() == b"second message body"


def test_cli_version_flag_prints_and_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "subtext-codec" in out
    assert subtext_codec.__version__ in out


def test_cli_no_store_model(tmp_path, cli_with_fake_model):
    input_bytes = tmp_path / "in.bin"
    input_bytes.write_bytes(b"anonymous")
    key_path = tmp_path / "key.json"

    cli.main(
        [
            "encode",
            "--model-name-or-path", "fake-cli-model",
            "--device", "cpu",
            "--prompt-prefix", PROMPT,
            "--input-bytes", str(input_bytes),
            "--output-text", str(tmp_path / "out.txt"),
            "--key", str(key_path),
            "--no-store-model",
        ]
    )
    assert json.loads(key_path.read_text())["model_name_or_path"] is None


def test_cli_decode_without_key_file_errors(tmp_path, cli_with_fake_model):
    with pytest.raises(SystemExit):
        cli.main(
            [
                "decode",
                "--input-text", str(tmp_path / "missing.txt"),
                "--output-bytes", str(tmp_path / "out.bin"),
                "--key", str(tmp_path / "absent.json"),
            ]
        )
