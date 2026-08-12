"""The CLI driving a real model in a separate process.

The fake-model CLI tests cover argument wiring; this covers the part they
cannot -- that a message written by one process is readable by another, with
the model loaded fresh, which is the only way the tool is ever actually used.
"""

import json
import subprocess
import sys

import pytest

MODEL = "sshleifer/tiny-gpt2"
PROMPT = "Once upon a time, "


def run_cli(*args, expect_success: bool = True):
    result = subprocess.run(
        [sys.executable, "-m", "subtext_codec", *args],
        capture_output=True,
        text=True,
    )
    if expect_success and result.returncode != 0:
        if "Can't load" in result.stderr or "connect" in result.stderr.lower():
            pytest.skip(f"{MODEL} unavailable")
        pytest.fail(f"exit {result.returncode}\nstderr:\n{result.stderr[-2000:]}")
    return result


@pytest.mark.slow
@pytest.mark.parametrize(
    "payload",
    [b"attack at dawn", b"\x00\x00leading zeros", b"\xff\xfe\x00\x01"],
    ids=["text", "leading-zeros", "binary"],
)
def test_cli_round_trip_across_processes(payload, tmp_path):
    payload_file = tmp_path / "in.bin"
    payload_file.write_bytes(payload)
    message = tmp_path / "message.txt"
    key = tmp_path / "key.json"
    out = tmp_path / "out.bin"

    run_cli(
        "encode",
        "--model-name-or-path", MODEL,
        "--torch-dtype", "float32",
        "--prompt-prefix", PROMPT,
        "--input-bytes", str(payload_file),
        "--output-text", str(message),
        "--key", str(key),
        "--top-k", "32",
        "--temperature", "1.5",
        "--verify",
        "--quiet",
    )

    assert message.read_text().startswith(PROMPT)

    # The key must be enough on its own: no model, prompt or parameters here.
    run_cli(
        "decode",
        "--input-text", str(message),
        "--output-bytes", str(out),
        "--key", str(key),
        "--quiet",
    )
    assert out.read_bytes() == payload


@pytest.mark.slow
def test_cli_compressed_and_chunked_round_trip_across_processes(tmp_path):
    """The v2 features driven the way the tool is really used: two processes."""
    payload = b"the quick brown fox jumps over the lazy dog. " * 3
    payload_file = tmp_path / "in.bin"
    payload_file.write_bytes(payload)
    message = tmp_path / "message.txt"
    key = tmp_path / "key.json"
    out = tmp_path / "out.bin"

    run_cli(
        "encode",
        "--model-name-or-path", MODEL,
        "--torch-dtype", "float32",
        "--prompt-prefix", PROMPT,
        "--input-bytes", str(payload_file),
        "--output-text", str(message),
        "--key", str(key),
        "--top-k", "32",
        "--temperature", "1.5",
        "--compress",
        "--chunk-bytes", "24",
        "--quiet",
    )

    stored = json.loads(key.read_text())
    assert stored["version"] == "v2"
    assert stored["compression"] == "zlib"

    run_cli(
        "decode",
        "--input-text", str(message),
        "--output-bytes", str(out),
        "--key", str(key),
        "--quiet",
    )
    assert out.read_bytes() == payload


def test_cli_version_flag():
    result = run_cli("--version")
    assert "subtext-codec" in result.stdout
    assert result.stdout.strip().split()[-1]  # a version string is present


@pytest.mark.slow
def test_cli_key_is_self_sufficient_and_stable(tmp_path):
    payload_file = tmp_path / "in.bin"
    payload_file.write_bytes(b"key contents")
    key = tmp_path / "key.json"

    run_cli(
        "encode",
        "--model-name-or-path", MODEL,
        "--torch-dtype", "float32",
        "--prompt-prefix", PROMPT,
        "--input-bytes", str(payload_file),
        "--output-text", str(tmp_path / "m.txt"),
        "--key", str(key),
        "--top-k", "32",
        "--temperature", "1.5",
        "--quiet",
    )

    stored = json.loads(key.read_text())
    assert stored == {
        "version": "v1",
        "top_k": 32,
        "temperature": 1.5,
        "prompt_prefix": PROMPT,
        "model_name_or_path": MODEL,
        "device": "cpu",
        "torch_dtype": "float32",
    }

    before = key.read_text()
    run_cli(
        "decode",
        "--input-text", str(tmp_path / "m.txt"),
        "--output-bytes", str(tmp_path / "o.bin"),
        "--key", str(key),
        "--quiet",
    )
    assert key.read_text() == before, "decode must not rewrite the key file"


def run_decode_with_key(tmp_path, key_data):
    key = tmp_path / "old.json"
    key.write_text(json.dumps(key_data))
    (tmp_path / "m.txt").write_text(PROMPT + "whatever")
    return run_cli(
        "decode",
        "--input-text", str(tmp_path / "m.txt"),
        "--output-bytes", str(tmp_path / "o.bin"),
        "--key", str(key),
        expect_success=False,
    )


def test_cli_names_a_pre_1_0_rank_coding_key(tmp_path):
    """Pre-1.0 also numbered a format "v1"; the collision must be explained.

    That key had `base`/`top_p` and no temperature, so it is distinguishable
    from the arithmetic-coding format that now owns the name.
    """
    result = run_decode_with_key(
        tmp_path,
        {"version": "v1", "top_k": 16, "top_p": 0.9, "prompt_prefix": PROMPT},
    )
    assert result.returncode != 0
    assert "predates subtext-codec 1.0" in result.stderr
    assert "0.2.0" in result.stderr


@pytest.mark.parametrize("version", ["v3", "v99"])
def test_cli_rejects_other_historical_versions(version, tmp_path):
    result = run_decode_with_key(
        tmp_path,
        {"version": version, "top_k": 16, "top_p": 0.9, "prompt_prefix": PROMPT},
    )
    assert result.returncode != 0
    assert "Unsupported codec key version" in result.stderr
