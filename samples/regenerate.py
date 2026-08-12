#!/usr/bin/env python3
"""Regenerate the sample fixture.

Run from the repository root:

    python samples/regenerate.py

Writes `message.txt`, `key.json` and `decoded.txt` next to `secret.txt`, using
`gpt2` on CPU so the fixture stays reproducible without a GPU or a gated
checkpoint. Encoding is deterministic, so re-running this reproduces the same
message byte for byte.
"""

import pathlib
import sys

import subtext_codec
from subtext_codec import (
    CodecConfig,
    decode_text_to_data,
    encode_data_to_text,
    save_codec_key,
)

SAMPLES = pathlib.Path(__file__).resolve().parent
MODEL = "gpt2"
PROMPT = "Today, I'm going to teach you about steganography. "
TOP_K = 64
TEMPERATURE = 1.5


def main() -> int:
    secret = (SAMPLES / "secret.txt").read_bytes()
    print(f"payload: {len(secret)} bytes")

    tokenizer, model = subtext_codec.load_model_and_tokenizer(MODEL, "cpu", "float32")

    cfg = CodecConfig(
        model_name_or_path=MODEL,
        device="cpu",
        prompt_prefix=PROMPT,
        top_k=TOP_K,
        temperature=TEMPERATURE,
        torch_dtype="float32",
        store_model_in_key=True,
    )

    def progress(done: int, total: int) -> None:
        print(f"\r  encoding {100 * done // max(total, 1)}%", end="", flush=True)

    text, key = encode_data_to_text(secret, cfg, model, tokenizer, progress=progress)
    print()

    prompt_tokens = len(tokenizer(PROMPT)["input_ids"])
    body_tokens = len(tokenizer(text)["input_ids"]) - prompt_tokens
    print(
        f"encoded: {len(text)} chars / {body_tokens} tokens "
        f"({8 * len(secret) / max(body_tokens, 1):.2f} bits per token)"
    )

    recovered = decode_text_to_data(
        text,
        key=key,
        prompt_prefix=PROMPT,
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )
    if recovered != secret:
        print("round trip FAILED; refusing to write the fixture", file=sys.stderr)
        return 1
    print("round trip: exact match")

    (SAMPLES / "message.txt").write_text(text, encoding="utf-8")
    (SAMPLES / "decoded.txt").write_bytes(recovered)
    save_codec_key(key, SAMPLES / "key.json")
    print("wrote message.txt, key.json, decoded.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
