# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Subtext-codec is a steganographic codec that hides arbitrary binary data inside LLM-generated text by steering token selection via logit rank. Uses adaptive mixed-radix encoding where the base varies dynamically per token based on the model's confidence (via `top_p` and optional `top_k`). The process is fully reversible and deterministic.

## Commands

```bash
# Setup
uv venv .env --python 3.13
uv pip install -r requirements.txt

# Run tests (fast, offline-friendly)
python -m pytest

# Encode binary data into text
subtext-codec encode \
  --model-name-or-path gpt2 \
  --prompt-prefix "Once upon a time, " \
  --input-bytes secret.txt \
  --output-text message.txt \
  --key key.json \
  --top-k 16 \
  --top-p 0.9

# Decode text back to binary
subtext-codec decode \
  --input-text message.txt \
  --key key.json \
  --output-bytes decoded.bin

# Can also run via module
python -m subtext_codec encode ...
python -m subtext_codec decode ...
```

## Architecture

**Core module** (`subtext_codec/codec.py`): Contains encoding/decoding logic, base conversion utilities, and dataclasses:
- `CodecConfig`: Runtime configuration for encoding (model, device, top_p/top_k, prompt prefix)
- `CodecKey`: Serializable metadata for decoding (auto-detects v1 fixed-base vs v2 dynamic-base)

**CLI** (`subtext_codec/cli.py`): Argparse-based interface with `encode` and `decode` subcommands. Supports stdin/stdout via `"-"` argument.

**Entry point** (`subtext_codec/__main__.py`): Sets `CUBLAS_WORKSPACE_CONFIG` for CUDA determinism, then calls CLI.

## Key Implementation Details

- **Codec versions**: v1 (legacy) uses fixed base stored in key JSON; v2 (current) uses dynamic mixed-radix with `top_p`/`top_k`. Decoder auto-detects from key.
- **Determinism**: `set_deterministic(seed)` ensures reproducibility via torch seeding and cuDNN config.
- **Termination**: Encoded text ends with a terminator token at rank equal to the active base; decoder stops there.
- **Context handling**: Decoder extracts prompt prefix from encoded text and only processes the generated portion.
- **Leading zeros**: Decoded bytes strip leading `\x00` (recoverable with external length metadata).

## Development Guidelines

- Use the virtual env from README; prefer `uv` for dependency management.
- Keep changes ASCII; preserve existing metadata header shapes.
- Add new dependencies to `pyproject.toml`, regenerate `uv.lock`, pin in `requirements.txt`.
- Stay in inference mode for models (`model.eval()`, `torch.no_grad()`).
- Raise clear `ValueError`/`RuntimeError` for user misconfiguration; no silent fallbacks.
- Mark slow/network tests with `@pytest.mark.slow` and skip gracefully when prerequisites missing.
- Encoding/decoding parameters must match exactly (model, prompt prefix, top_p/top_k).
