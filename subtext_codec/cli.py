import argparse
import os
import sys
from typing import List, Optional

from . import __version__
from .codec import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    CodecConfig,
    CodecKey,
    EncodeStats,
    decode_text_to_data,
    encode_data_to_text,
    load_codec_key,
    load_model_and_tokenizer,
    save_codec_key,
    set_deterministic,
)


def _read_bytes(path: str) -> bytes:
    if path == "-":
        return sys.stdin.buffer.read()
    with open(path, "rb") as f:
        return f.read()


def _read_text(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_bytes(path: str, data: bytes) -> None:
    if path == "-":
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    else:
        with open(path, "wb") as f:
            f.write(data)


def _write_text(path: str, text: str) -> None:
    if path == "-":
        sys.stdout.write(text)
        sys.stdout.flush()
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


class _Progress:
    """Single-line progress on stderr, so it never pollutes piped output."""

    def __init__(self, label: str, enabled: bool):
        self.label = label
        self.enabled = enabled and sys.stderr.isatty()
        self._last = -1

    def __call__(self, done: int, total: Optional[int]) -> None:
        if not self.enabled:
            return
        pct = int(100 * done / total) if total else 0
        if pct == self._last:
            return
        self._last = pct
        sys.stderr.write(f"\r{self.label}: {pct:3d}%")
        sys.stderr.flush()

    def done(self) -> None:
        if self.enabled and self._last >= 0:
            sys.stderr.write("\n")
            sys.stderr.flush()


def _stats_reporter(enabled: bool):
    """Return a report_stats callback that prints one summary line to stderr.

    Uses stderr so it never contaminates piped output, and returns None when
    disabled so the encoder does no accounting work it will not use.
    """
    if not (enabled and sys.stderr.isatty()):
        return None

    def report(stats: EncodeStats) -> None:
        line = (
            f"encoded {stats.payload_bytes} B into {stats.tokens} tokens "
            f"({stats.bits_per_token:.2f} bits/token, "
            f"surprisal {stats.mean_surprisal_bits:.2f})"
        )
        if stats.resets:
            line += f"; rolled the context {stats.resets}x"
        if stats.compressed:
            line += f"; compressed {stats.payload_bytes} -> {stats.stored_bytes} B"
        sys.stderr.write(line + "\n")
        sys.stderr.flush()

    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subtext-codec",
        description=(
            "Hide bytes inside LLM-generated text by arithmetic coding against "
            "the model's own next-token distribution."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="print the installed subtext-codec version and exit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--model-name-or-path",
        help="HuggingFace model id or local path to a causal LM",
    )
    common.add_argument("--device", help="torch device, e.g. cpu or cuda (default: cpu)")
    common.add_argument(
        "--prompt-prefix", help="prefix text; must be identical for encode and decode"
    )
    common.add_argument(
        "--max-context-length",
        type=int,
        default=None,
        help="cap on total sequence length (default: the model's own limit)",
    )
    common.add_argument("--seed", type=int, default=0, help="deterministic seed")
    common.add_argument(
        "--torch-dtype",
        help="model weight dtype (auto, float16/fp16/half, bfloat16/bf16, float32/fp32)",
    )
    common.add_argument(
        "--key",
        required=True,
        help="path to the codec key file",
    )
    common.add_argument(
        "--quiet", action="store_true", help="suppress progress output on stderr"
    )

    enc = subparsers.add_parser(
        "encode", parents=[common], help="hide bytes inside generated text"
    )
    enc.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=(
            "candidates considered per step; bounds the stability filter's work "
            f"rather than capacity (default: {DEFAULT_TOP_K})"
        ),
    )
    enc.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "sampling temperature, and the capacity dial: higher packs more "
            "payload per token but makes the cover text more erratic "
            f"(default: {DEFAULT_TEMPERATURE})"
        ),
    )
    enc.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="fail rather than generate more than this many tokens per segment",
    )
    enc.add_argument(
        "--compress",
        action="store_true",
        help=(
            "zlib-compress the payload before encoding (recorded in the key). "
            "Shortens the cover text and makes the payload bits more uniform. "
            "Uses the v2 wire format."
        ),
    )
    enc.add_argument("--input-bytes", required=True, help="payload file, or - for stdin")
    enc.add_argument(
        "--output-text", required=True, help="destination for the text, or - for stdout"
    )
    enc.add_argument(
        "--no-store-model",
        action="store_true",
        help="keep the model id out of the key file",
    )
    enc.add_argument(
        "--verify",
        action="store_true",
        help="decode the result before writing it, to confirm the round trip",
    )

    dec = subparsers.add_parser(
        "decode", parents=[common], help="recover bytes from encoded text"
    )
    dec.add_argument("--input-text", required=True, help="encoded text, or - for stdin")
    dec.add_argument(
        "--output-bytes", required=True, help="destination for the payload, or - for stdout"
    )
    dec.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="override the key's temperature (it must match the encode run)",
    )

    return parser


def _first(*values):
    """First value that was actually supplied."""
    for value in values:
        if value is not None:
            return value
    return None


def run_encode(args) -> None:
    existing = load_codec_key(args.key) if os.path.exists(args.key) else None

    model_name = _first(
        args.model_name_or_path, existing.model_name_or_path if existing else None
    )
    if model_name is None:
        raise ValueError("--model-name-or-path is required unless stored in --key")

    prompt_prefix = _first(
        args.prompt_prefix, existing.prompt_prefix if existing else None
    )
    if prompt_prefix is None:
        raise ValueError("--prompt-prefix is required unless stored in --key")

    top_k = _first(args.top_k, existing.top_k if existing else None, DEFAULT_TOP_K)
    temperature = _first(
        args.temperature,
        existing.temperature if existing else None,
        DEFAULT_TEMPERATURE,
    )
    torch_dtype = _first(args.torch_dtype, existing.torch_dtype if existing else None)
    device = _first(args.device, existing.device if existing else None, "cpu")
    # A key that already records compression keeps compressing on re-encode, so a
    # second message written against it matches the first without restating it.
    compress = args.compress or bool(existing and existing.compression == "zlib")

    set_deterministic(args.seed)
    tokenizer, model = load_model_and_tokenizer(
        model_name, device, torch_dtype=torch_dtype, seed=args.seed
    )

    cfg = CodecConfig(
        model_name_or_path=model_name,
        device=device,
        prompt_prefix=prompt_prefix,
        max_context_length=args.max_context_length,
        top_k=top_k,
        temperature=temperature,
        torch_dtype=torch_dtype,
        store_model_in_key=not args.no_store_model,
        max_new_tokens=args.max_new_tokens,
        compress=compress,
    )

    payload = _read_bytes(args.input_bytes)
    progress = _Progress("encoding", not args.quiet)
    reporter = _stats_reporter(not args.quiet)
    try:
        text, key = encode_data_to_text(
            payload, cfg, model, tokenizer, progress=progress, report_stats=reporter
        )
    finally:
        progress.done()

    if args.verify:
        check = _Progress("verifying", not args.quiet)
        try:
            recovered = decode_text_to_data(
                text,
                key=key,
                prompt_prefix=prompt_prefix,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_context_length=args.max_context_length,
                progress=check,
            )
        finally:
            check.done()
        if recovered != payload:
            raise ValueError("verification failed: the message did not round-trip")

    save_codec_key(key, args.key)
    _write_text(args.output_text, text)


def run_decode(args) -> None:
    if not os.path.exists(args.key):
        raise ValueError(f"key file not found: {args.key}")
    key = load_codec_key(args.key)

    model_name = _first(args.model_name_or_path, key.model_name_or_path)
    if model_name is None:
        raise ValueError("--model-name-or-path is required unless present in --key")

    prompt_prefix = _first(args.prompt_prefix, key.prompt_prefix)
    if prompt_prefix is None:
        raise ValueError("--prompt-prefix is required unless present in --key")

    torch_dtype = _first(args.torch_dtype, key.torch_dtype)
    device = _first(args.device, key.device, "cpu")
    temperature = _first(args.temperature, key.temperature)
    if temperature is None:
        raise ValueError("--temperature is required unless present in --key")

    set_deterministic(args.seed)
    tokenizer, model = load_model_and_tokenizer(
        model_name, device, torch_dtype=torch_dtype, seed=args.seed
    )

    # Decoding never rewrites the key file; a read operation should not be able
    # to damage the one artifact the message cannot be recovered without.
    run_key = CodecKey(**{**vars(key), "temperature": temperature})

    encoded_text = _read_text(args.input_text)
    progress = _Progress("decoding", not args.quiet)
    try:
        data = decode_text_to_data(
            encoded_text,
            key=run_key,
            prompt_prefix=prompt_prefix,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_context_length=args.max_context_length,
            progress=progress,
        )
    finally:
        progress.done()

    _write_bytes(args.output_bytes, data)


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "encode":
            run_encode(args)
        elif args.command == "decode":
            run_decode(args)
        else:
            parser.error("Unknown command")
    except (ValueError, FileNotFoundError, OSError) as exc:
        parser.error(str(exc))


__all__ = ["build_arg_parser", "run_encode", "run_decode", "main"]
