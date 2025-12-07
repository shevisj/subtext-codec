import argparse
import sys
from typing import List, Optional

from .codec import (
    CodecConfig,
    decode_text_to_data,
    encode_data_to_text,
    load_model_and_tokenizer,
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM logit-rank codec PoC")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model-name-or-path", required=True)
    common.add_argument("--device", default="cpu")
    common.add_argument("--prompt-prefix", required=True)
    common.add_argument("--top-k", type=int, default=None)
    common.add_argument("--max-context-length", type=int, default=None)
    common.add_argument("--seed", type=int, default=0)

    enc = subparsers.add_parser("encode", parents=[common])
    enc.add_argument("--base", type=int, required=True)
    enc.add_argument("--input-bytes", required=True)
    enc.add_argument("--output-text", required=True)
    enc.add_argument("--max-new-tokens", type=int, default=512)

    dec = subparsers.add_parser("decode", parents=[common])
    dec.add_argument("--input-text", required=True)
    dec.add_argument("--output-bytes", required=True)

    return parser


def run_encode(args) -> None:
    set_deterministic(args.seed)
    tokenizer, model = load_model_and_tokenizer(args.model_name_or_path, args.device)
    cfg = CodecConfig(
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        base=args.base,
        prompt_prefix=args.prompt_prefix,
        max_new_tokens=args.max_new_tokens,
        max_context_length=args.max_context_length,
        top_k=args.top_k,
    )
    payload = _read_bytes(args.input_bytes)
    text = encode_data_to_text(payload, cfg, model, tokenizer)
    _write_text(args.output_text, text)


def run_decode(args) -> None:
    set_deterministic(args.seed)
    tokenizer, model = load_model_and_tokenizer(args.model_name_or_path, args.device)
    encoded_text = _read_text(args.input_text)
    data = decode_text_to_data(
        encoded_text,
        prompt_prefix=args.prompt_prefix,
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        top_k_override=args.top_k,
        max_context_length=args.max_context_length,
    )
    _write_bytes(args.output_bytes, data)


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "encode":
        run_encode(args)
    elif args.command == "decode":
        run_decode(args)
    else:
        parser.error("Unknown command")


__all__ = ["build_arg_parser", "run_encode", "run_decode", "main"]
