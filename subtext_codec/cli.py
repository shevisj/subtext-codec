import argparse
import os
import sys
from typing import List, Optional

from .codec import (
    CodecConfig,
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM logit-rank codec PoC")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model-name-or-path")
    common.add_argument("--device")
    common.add_argument("--prompt-prefix")
    common.add_argument("--max-context-length", type=int, default=None)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument(
        "--torch-dtype",
        help="torch dtype for model weights (auto, float16/fp16/half, bfloat16/bf16, float32/fp32)",
    )

    enc = subparsers.add_parser("encode", parents=[common])
    enc.add_argument("--top-k", type=int, default=None)
    enc.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Cumulative probability threshold for variable bases (0 < top-p <= 1)",
    )
    enc.add_argument("--input-bytes", required=True)
    enc.add_argument("--output-text", required=True)
    enc.add_argument(
        "--key",
        required=True,
        help="Path to the codec key (loads existing values and writes updates)",
    )
    enc.add_argument(
        "--include-model-in-key",
        action="store_true",
        help="Store the model name in the generated key file",
    )

    dec = subparsers.add_parser("decode", parents=[common])
    dec.add_argument("--input-text", required=True)
    dec.add_argument("--output-bytes", required=True)
    dec.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Override top-p from the key (0 < top-p <= 1)",
    )
    dec.add_argument(
        "--key",
        required=True,
        help="Path to the codec key (loads existing values and writes updates)",
    )

    return parser


def run_encode(args) -> None:
    key_from_input = load_codec_key(args.key) if os.path.exists(args.key) else None

    model_name = args.model_name_or_path or (
        key_from_input.model_name_or_path if key_from_input else None
    )
    if model_name is None:
        raise ValueError(
            "model-name-or-path is required unless provided via --key"
        )

    prompt_prefix = args.prompt_prefix or (
        key_from_input.prompt_prefix if key_from_input else None
    )
    if prompt_prefix is None:
        raise ValueError("prompt-prefix is required unless provided via --key")

    top_k = (
        args.top_k
        if args.top_k is not None
        else key_from_input.top_k if key_from_input else None
    )
    top_p = (
        args.top_p
        if args.top_p is not None
        else key_from_input.top_p if key_from_input else None
    )
    if top_p is None:
        top_p = 0.9
    torch_dtype = (
        args.torch_dtype
        if args.torch_dtype is not None
        else key_from_input.torch_dtype if key_from_input else None
    )

    device = (
        args.device
        if args.device is not None
        else key_from_input.device if key_from_input else "cpu"
    )

    set_deterministic(args.seed)
    tokenizer, model = load_model_and_tokenizer(
        model_name, device, torch_dtype=torch_dtype
    )
    cfg = CodecConfig(
        model_name_or_path=model_name,
        device=device,
        prompt_prefix=prompt_prefix,
        max_context_length=args.max_context_length,
        top_k=top_k,
        top_p=top_p,
        torch_dtype=torch_dtype,
        store_model_in_key=(
            args.include_model_in_key
            or (key_from_input is not None and key_from_input.model_name_or_path is not None)
            or args.model_name_or_path is not None
        ),
    )
    payload = _read_bytes(args.input_bytes)
    text, key = encode_data_to_text(payload, cfg, model, tokenizer)
    save_codec_key(key, args.key)
    _write_text(args.output_text, text)


def run_decode(args) -> None:
    if not os.path.exists(args.key):
        raise ValueError("--key file not found; create one during encode first")
    key = load_codec_key(args.key)

    model_name = args.model_name_or_path or key.model_name_or_path
    if model_name is None:
        raise ValueError("model-name-or-path is required unless present in --key")

    prompt_prefix = args.prompt_prefix or key.prompt_prefix
    if prompt_prefix is None:
        raise ValueError("prompt-prefix is required unless present in --key")
    torch_dtype = args.torch_dtype or key.torch_dtype
    device = args.device or key.device or "cpu"
    top_p = args.top_p if args.top_p is not None else key.top_p
    if key.version != "v1" and top_p is None:
        raise ValueError("top-p is required unless present in --key for variable-base decoding")

    set_deterministic(args.seed)
    tokenizer, model = load_model_and_tokenizer(
        model_name, device, torch_dtype=torch_dtype
    )
    encoded_text = _read_text(args.input_text)
    key.top_p = top_p
    data = decode_text_to_data(
        encoded_text,
        key=key,
        prompt_prefix=prompt_prefix,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_context_length=args.max_context_length,
    )
    key.model_name_or_path = model_name
    key.prompt_prefix = prompt_prefix
    key.device = device
    key.torch_dtype = torch_dtype
    save_codec_key(key, args.key)
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
    except ValueError as exc:
        parser.error(str(exc))


__all__ = ["build_arg_parser", "run_encode", "run_decode", "main"]
