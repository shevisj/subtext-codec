"""CLI shim for running the codec directly from the repository checkout."""

from subtext.cli import main
from subtext.codec import (
    HEADER_PATTERN,
    CodecConfig,
    base_digits_to_bytes,
    bytes_to_base_digits,
    decode_text_to_data,
    encode_data_to_text,
    load_model_and_tokenizer,
    set_deterministic,
)

__all__ = [
    "HEADER_PATTERN",
    "CodecConfig",
    "base_digits_to_bytes",
    "bytes_to_base_digits",
    "decode_text_to_data",
    "encode_data_to_text",
    "load_model_and_tokenizer",
    "main",
    "set_deterministic",
]


if __name__ == "__main__":
    main()
