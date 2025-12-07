"""Steganographic data encoding in LLM-generated text."""

from .codec import (
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
    "set_deterministic",
]

__version__ = "0.1.0"
