"""Steganographic data encoding in LLM-generated text."""

from .codec import (
    CodecConfig,
    CodecKey,
    base_digits_to_bytes,
    bytes_to_base_digits,
    decode_text_to_data,
    encode_data_to_text,
    load_codec_key,
    load_model_and_tokenizer,
    mixed_radix_digits_to_bytes,
    save_codec_key,
    set_deterministic,
)

__all__ = [
    "CodecConfig",
    "CodecKey",
    "base_digits_to_bytes",
    "bytes_to_base_digits",
    "decode_text_to_data",
    "encode_data_to_text",
    "load_codec_key",
    "load_model_and_tokenizer",
    "mixed_radix_digits_to_bytes",
    "save_codec_key",
    "set_deterministic",
]

__version__ = "0.1.0"
