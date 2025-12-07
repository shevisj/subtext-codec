"""CLI shim for running the codec directly from the repository checkout."""

from subtext_codec.cli import main
from subtext_codec.codec import (
    CodecConfig,
    CodecKey,
    base_digits_to_bytes,
    bytes_to_base_digits,
    decode_text_to_data,
    encode_data_to_text,
    load_codec_key,
    load_model_and_tokenizer,
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
    "save_codec_key",
    "main",
    "set_deterministic",
]


if __name__ == "__main__":
    main()
