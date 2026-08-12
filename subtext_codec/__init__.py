"""Steganographic data encoding in LLM-generated text."""

import os

# cuBLAS only honours this when it is set before torch initialises, and the
# codec needs bit-identical matmuls across the encode and decode passes. Set as
# a default so callers who have their own value keep it.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .codec import (  # noqa: E402  (must follow the environment default above)
    CODEC_VERSION,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    CodecConfig,
    CodecKey,
    decode_text_to_data,
    encode_data_to_text,
    load_codec_key,
    load_model_and_tokenizer,
    save_codec_key,
    set_deterministic,
)

try:  # pragma: no cover - depends on install method
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("subtext-codec")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "CODEC_VERSION",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_K",
    "CodecConfig",
    "CodecKey",
    "__version__",
    "decode_text_to_data",
    "encode_data_to_text",
    "load_codec_key",
    "load_model_and_tokenizer",
    "save_codec_key",
    "set_deterministic",
]
