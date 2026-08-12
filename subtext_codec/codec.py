"""Steganographic data encoding in LLM-generated text.

The payload drives an arithmetic decoder whose model is the language model's
own next-token distribution. Emitting a token of probability ``p`` consumes
``-log2(p)`` payload bits, so confident steps carry almost nothing and
uncertain ones carry a lot. Two consequences follow, and they are the whole
point of the design:

* The generated text is distributed exactly as ordinary sampling at the chosen
  temperature would be. Nothing is forced off-distribution.
* Capacity per token is the distribution's entropy, which is the information
  -theoretic ceiling for text that stays indistinguishable from sampling.

Recovering the payload replays the same distributions and runs the arithmetic
*encoder* over the observed tokens, which reproduces the bitstream.

Three invariants hold it together:

* **Symmetry** -- encoder and decoder derive the candidate set and its
  frequency table from the same function of the same token prefix.
* **Tokenizer stability** -- a candidate is only usable if appending it leaves
  the surrounding text re-tokenizing to the same ids. Without this the decoder
  reads a different id stream than the encoder wrote.
* **Self-inverse coding** -- the encoder re-encodes as it goes and checks it
  reproduces the payload before returning. This is not paranoia: feeding the
  coder a degenerate tail silently costs the payload's last two bits.
"""

from __future__ import annotations

import dataclasses
import json
import math
import struct
import zlib
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .arithmetic import (
    ENCODER_TAIL,
    FREQ_TOTAL,
    ArithmeticDecoder,
    ArithmeticEncoder,
    cumulative,
    from_bits,
    quantize_frequencies,
    to_bits,
)

#: Classic wire format, matching the 1.0 release: a single generated segment
#: whose body frames ``length || crc32 || data``. A message that needs neither
#: compression nor chunking is written in this format, so it stays byte-for-byte
#: readable by 1.0.
CODEC_VERSION = "v1"

#: Wire format added in 1.1 for the features 1.0 could not express: zlib
#: compression (signalled by the key's ``compression`` field) and a rolling
#: context window (signalled by the key's ``window`` field) that lets a payload
#: outrun the model's context. A 1.0 reader rejects a ``v2`` key rather than
#: mis-reading it, which is why the new features carry a new version.
CODEC_VERSION_V2 = "v2"

#: Versions this build can read. ``from_dict`` still distinguishes a genuine
#: pre-1.0 rank-coding key (which also called itself ``v1``) from these.
SUPPORTED_VERSIONS = frozenset({CODEC_VERSION, CODEC_VERSION_V2})

#: Compression codecs the ``compression`` key field may name. ``None`` means the
#: payload is stored verbatim, exactly as 1.0 did.
_COMPRESSIONS = frozenset({"zlib"})

#: Candidate cap used when a config leaves ``top_k`` unset.
DEFAULT_TOP_K = 64

#: Sampling temperature used when a config leaves it unset. This is the
#: capacity dial: 1.0 gives the most natural text but the least payload per
#: token, and raising it trades naturalness for density. 1.5 measures as the
#: knee of that curve on a 7B model.
DEFAULT_TEMPERATURE = 1.5

#: Fewest candidates a step can work with. Arithmetic coding needs no
#: terminator token, so two is genuinely enough.
MIN_ALPHABET = 2

#: Trailing tokens fed to the tokenizer-stability check. Comfortably longer
#: than the longest merge any mainstream BPE vocabulary performs.
SAFE_WINDOW = 16

#: Refuse payloads whose declared length is implausible rather than allocating
#: for them; a wrong key or model usually shows up as a nonsense length.
MAX_PAYLOAD_BYTES = 1 << 24

#: Backstop against a coder that stops making progress. Ordinary low-entropy
#: runs emit no bits for a few steps, so this is deliberately generous.
_STALL_LIMIT = 1000

#: length (uint32) + crc32 (uint32)
_HEADER = struct.Struct(">II")
_HEADER_BITS = _HEADER.size * 8

#: Called as ``progress(done, total)``; ``total`` is None when unknown.
ProgressFn = Callable[[int, Optional[int]], None]


@dataclasses.dataclass(frozen=True)
class EncodeStats:
    """What one encode produced, for reporting and capacity checks.

    ``bits_per_token`` is the user-facing density (raw payload bits over cover
    tokens). ``mean_surprisal_bits`` is the mean ``-log2(p)`` the coder actually
    spent per emitted token; the two being close is the confirmation that the
    coder is running at the distribution's entropy rather than leaving capacity
    on the table. They differ slightly because framing and (with compression)
    the transformed payload are what the coder encodes, not the raw bytes.

    Attributes:
        payload_bytes: Size of the original payload handed to the encoder.
        stored_bytes: Bytes actually encoded, after optional compression.
        tokens: Cover tokens generated (excludes the prompt).
        resets: How many times the rolling context window reset; 0 means the
            message fit in a single window.
        compressed: Whether the payload was zlib-compressed.
        bits_per_token: ``8 * payload_bytes / tokens`` (0 if no tokens).
        mean_surprisal_bits: Mean ``-log2(p)`` over the emitted tokens.
    """

    payload_bytes: int
    stored_bytes: int
    tokens: int
    resets: int
    compressed: bool
    bits_per_token: float
    mean_surprisal_bits: float


#: Called once when encoding finishes, with the run's :class:`EncodeStats`.
StatsFn = Callable[[EncodeStats], None]


# --------------------------------------------------------------------------
# configuration and keys
# --------------------------------------------------------------------------


def _validate_temperature(temperature: Optional[float]) -> float:
    if temperature is None:
        return DEFAULT_TEMPERATURE
    try:
        value = float(temperature)
    except (TypeError, ValueError) as exc:
        raise ValueError("temperature must be a positive number") from exc
    if not 0 < value <= 10:
        raise ValueError("temperature must be in the interval (0, 10]")
    return value


def _validate_top_k(top_k: Optional[int]) -> int:
    if top_k is None:
        return DEFAULT_TOP_K
    value = int(top_k)
    if value < MIN_ALPHABET:
        raise ValueError(f"top_k must be >= {MIN_ALPHABET}")
    return value


@dataclasses.dataclass
class CodecConfig:
    """Parameters controlling how a payload is hidden in generated text.

    Attributes:
        model_name_or_path: HuggingFace model id or local path to a causal LM.
        device: Torch device string, e.g. ``"cpu"``, ``"cuda"``, ``"cuda:0"``.
        prompt_prefix: Text preceding the generated content. It anchors the
            model's distribution and must be reproduced exactly to decode.
        max_context_length: Optional cap on total sequence length. Clamped to
            the model's own context window, which is used when this is None.
        top_k: Candidates considered per step. This bounds how much work the
            stability filter does; it is not a capacity dial.
        temperature: Sampling temperature, and the real capacity dial. Higher
            values flatten the distribution, so each token carries more payload
            and the cover text gets shorter but more erratic.
        torch_dtype: Model precision, one of ``auto``, ``float16``/``fp16``,
            ``bfloat16``/``bf16``, ``float32``/``fp32``. None keeps the
            checkpoint's dtype.
        store_model_in_key: Persist the model id in the key so decoding does
            not have to restate it.
        max_new_tokens: Optional ceiling on generated tokens, applied per
            segment; encoding raises rather than running away if a chunk does
            not fit.
        compress: zlib-compress the payload before encoding. Recorded in the
            key so decoding reverses it. Makes the payload bits more uniform
            (which the indistinguishability argument wants) and usually shortens
            the cover text. Uses the ``v2`` wire format.
    """

    model_name_or_path: str
    device: str
    prompt_prefix: str
    max_context_length: Optional[int] = None
    top_k: Optional[int] = DEFAULT_TOP_K
    temperature: float = DEFAULT_TEMPERATURE
    torch_dtype: Optional[str] = None
    store_model_in_key: bool = False
    max_new_tokens: Optional[int] = None
    compress: bool = False


@dataclasses.dataclass
class CodecKey:
    """Everything needed to turn encoded text back into the original bytes.

    Attributes:
        top_k: Candidate cap used at encode time. Must match to decode.
        temperature: Temperature used at encode time. Must match to decode.
        prompt_prefix: Prompt the encoded text was generated from.
        model_name_or_path: Model id, when the encoder was told to store it.
        device: Device used at encode time, reused as a decode default.
        torch_dtype: Precision used at encode time. Decoding in a different
            precision changes the logits and will not round-trip.
        version: Wire format, ``CODEC_VERSION`` (``v1``) or ``CODEC_VERSION_V2``
            (``v2``). ``v2`` is written only when compression or chunking is
            used, so a plain message stays readable by 1.0.
        compression: Codec applied to the payload before encoding, or ``None``.
            Only ``"zlib"`` is defined. A ``v1`` key never sets it.
        window: Rolling context size, in tokens, if the message was long enough
            to reset the context; ``None`` if it fit in one window (the common
            case, and always so for ``v1``). Decoding must reset at the same
            size, so it is carried in the key rather than recomputed.
    """

    top_k: Optional[int]
    temperature: Optional[float] = None
    prompt_prefix: Optional[str] = None
    model_name_or_path: Optional[str] = None
    device: Optional[str] = None
    torch_dtype: Optional[str] = None
    version: str = CODEC_VERSION
    compression: Optional[str] = None
    window: Optional[int] = None

    def to_dict(self) -> dict:
        if self.temperature is None:
            raise ValueError("temperature is required to serialize a codec key")
        data = {
            "version": self.version,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "prompt_prefix": self.prompt_prefix,
            "model_name_or_path": self.model_name_or_path,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
        }
        # Keep a v1 key byte-for-byte identical to what 1.0 wrote: the v2-only
        # fields appear only when they are actually in use.
        if self.compression is not None:
            data["compression"] = self.compression
        if self.window is not None:
            data["window"] = self.window
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "CodecKey":
        version = data.get("version")
        if version not in SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported codec key version: {version!r}")
        # Pre-1.0 releases also numbered a format "v1", but it was rank coding
        # and carried `base`/`top_p` instead of a temperature. Name that case
        # rather than letting it fail as a missing field.
        if "temperature" not in data and ("base" in data or "top_p" in data):
            raise ValueError(
                "this key predates subtext-codec 1.0 (it uses the old rank-coding "
                "format) and can no longer be read. Install subtext-codec~=0.2.0 "
                "to recover the message, or re-encode the payload"
            )

        temperature = data.get("temperature")
        if temperature is None:
            raise ValueError("temperature missing from codec key")
        top_k_raw = data.get("top_k")

        compression = data.get("compression")
        if compression is not None and compression not in _COMPRESSIONS:
            raise ValueError(f"unsupported compression: {compression!r}")

        window_raw = data.get("window")
        window = None if window_raw is None else int(window_raw)
        if window is not None and window < 2:
            raise ValueError(f"window must be at least 2 tokens, got {window}")

        return cls(
            top_k=None if top_k_raw is None else int(top_k_raw),
            temperature=_validate_temperature(float(temperature)),
            prompt_prefix=data.get("prompt_prefix"),
            model_name_or_path=data.get("model_name_or_path"),
            device=data.get("device"),
            torch_dtype=data.get("torch_dtype"),
            version=version,
            compression=compression,
            window=window,
        )


def save_codec_key(key: CodecKey, path: str) -> None:
    """Write a codec key to ``path`` as JSON, overwriting any existing file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(key.to_dict(), f, indent=2)
        f.write("\n")


def load_codec_key(path: str) -> CodecKey:
    """Read a codec key from a JSON file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a readable key for this version.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return CodecKey.from_dict(raw)


# --------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------


def set_deterministic(seed: int = 0) -> None:
    """Seed torch and request deterministic, full-precision kernels.

    Encoding and decoding must observe identical logits, so this pins the RNGs,
    disables autotuning, and holds fp32 matmuls at full precision. Full CUDA
    determinism additionally needs ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` in the
    environment before torch is imported; importing :mod:`subtext_codec` sets
    that default for you.

    Note that this mutates global torch state, including for code outside this
    package. That is deliberate: a mismatch here costs the whole message.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # TF32 is deterministic in itself, but whether it is enabled varies with the
    # torch version and the GPU generation, and it carries only ~10 mantissa
    # bits. A message encoded with it on will not decode where it is off, so pin
    # it rather than inherit whatever the environment happens to default to.
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)


_DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _parse_torch_dtype(dtype: Optional[str]) -> Optional[Union[str, torch.dtype]]:
    if dtype is None:
        return None
    lowered = dtype.lower()
    if lowered == "auto":
        return "auto"
    if lowered not in _DTYPES:
        raise ValueError(
            "torch-dtype must be one of: auto, float16/fp16/half, "
            "bfloat16/bf16, float32/fp32"
        )
    return _DTYPES[lowered]


def load_model_and_tokenizer(
    model_name_or_path: str,
    device: str,
    torch_dtype: Optional[str] = None,
    seed: int = 0,
):
    """Load a causal LM and its tokenizer in eval mode with deterministic settings.

    The same checkpoint *and* the same dtype must be used for encoding and
    decoding; changing either perturbs the logits enough to reorder candidates.

    Returns:
        ``(tokenizer, model)``.
    """
    set_deterministic(seed)
    resolved_dtype = _parse_torch_dtype(torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model_kwargs = {}
    if resolved_dtype is not None:
        # `torch_dtype` was renamed to `dtype` in transformers 4.56.
        model_kwargs["dtype"] = resolved_dtype
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.to(device)
    model.eval()
    return tokenizer, model


def _model_context_limit(model) -> Optional[int]:
    config = getattr(model, "config", None)
    for attr in ("max_position_embeddings", "n_positions", "n_ctx"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _resolve_context_limit(model, requested: Optional[int]) -> Optional[int]:
    limits = [x for x in (requested, _model_context_limit(model)) if x is not None]
    return min(limits) if limits else None


# --------------------------------------------------------------------------
# payload framing
# --------------------------------------------------------------------------


def _frame_payload(data: bytes) -> List[int]:
    """Length and checksum, then the data, as a bitstream."""
    return to_bits(_HEADER.pack(len(data), zlib.crc32(data)) + data)


def _unframe_payload(bits: Sequence[int]) -> bytes:
    """Inverse of :func:`_frame_payload`, with the header checked."""
    if len(bits) < _HEADER_BITS:
        raise ValueError(
            "not enough data recovered to read the header; the message may be "
            "truncated, or the key, prompt or model may not match the encoder"
        )
    length, checksum = _HEADER.unpack(from_bits(bits[:_HEADER_BITS]))
    if length > MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"decoded payload length ({length} bytes) is implausible; the key, "
            "prompt or model almost certainly do not match the encoder"
        )
    needed = _HEADER_BITS + 8 * length
    if len(bits) < needed:
        raise ValueError(
            f"message declares {length} bytes but only "
            f"{(len(bits) - _HEADER_BITS) // 8} were recovered; it is truncated"
        )
    data = from_bits(bits[_HEADER_BITS:needed])
    if zlib.crc32(data) != checksum:
        raise ValueError(
            "decoded payload failed its checksum; the message was altered, or "
            "the key, prompt or model do not match the encoder"
        )
    return data


def _declared_length(bits: Sequence[int]) -> Optional[int]:
    """Total bits the message occupies, once the header is readable."""
    if len(bits) < _HEADER_BITS:
        return None
    length, _ = _HEADER.unpack(from_bits(bits[:_HEADER_BITS]))
    if length > MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"decoded payload length ({length} bytes) is implausible; the key, "
            "prompt or model almost certainly do not match the encoder"
        )
    return _HEADER_BITS + 8 * length


# --------------------------------------------------------------------------
# per-step distributions
# --------------------------------------------------------------------------


def _token_ids(tokenizer, text: str) -> List[int]:
    return [int(i) for i in tokenizer(text)["input_ids"]]


def _special_ids(tokenizer) -> frozenset:
    return frozenset(int(i) for i in (getattr(tokenizer, "all_special_ids", None) or ()))


def _stable_candidates(
    tokenizer, context_ids: Sequence[int], candidates: Sequence[int]
) -> List[int]:
    """Keep only candidates that append cleanly to the text of ``context_ids``.

    A candidate survives when writing out ``context + [candidate]`` and reading
    it back yields exactly the previous tokenization plus that one token. This
    rejects tokens that would merge with preceding characters, partial UTF-8
    fragments, and special tokens that vanish when the text is written out --
    each of which would make the decoder recover a different id stream than the
    encoder chose.

    Only the trailing ``SAFE_WINDOW`` tokens are examined. The comparison is
    relative to the window's own re-tokenization, so it does not matter that
    the window may start mid-word.
    """
    specials = _special_ids(tokenizer)
    usable = [c for c in candidates if c not in specials]
    if not usable:
        return []

    window = list(context_ids)[-SAFE_WINDOW:]
    sequences = [window] + [window + [c] for c in usable]
    texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    encodings = tokenizer(texts, add_special_tokens=False)["input_ids"]

    base_text, base_ids = texts[0], [int(i) for i in encodings[0]]
    width = len(base_ids)

    stable: List[int] = []
    for candidate, text, encoded in zip(usable, texts[1:], encodings[1:]):
        ids = [int(i) for i in encoded]
        if len(ids) != width + 1 or ids[width] != candidate or ids[:width] != base_ids:
            continue
        if not text.startswith(base_text) or len(text) == len(base_text):
            continue
        stable.append(candidate)
    return stable


def _step_distribution(
    logits: torch.Tensor,
    context_ids: Sequence[int],
    tokenizer,
    top_k: int,
    temperature: float,
) -> Tuple[List[int], List[int]]:
    """Candidate tokens for one step and their cumulative frequency table."""
    # float32 keeps the softmax well-conditioned for bf16/fp16 checkpoints, and
    # a stable sort makes tie order a property of the token ids rather than of
    # whichever sort kernel torch happens to dispatch to.
    scores = logits.detach().to(torch.float32)
    ranked = torch.argsort(scores, descending=True, stable=True)[:top_k].tolist()
    alphabet = _stable_candidates(tokenizer, context_ids, ranked)
    if len(alphabet) < MIN_ALPHABET:
        raise ValueError(
            f"only {len(alphabet)} of the top-{top_k} tokens are safe to emit here "
            f"(need {MIN_ALPHABET}); raise top_k or choose a different prompt_prefix"
        )
    subset = scores[torch.tensor(alphabet, dtype=torch.long)] / temperature
    probs = torch.softmax(subset.double(), dim=0)
    return alphabet, cumulative(quantize_frequencies(probs))


class _Stepper:
    """Incremental forward pass over a growing prefix, reusing the KV cache.

    Encoding and decoding both drive this the same way -- prefill the prompt,
    then feed one token at a time -- so the logits they observe at each step
    are produced by identical work, which is what the codec's reversibility
    depends on.

    With a ``window`` the context is a rolling one: when it would exceed
    ``window`` tokens, the last ``window // 2`` tokens are re-prefilled from a
    fresh cache and generation continues from there. This is what lets a payload
    outrun the model's context window -- the text stays one continuous passage,
    conditioned on a sliding window of recent tokens rather than a re-inserted
    prompt. The reset is driven purely by token count, so encode and decode
    reset at the same steps and observe the same logits. ``window=None`` keeps
    the full context, which is the 1.0 behaviour.
    """

    def __init__(
        self, model, device: str, prompt_ids: Sequence[int], window: Optional[int] = None
    ):
        if len(prompt_ids) == 0:
            raise ValueError("prompt_prefix must tokenize to at least one token")
        if window is not None and len(prompt_ids) > window:
            raise ValueError(
                f"prompt_prefix is {len(prompt_ids)} tokens but the context window "
                f"is {window}; shorten the prompt or raise --max-context-length"
            )
        self._model = model
        self._device = device
        self._window = window
        self._carryover = max(1, window // 2) if window else None
        self._cache = None
        self._logits: Optional[torch.Tensor] = None
        self.length = 0
        #: Recent token ids kept for a context reset; bounded by the carryover.
        self._recent: List[int] = []
        #: How many times the context has been reset; 0 means it never rolled.
        self.resets = 0
        self._prefill(list(prompt_ids))

    def _feed(self, token_ids: List[int]) -> None:
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self._device)
        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids, past_key_values=self._cache, use_cache=True
            )
        self._cache = outputs.past_key_values
        self._logits = outputs.logits[0, -1, :].detach().to("cpu")

    def _prefill(self, token_ids: List[int]) -> None:
        """Fresh forward pass over ``token_ids`` from an empty cache."""
        self._cache = None
        self._feed(token_ids)
        self.length = len(token_ids)
        self._recent = (
            token_ids[-self._carryover:] if self._carryover else list(token_ids)
        )

    @property
    def logits(self) -> torch.Tensor:
        assert self._logits is not None
        return self._logits

    def advance(self, token_id: int) -> None:
        token_id = int(token_id)
        if self._window is not None and self.length + 1 > self._window:
            # Re-prefill the most recent carryover tokens and roll forward. The
            # logits from that prefill are discarded; the token below is what we
            # actually need a distribution after.
            keep = list(self._recent[-self._carryover:])
            self._prefill(keep)
            self.resets += 1
        self._feed([token_id])
        self.length += 1
        if self._carryover:
            self._recent = (self._recent + [token_id])[-self._carryover:]
        else:
            self._recent.append(token_id)


def _check_limit(length: int, limit: Optional[int]) -> None:
    if limit is not None and length > limit:
        raise ValueError(
            f"sequence length {length} exceeds the context limit {limit}; "
            "shorten the payload or the prompt, or raise --max-context-length"
        )


# --------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------


def _verify_text_round_trip(tokenizer, text: str, ids: Sequence[int]) -> None:
    actual = _token_ids(tokenizer, text)
    if actual[: len(ids)] != list(ids):
        divergence = next(
            (i for i, (a, b) in enumerate(zip(actual, ids)) if a != b), len(ids)
        )
        raise ValueError(
            "generated text does not tokenize back to the tokens it was built "
            f"from (first difference at token {divergence}); this message would "
            "not decode. Try a different prompt_prefix or a lower top_k"
        )


def _frame_compressed(data: bytes) -> List[int]:
    """Frame ``data`` for compressed transport: ``len(zlib) || crc32(data) || zlib``.

    The length is over the compressed bytes (so the decoder knows when to stop),
    but the CRC is over the *original* bytes. That is what catches a key whose
    ``compression`` field is wrong in either direction: a compressed message read
    as plain fails the CRC over the compressed bytes, and a plain message read as
    compressed fails to decompress.
    """
    stored = zlib.compress(data, 9)
    return to_bits(_HEADER.pack(len(stored), zlib.crc32(data)) + stored)


def _unframe_compressed(bits: Sequence[int]) -> bytes:
    """Inverse of :func:`_frame_compressed`, with the original-data CRC checked."""
    if len(bits) < _HEADER_BITS:
        raise ValueError(
            "not enough data recovered to read the header; the message may be "
            "truncated, or the key, prompt or model may not match the encoder"
        )
    length, checksum = _HEADER.unpack(from_bits(bits[:_HEADER_BITS]))
    if length > MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"decoded payload length ({length} bytes) is implausible; the key, "
            "prompt or model almost certainly do not match the encoder"
        )
    needed = _HEADER_BITS + 8 * length
    if len(bits) < needed:
        raise ValueError(
            f"message declares {length} compressed bytes but only "
            f"{(len(bits) - _HEADER_BITS) // 8} were recovered; it is truncated"
        )
    stored = from_bits(bits[_HEADER_BITS:needed])
    try:
        data = zlib.decompress(stored)
    except zlib.error as exc:
        raise ValueError(
            "failed to decompress the recovered payload; the key, prompt or model "
            "do not match the encoder"
        ) from exc
    if zlib.crc32(data) != checksum:
        raise ValueError(
            "decoded payload failed its checksum; the message was altered, or the "
            "key, prompt or model do not match the encoder"
        )
    return data


def encode_data_to_text(
    data: bytes,
    cfg: CodecConfig,
    model,
    tokenizer,
    progress: Optional[ProgressFn] = None,
    report_stats: Optional[StatsFn] = None,
) -> Tuple[str, CodecKey]:
    """Hide ``data`` inside text generated from ``cfg.prompt_prefix``.

    The framed payload drives an arithmetic decoder against the model's own
    distribution, so each emitted token is drawn exactly as sampling at
    ``cfg.temperature`` would draw it, and carries ``-log2(p)`` payload bits.
    Generation stops as soon as the emitted tokens pin down every payload bit.

    ``cfg.compress`` zlib-compresses the payload first. A payload too long for
    one context window is handled automatically by a rolling window: when the
    context fills, the last half is re-prefixed and generation continues, so the
    text stays one flowing passage with no repeated prompt. Either feature uses
    the ``v2`` wire format (recorded in the key); a plain message that fits one
    window stays ``v1`` and byte-compatible with 1.0.

    Args:
        data: Payload to hide. Any bytes, including empty.
        cfg: Encoding parameters.
        model: Causal LM from :func:`load_model_and_tokenizer`.
        tokenizer: Its tokenizer.
        progress: Optional ``progress(bits_written, bits_total)`` callback.
        report_stats: Optional callback invoked once with an :class:`EncodeStats`.

    Returns:
        ``(text, key)``. The text begins with ``cfg.prompt_prefix``; the key is
        required to decode and should be kept as carefully as a password.

    Raises:
        ValueError: If parameters are invalid, the payload does not fit within
            ``max_new_tokens``, too few candidates are safe to emit, or the
            generated text does not tokenize back to itself.
    """
    temperature = _validate_temperature(cfg.temperature)
    top_k = _validate_top_k(cfg.top_k)
    limit = _resolve_context_limit(model, cfg.max_context_length)

    # A payload past this size frames a length the decoder rejects as
    # implausible, so refuse it here rather than emit a message that cannot be
    # decoded -- the same ceiling _unframe_payload enforces on the way back.
    if len(data) > MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"payload is {len(data)} bytes; the maximum is {MAX_PAYLOAD_BYTES} "
            "(a larger message could not be decoded)"
        )

    compressed = bool(cfg.compress)
    if compressed:
        payload_bits = _frame_compressed(data)
        stored_bytes = zlib.compress(data, 9)  # for the stats line only
    else:
        payload_bits = _frame_payload(data)
        stored_bytes = data
    total_bits = len(payload_bits)

    prompt_ids = _token_ids(tokenizer, cfg.prompt_prefix)
    # The window is the model's context limit; the stepper resets at it so a
    # long payload keeps flowing. None (no known limit) means a single full
    # context, which raises below if the payload outgrows it.
    window = limit
    stepper = _Stepper(model, cfg.device, prompt_ids, window=window)
    ids = list(prompt_ids)

    # The tail keeps the coder out of the degenerate all-zeros state that would
    # otherwise cost the payload's last two bits; see arithmetic.ENCODER_TAIL.
    reader = ArithmeticDecoder(payload_bits + ENCODER_TAIL)
    # Re-encode as we go. Only bits the mirror has actually emitted are pinned
    # down by the tokens so far, which makes this the correct stopping rule --
    # and the check below turns any coder defect into a loud failure.
    mirror = ArithmeticEncoder()

    generated = 0
    stagnant = 0
    surprisal = 0.0
    while len(mirror.bits) < total_bits:
        if cfg.max_new_tokens is not None and generated >= cfg.max_new_tokens:
            raise ValueError(
                f"payload needs more than max_new_tokens={cfg.max_new_tokens} tokens; "
                "raise it, raise the temperature, or shrink the payload"
            )
        if stagnant >= _STALL_LIMIT:
            raise ValueError(
                f"the arithmetic coder stopped making progress after {generated} "
                f"tokens ({len(mirror.bits)}/{total_bits} bits)"
            )

        emitted = len(mirror.bits)
        alphabet, cum = _step_distribution(
            stepper.logits, ids, tokenizer, top_k, temperature
        )
        symbol = reader.decode(cum)
        mirror.encode(symbol, cum)
        surprisal += -math.log2((cum[symbol + 1] - cum[symbol]) / FREQ_TOTAL)

        token = alphabet[symbol]
        ids.append(token)
        generated += 1
        if window is None:
            _check_limit(len(ids), limit)
        stepper.advance(token)

        stagnant = 0 if len(mirror.bits) > emitted else stagnant + 1
        if progress is not None:
            progress(min(len(mirror.bits), total_bits), total_bits)

    if mirror.bits[:total_bits] != payload_bits:
        raise ValueError(
            "internal error: the arithmetic coder did not reproduce the payload; "
            "refusing to emit a message that cannot be decoded"
        )

    text = tokenizer.decode(ids, skip_special_tokens=True)
    _verify_text_round_trip(tokenizer, text, ids)

    rolled = stepper.resets > 0
    key = CodecKey(
        top_k=top_k,
        temperature=temperature,
        prompt_prefix=cfg.prompt_prefix,
        model_name_or_path=cfg.model_name_or_path if cfg.store_model_in_key else None,
        device=cfg.device,
        torch_dtype=cfg.torch_dtype,
        version=CODEC_VERSION_V2 if (compressed or rolled) else CODEC_VERSION,
        compression="zlib" if compressed else None,
        window=window if rolled else None,
    )

    if report_stats is not None:
        report_stats(
            EncodeStats(
                payload_bytes=len(data),
                stored_bytes=len(stored_bytes),
                tokens=generated,
                resets=stepper.resets,
                compressed=compressed,
                bits_per_token=(8 * len(data) / generated) if generated else 0.0,
                mean_surprisal_bits=(surprisal / generated) if generated else 0.0,
            )
        )
    return text, key


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------


def decode_text_to_data(
    encoded_text: str,
    key: CodecKey,
    prompt_prefix: str,
    model,
    tokenizer,
    device: str,
    max_context_length: Optional[int] = None,
    progress: Optional[ProgressFn] = None,
) -> bytes:
    """Recover the original payload from encoded text.

    Replays the same per-step distributions the encoder saw and runs the
    arithmetic encoder over the observed tokens, which reproduces the payload
    bitstream. Reading stops as soon as the declared length has been recovered,
    so text after the message is ignored, as is text before the prompt. If the
    key records a rolling ``window``, the context is reset at the same points
    the encoder used, so a long message decodes exactly as it was written.

    Args:
        encoded_text: Text containing the encoded message.
        key: Key produced at encode time.
        prompt_prefix: Prompt used at encode time; must match exactly.
        model: The same checkpoint, at the same precision, used to encode.
        tokenizer: Its tokenizer.
        device: Torch device string.
        max_context_length: Optional cap on sequence length.
        progress: Optional ``progress(tokens_read, tokens_total)`` callback.

    Returns:
        The original payload, byte for byte.

    Raises:
        ValueError: If the prompt does not match or cannot be located, the
            message is truncated, or the payload fails its checksum -- which is
            what a wrong model, key or prompt looks like.
    """
    if key.version not in SUPPORTED_VERSIONS:
        raise ValueError(f"Unsupported codec key version: {key.version!r}")
    if key.prompt_prefix is not None and key.prompt_prefix != prompt_prefix:
        raise ValueError("Prompt prefix does not match codec key")

    temperature = _validate_temperature(key.temperature)
    top_k = _validate_top_k(key.top_k)
    prompt_ids = _token_ids(tokenizer, prompt_prefix)

    start = encoded_text.find(prompt_prefix)
    if start == -1:
        raise ValueError("prompt_prefix not found in encoded_text")

    ids = _token_ids(tokenizer, encoded_text[start:])
    if ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "the encoded text does not tokenize to the prompt_prefix followed by "
            "the message; text preceding the prompt may have merged into it"
        )
    body = ids[len(prompt_ids):]
    total_tokens = max(len(body), 1)

    # The key's window is what the encoder rolled at; None means it never did.
    # The stepper tracks the (rolling) model context; `context` is the full
    # token history the stability filter needs, exactly as encode kept it.
    stepper = _Stepper(model, device, prompt_ids, window=key.window)
    context = list(prompt_ids)
    writer = ArithmeticEncoder()
    needed: Optional[int] = None

    for index, token in enumerate(body):
        if needed is not None and len(writer.bits) >= needed:
            break
        alphabet, cum = _step_distribution(
            stepper.logits, context, tokenizer, top_k, temperature
        )
        # A token outside the candidate set means the stream has diverged --
        # most often because trailing text merged into the final token. Stop and
        # let the header and checksum judge what was recovered.
        if token not in alphabet:
            break

        writer.encode(alphabet.index(token), cum)
        context.append(token)
        stepper.advance(token)

        if needed is None:
            needed = _declared_length(writer.bits)
        if progress is not None:
            progress(index + 1, total_tokens)

    if key.compression == "zlib":
        return _unframe_compressed(writer.bits)
    if key.compression is not None:
        raise ValueError(f"unsupported compression: {key.compression!r}")
    return _unframe_payload(writer.bits)


__all__ = [
    "CODEC_VERSION",
    "CODEC_VERSION_V2",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_K",
    "SUPPORTED_VERSIONS",
    "CodecConfig",
    "CodecKey",
    "EncodeStats",
    "decode_text_to_data",
    "encode_data_to_text",
    "load_codec_key",
    "load_model_and_tokenizer",
    "save_codec_key",
    "set_deterministic",
]
