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
#: compression (signalled by the key's ``compression`` field) and multi-segment
#: chunking (the payload split across several re-anchored segments, each framed
#: with a chunk header). A 1.0 reader rejects a ``v2`` key rather than
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

#: v2 per-chunk header: index (uint16) + count (uint16) + length (uint32) +
#: crc32 (uint32). Every segment of a chunked message carries one, so the
#: decoder learns the total count from the first segment and can tell a genuine
#: segment boundary from noise. ``count`` bounds a message to 65535 segments.
_CHUNK_HEADER = struct.Struct(">HHII")
_CHUNK_HEADER_BITS = _CHUNK_HEADER.size * 8
MAX_SEGMENTS = (1 << 16) - 1

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
        tokens: Cover tokens generated, summed across segments (excludes the
            prompt, which is repeated once per segment).
        segments: Number of re-anchored segments the message occupies.
        compressed: Whether the payload was zlib-compressed.
        bits_per_token: ``8 * payload_bytes / tokens`` (0 if no tokens).
        mean_surprisal_bits: Mean ``-log2(p)`` over the emitted tokens.
    """

    payload_bytes: int
    stored_bytes: int
    tokens: int
    segments: int
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
        chunk_bytes: If set and the (post-compression) payload is larger, split
            it into chunks of this many bytes, each encoded as its own
            re-anchored segment. This is how a payload can exceed the model's
            context window. Uses the ``v2`` wire format.
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
    chunk_bytes: Optional[int] = None


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
    """

    top_k: Optional[int]
    temperature: Optional[float] = None
    prompt_prefix: Optional[str] = None
    model_name_or_path: Optional[str] = None
    device: Optional[str] = None
    torch_dtype: Optional[str] = None
    version: str = CODEC_VERSION
    compression: Optional[str] = None

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
        # Keep a v1 key byte-for-byte identical to what 1.0 wrote: the
        # compression field appears only when it is actually in use (v2).
        if self.compression is not None:
            data["compression"] = self.compression
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

        return cls(
            top_k=None if top_k_raw is None else int(top_k_raw),
            temperature=_validate_temperature(float(temperature)),
            prompt_prefix=data.get("prompt_prefix"),
            model_name_or_path=data.get("model_name_or_path"),
            device=data.get("device"),
            torch_dtype=data.get("torch_dtype"),
            version=version,
            compression=compression,
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


def _frame_chunk(index: int, count: int, data: bytes) -> List[int]:
    """One segment's chunk header and data, as a bitstream (v2)."""
    header = _CHUNK_HEADER.pack(index, count, len(data), zlib.crc32(data))
    return to_bits(header + data)


def _unframe_chunk(bits: Sequence[int]) -> Tuple[int, int, bytes]:
    """Inverse of :func:`_frame_chunk`: ``(index, count, data)`` with the CRC checked."""
    if len(bits) < _CHUNK_HEADER_BITS:
        raise ValueError(
            "not enough data recovered to read the chunk header; the message may "
            "be truncated, or the key, prompt or model may not match the encoder"
        )
    index, count, length, checksum = _CHUNK_HEADER.unpack(
        from_bits(bits[:_CHUNK_HEADER_BITS])
    )
    if length > MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"decoded chunk length ({length} bytes) is implausible; the key, "
            "prompt or model almost certainly do not match the encoder"
        )
    needed = _CHUNK_HEADER_BITS + 8 * length
    if len(bits) < needed:
        raise ValueError(
            f"chunk declares {length} bytes but only "
            f"{(len(bits) - _CHUNK_HEADER_BITS) // 8} were recovered; it is truncated"
        )
    data = from_bits(bits[_CHUNK_HEADER_BITS:needed])
    if zlib.crc32(data) != checksum:
        raise ValueError(
            "decoded chunk failed its checksum; the message was altered, or the "
            "key, prompt or model do not match the encoder"
        )
    return index, count, data


def _declared_chunk_length(bits: Sequence[int]) -> Optional[int]:
    """Total bits a chunk occupies, once its header is readable (v2)."""
    if len(bits) < _CHUNK_HEADER_BITS:
        return None
    _, _, length, _ = _CHUNK_HEADER.unpack(from_bits(bits[:_CHUNK_HEADER_BITS]))
    if length > MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"decoded chunk length ({length} bytes) is implausible; the key, "
            "prompt or model almost certainly do not match the encoder"
        )
    return _CHUNK_HEADER_BITS + 8 * length


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
    """

    def __init__(self, model, device: str, prompt_ids: Sequence[int]):
        if len(prompt_ids) == 0:
            raise ValueError("prompt_prefix must tokenize to at least one token")
        self._model = model
        self._device = device
        self._cache = None
        self._logits: Optional[torch.Tensor] = None
        self.length = 0
        self._feed(list(prompt_ids))

    def _feed(self, token_ids: List[int]) -> None:
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self._device)
        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids, past_key_values=self._cache, use_cache=True
            )
        self._cache = outputs.past_key_values
        self._logits = outputs.logits[0, -1, :].detach().to("cpu")
        self.length += len(token_ids)

    @property
    def logits(self) -> torch.Tensor:
        assert self._logits is not None
        return self._logits

    def advance(self, token_id: int) -> None:
        self._feed([int(token_id)])


def _check_limit(length: int, limit: Optional[int]) -> None:
    if limit is not None and length > limit:
        raise ValueError(
            f"sequence length {length} exceeds the context limit {limit}; "
            "shorten the payload or the prompt, or raise --max-context-length"
        )


# --------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------


def _verify_segments_round_trip(
    tokenizer,
    text: str,
    prompt_prefix: str,
    prompt_ids: Sequence[int],
    segments: Sequence[Tuple[Sequence[int], int]],
) -> None:
    """Check every segment's payload is recoverable, exactly as decode reads it.

    Walks the text the way the decoder will -- find the prompt, tokenize the
    slice, and confirm the prompt plus that segment's payload tokens come back
    unchanged -- then steps past the payload to the next segment. Boundary
    buffer tokens past the payload are deliberately not checked: the decoder
    stops at the chunk's declared length and never reads them. For a single
    segment this reduces to the classic whole-text round-trip check.
    """
    prompt_len = len(prompt_ids)
    offset = 0
    for seg_index, (body, payload_tokens) in enumerate(segments):
        start = text.find(prompt_prefix, offset)
        if start == -1:
            raise ValueError(
                f"internal error: segment {seg_index}'s prompt is missing from the "
                "generated text; refusing to emit a message that cannot be decoded"
            )
        want = list(prompt_ids) + list(body[:payload_tokens])
        got = _token_ids(tokenizer, text[start:])
        if got[: len(want)] != want:
            divergence = next(
                (i for i, (a, b) in enumerate(zip(got, want)) if a != b), len(want)
            )
            raise ValueError(
                "generated text does not tokenize back to the tokens it was built "
                f"from (segment {seg_index}, first difference at token {divergence}); "
                "this message would not decode. Try a different prompt_prefix or a "
                "lower top_k"
            )
        offset = start + len(tokenizer.decode(want, skip_special_tokens=True))


def _payload_survives_next_prompt(
    tokenizer, ids: Sequence[int], payload_len: int, prompt_text: str
) -> bool:
    """Do the first ``payload_len`` tokens survive the next prompt being appended?

    A segment's last payload token can merge into the re-anchored prompt that
    follows it (byte-level BPE reading ``' '`` + ``'a'`` back as ``' a'``), which
    would make the decoder recover a different token. Appending the prompt text
    to the segment so far and re-tokenizing is exactly the boundary the decoder
    will see; the payload tokens must come back unchanged.
    """
    seg_text = tokenizer.decode(ids, skip_special_tokens=True)
    retok = _token_ids(tokenizer, seg_text + prompt_text)
    return retok[:payload_len] == list(ids[:payload_len])


def _encode_one_segment(
    framed_bits: Sequence[int],
    cfg: CodecConfig,
    model,
    tokenizer,
    prompt_ids: Sequence[int],
    top_k: int,
    temperature: float,
    limit: Optional[int],
    progress: Optional[ProgressFn],
    bits_before: int,
    total_all_bits: int,
    pad_boundary: bool,
) -> Tuple[List[int], int, float]:
    """Generate one segment's tokens for ``framed_bits`` from a fresh prompt.

    Each segment re-prefills the prompt (a new stepper, a new coder pair), so
    it stands on its own within the context window -- that is what lets a
    chunked message exceed the model's context.

    When ``pad_boundary`` is set (every segment but the last), a few extra
    tokens are generated past the payload until the payload tokens are stable
    against the next prompt. These buffer tokens sit beyond the chunk's declared
    length, so the decoder stops before them and skips them when it searches for
    the next prompt. Returns ``(body_ids, payload_tokens, surprisal_bits)``;
    ``payload_tokens`` counts only the data-carrying tokens, and ``body_ids``
    includes any buffer.
    """
    total_bits = len(framed_bits)
    stepper = _Stepper(model, cfg.device, prompt_ids)
    ids = list(prompt_ids)

    # The tail keeps the coder out of the degenerate all-zeros state that would
    # otherwise cost the payload's last two bits; see arithmetic.ENCODER_TAIL.
    # A little more than the tail lets the boundary buffer stay model-driven
    # rather than parking the coder on the interval midpoint.
    reader = ArithmeticDecoder(list(framed_bits) + ENCODER_TAIL * (2 if pad_boundary else 1))
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
                f"a segment needs more than max_new_tokens={cfg.max_new_tokens} "
                "tokens; raise it, raise the temperature, shrink the payload, or "
                "lower chunk_bytes"
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
        _check_limit(len(ids), limit)
        stepper.advance(token)

        stagnant = 0 if len(mirror.bits) > emitted else stagnant + 1
        if progress is not None:
            progress(min(bits_before + len(mirror.bits), total_all_bits), total_all_bits)

    if mirror.bits[:total_bits] != list(framed_bits):
        raise ValueError(
            "internal error: the arithmetic coder did not reproduce the payload; "
            "refusing to emit a message that cannot be decoded"
        )

    payload_tokens = generated
    if pad_boundary:
        payload_len = len(prompt_ids) + payload_tokens
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        # SAFE_WINDOW bounds how far a merge can reach, so that many buffer
        # tokens is always enough to move the payload out of its range.
        for _ in range(SAFE_WINDOW + 1):
            if _payload_survives_next_prompt(tokenizer, ids, payload_len, prompt_text):
                break
            alphabet, cum = _step_distribution(
                stepper.logits, ids, tokenizer, top_k, temperature
            )
            token = alphabet[reader.decode(cum)]
            ids.append(token)
            _check_limit(len(ids), limit)
            stepper.advance(token)
        else:
            raise ValueError(
                "could not stabilize a segment boundary against the prompt; try a "
                "different prompt_prefix or a larger chunk_bytes"
            )

    return ids[len(prompt_ids):], payload_tokens, surprisal


def _chunk_payload(payload: bytes, chunk_bytes: Optional[int]) -> List[bytes]:
    """Split ``payload`` into chunks of at most ``chunk_bytes`` bytes.

    ``None`` (or a payload that already fits) yields a single chunk, which is
    the classic single-segment case.
    """
    if chunk_bytes is None or len(payload) <= chunk_bytes:
        return [payload]
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be >= 1")
    return [payload[i : i + chunk_bytes] for i in range(0, len(payload), chunk_bytes)]


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

    With ``cfg.compress`` the payload is zlib-compressed first; with
    ``cfg.chunk_bytes`` it is split across several re-anchored segments so it
    can exceed the model's context window. Either uses the ``v2`` wire format
    (recorded in the key); a plain message stays ``v1`` and byte-compatible
    with 1.0.

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
            the context or ``max_new_tokens``, too few candidates are safe to
            emit, or the generated text does not tokenize back to itself.
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

    # A v2 message frames the whole payload once (length + crc over the original
    # data) before compressing or chunking. Reassembly checks that manifest, so
    # a wrong compression flag, a lost segment, or corruption is caught rather
    # than returning plausible bytes -- the same guarantee v1's framing gives.
    compressed = bool(cfg.compress)
    manifest = _HEADER.pack(len(data), zlib.crc32(data)) + data
    stored = zlib.compress(manifest, 9) if compressed else manifest
    will_chunk = cfg.chunk_bytes is not None and len(stored) > cfg.chunk_bytes
    use_v2 = compressed or will_chunk

    if use_v2:
        chunks = _chunk_payload(stored, cfg.chunk_bytes)
        count = len(chunks)
        if count > MAX_SEGMENTS:
            raise ValueError(
                f"payload needs {count} segments; the maximum is {MAX_SEGMENTS}. "
                "Raise chunk_bytes."
            )
        framed_list = [_frame_chunk(i, count, chunk) for i, chunk in enumerate(chunks)]
    else:
        count = 1
        framed_list = [_frame_payload(data)]
    total_all_bits = sum(len(f) for f in framed_list)

    prompt_ids = _token_ids(tokenizer, cfg.prompt_prefix)
    _check_limit(len(prompt_ids), limit)

    full_ids: List[int] = []
    segments: List[Tuple[List[int], int]] = []  # (body incl. buffer, payload tokens)
    tokens_total = 0
    surprisal_total = 0.0
    bits_before = 0
    for position, framed in enumerate(framed_list):
        body, generated, surprisal = _encode_one_segment(
            framed, cfg, model, tokenizer, prompt_ids, top_k, temperature, limit,
            progress, bits_before, total_all_bits,
            pad_boundary=position < len(framed_list) - 1,
        )
        segments.append((body, generated))
        full_ids.extend(prompt_ids)
        full_ids.extend(body)
        tokens_total += generated
        surprisal_total += surprisal
        bits_before += len(framed)

    text = tokenizer.decode(full_ids, skip_special_tokens=True)
    _verify_segments_round_trip(tokenizer, text, cfg.prompt_prefix, prompt_ids, segments)

    key = CodecKey(
        top_k=top_k,
        temperature=temperature,
        prompt_prefix=cfg.prompt_prefix,
        model_name_or_path=cfg.model_name_or_path if cfg.store_model_in_key else None,
        device=cfg.device,
        torch_dtype=cfg.torch_dtype,
        version=CODEC_VERSION_V2 if use_v2 else CODEC_VERSION,
        compression="zlib" if compressed else None,
    )

    if report_stats is not None:
        report_stats(
            EncodeStats(
                payload_bytes=len(data),
                stored_bytes=len(stored) if compressed else len(data),
                tokens=tokens_total,
                segments=count,
                compressed=compressed,
                bits_per_token=(8 * len(data) / tokens_total) if tokens_total else 0.0,
                mean_surprisal_bits=(surprisal_total / tokens_total) if tokens_total else 0.0,
            )
        )
    return text, key


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------


def _decode_one_segment(
    encoded_text: str,
    start: int,
    prompt_ids: Sequence[int],
    model,
    tokenizer,
    device: str,
    top_k: int,
    temperature: float,
    limit: Optional[int],
    declared_length: Callable[[Sequence[int]], Optional[int]],
    progress: Optional[ProgressFn],
    tokens_before: int,
    total_tokens: int,
) -> Tuple[List[int], int, int]:
    """Recover one segment's bits from the text beginning at ``start``.

    ``start`` must be a prompt occurrence. Reading stops as soon as the
    segment's declared length is reached, so the tokens consumed are exactly
    the segment the encoder wrote. Returns ``(bits, char_end, tokens)`` where
    ``char_end`` is the offset just past this segment -- where the next prompt,
    if any, begins.
    """
    ids = _token_ids(tokenizer, encoded_text[start:])
    if ids[: len(prompt_ids)] != list(prompt_ids):
        raise ValueError(
            "the encoded text does not tokenize to the prompt_prefix followed by "
            "the message; text preceding the prompt may have merged into it"
        )
    body = ids[len(prompt_ids):]

    stepper = _Stepper(model, device, prompt_ids)
    context = list(prompt_ids)
    writer = ArithmeticEncoder()
    needed: Optional[int] = None
    consumed = 0

    for token in body:
        if needed is not None and len(writer.bits) >= needed:
            break
        _check_limit(len(context) + 1, limit)

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
        consumed += 1

        if needed is None:
            needed = declared_length(writer.bits)
        if progress is not None:
            progress(min(tokens_before + consumed, total_tokens), total_tokens)

    consumed_text = tokenizer.decode(
        list(prompt_ids) + body[:consumed], skip_special_tokens=True
    )
    return writer.bits, start + len(consumed_text), consumed


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
    so text after the message is ignored, as is text before the prompt.

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
    limit = _resolve_context_limit(model, max_context_length)
    prompt_ids = _token_ids(tokenizer, prompt_prefix)

    first = encoded_text.find(prompt_prefix)
    if first == -1:
        raise ValueError("prompt_prefix not found in encoded_text")
    # A rough denominator for the progress bar: tokens after the first prompt.
    total_tokens = max(len(_token_ids(tokenizer, encoded_text[first:])) - len(prompt_ids), 1)

    if key.version == CODEC_VERSION:
        bits, _, _ = _decode_one_segment(
            encoded_text, first, prompt_ids, model, tokenizer, device, top_k,
            temperature, limit, _declared_length, progress, 0, total_tokens,
        )
        return _unframe_payload(bits)

    # v2: one re-anchored segment per chunk. The first chunk's header gives the
    # total count; each subsequent segment is found by re-anchoring on the next
    # prompt occurrence, starting just past the previous segment's tokens.
    recovered: dict = {}
    count: Optional[int] = None
    search_from = 0
    consumed = 0
    while count is None or len(recovered) < count:
        start = encoded_text.find(prompt_prefix, search_from)
        if start == -1:
            if count is None:
                raise ValueError("prompt_prefix not found in encoded_text")
            raise ValueError(
                f"expected {count} segments but found {len(recovered)}; the "
                "message is truncated or a segment boundary was lost"
            )
        bits, char_end, seg_tokens = _decode_one_segment(
            encoded_text, start, prompt_ids, model, tokenizer, device, top_k,
            temperature, limit, _declared_chunk_length, progress, consumed, total_tokens,
        )
        index, this_count, chunk = _unframe_chunk(bits)
        if count is None:
            if not 1 <= this_count <= MAX_SEGMENTS:
                raise ValueError(
                    f"decoded segment count ({this_count}) is implausible; the "
                    "key, prompt or model do not match the encoder"
                )
            count = this_count
        elif this_count != count:
            raise ValueError(
                "segments disagree on the total count; the key, prompt or model "
                "do not match the encoder"
            )
        if not 0 <= index < count or index in recovered:
            raise ValueError(
                f"segment index {index} is out of range or repeated; the key, "
                "prompt or model do not match the encoder"
            )
        recovered[index] = chunk
        consumed += seg_tokens
        search_from = char_end

    stored = b"".join(recovered[i] for i in range(count))
    if key.compression == "zlib":
        try:
            manifest = zlib.decompress(stored)
        except zlib.error as exc:
            raise ValueError(
                "failed to decompress the recovered payload; the key, prompt or "
                "model do not match the encoder"
            ) from exc
    elif key.compression is None:
        manifest = stored
    else:
        raise ValueError(f"unsupported compression: {key.compression!r}")
    # The manifest's length + crc are over the original data, so this is what
    # catches a wrong compression flag or a dropped segment.
    return _unframe_payload(to_bits(manifest))


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
