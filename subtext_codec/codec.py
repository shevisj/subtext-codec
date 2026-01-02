import dataclasses
import json
from typing import Iterable, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _validate_top_p(top_p: Optional[float]) -> float:
    if top_p is None:
        raise ValueError("top_p is required for variable-base encoding/decoding")
    try:
        value = float(top_p)
    except (TypeError, ValueError) as exc:
        raise ValueError("top_p must be a float between 0 and 1") from exc
    if value <= 0 or value > 1:
        raise ValueError("top_p must be in the interval (0, 1]")
    return value


@dataclasses.dataclass
class CodecConfig:
    """Configuration for encoding binary data into steganographic text.

    This dataclass holds all parameters needed to encode data using the codec.
    The encoder uses these settings to control token selection during text generation.

    Attributes:
        model_name_or_path: HuggingFace model identifier or local path to the model.
        device: Computation device ("cuda", "cpu", or specific device like "cuda:0").
        prompt_prefix: Text prompt that precedes the generated steganographic content.
            Choose a prompt that provides good context for natural text generation.
        max_context_length: Optional maximum context window size. If None, uses the
            model's default. Raises ValueError if exceeded during encoding.
        top_k: Maximum number of candidate tokens considered at each generation step.
            Higher values increase encoding capacity but may reduce text naturalness.
            If None, all vocabulary tokens are candidates (limited by top_p).
        top_p: Nucleus sampling threshold (0.0 to 1.0]. Only tokens within this
            cumulative probability mass are considered. Lower values produce more
            focused/natural text but reduce encoding capacity per token. Default: 0.9.
        torch_dtype: Model precision ("float16", "bf16", "float32", or "auto").
            Use "bf16" for bfloat16 on supported hardware. If None, uses model default.
        store_model_in_key: If True, saves model_name_or_path in the codec key for
            easier decoding. Set to False if you want to keep model choice private.
    """

    model_name_or_path: str
    device: str
    prompt_prefix: str
    max_context_length: Optional[int] = None
    top_k: Optional[int] = None
    top_p: float = 0.9
    torch_dtype: Optional[str] = None
    store_model_in_key: bool = False


@dataclasses.dataclass
class CodecKey:
    """Metadata required to decode steganographic text back to original bytes.

    The codec key is generated during encoding and must be provided (along with the
    same model and prompt) to decode the hidden data. Treat this like a password -
    without it, the message cannot be recovered.

    The key supports two versions:
        - v1 (legacy): Fixed-base encoding with a constant radix stored in `base`
        - v2 (current): Dynamic mixed-radix encoding using `top_p` for adaptive bases

    Attributes:
        top_k: Maximum token candidates used during encoding. Must match decoding.
        top_p: Nucleus sampling threshold for v2 keys. Required for v2, None for v1.
        prompt_prefix: The prompt that preceded the encoded text. Used for validation.
        model_name_or_path: HuggingFace model identifier (if stored during encoding).
        device: Computation device used during encoding.
        torch_dtype: Model precision used during encoding.
        version: Codec version ("v1" for fixed-base, "v2" for dynamic mixed-radix).
        base: Fixed radix for v1 keys only. Must be >= 2.
        payload_length: Exact byte count of the original payload. Required to
            correctly reconstruct payloads with leading zero bytes.

    Methods:
        to_dict: Serialize the key to a JSON-compatible dictionary.
        from_dict: Deserialize a key from a dictionary (classmethod).
    """

    top_k: Optional[int]
    top_p: Optional[float] = None
    prompt_prefix: Optional[str] = None
    model_name_or_path: Optional[str] = None
    device: Optional[str] = None
    torch_dtype: Optional[str] = None
    version: str = "v2"
    base: Optional[int] = None  # legacy fixed-base keys
    payload_length: Optional[int] = None  # exact byte length for proper reconstruction

    def to_dict(self) -> dict:
        data = {
            "version": self.version,
            "top_k": self.top_k,
            "prompt_prefix": self.prompt_prefix,
            "model_name_or_path": self.model_name_or_path,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
        }
        if self.version == "v1":
            if self.base is None:
                raise ValueError("base is required for v1 codec keys")
            data["base"] = self.base
        else:
            if self.top_p is None:
                raise ValueError("top_p is required for v2 codec keys")
            data["top_p"] = self.top_p
        if self.payload_length is not None:
            data["payload_length"] = self.payload_length
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "CodecKey":
        version = data.get("version", "v1")
        top_k_raw = data.get("top_k")
        if isinstance(top_k_raw, str) and top_k_raw.lower() == "none":
            top_k = None
        else:
            top_k = None if top_k_raw is None else int(top_k_raw)
        prompt_prefix = data.get("prompt_prefix")
        model_name_or_path = data.get("model_name_or_path")
        device = data.get("device")
        torch_dtype = data.get("torch_dtype")
        payload_length_raw = data.get("payload_length")
        payload_length = None if payload_length_raw is None else int(payload_length_raw)

        if version == "v1":
            base = int(data["base"])
            if base < 2:
                raise ValueError("base must be >= 2")
            return cls(
                base=base,
                top_k=top_k,
                prompt_prefix=prompt_prefix,
                model_name_or_path=model_name_or_path,
                device=device,
                torch_dtype=torch_dtype,
                version=version,
                payload_length=payload_length,
            )

        if version != "v2":
            raise ValueError(f"Unsupported codec key version: {version}")

        top_p_raw = data.get("top_p")
        if top_p_raw is None:
            raise ValueError("top_p missing from codec key (v2 requires it)")
        top_p = _validate_top_p(float(top_p_raw))

        return cls(
            top_p=top_p,
            top_k=top_k,
            prompt_prefix=prompt_prefix,
            model_name_or_path=model_name_or_path,
            device=device,
            torch_dtype=torch_dtype,
            version=version,
            payload_length=payload_length,
        )


def save_codec_key(key: CodecKey, path: str) -> None:
    """Save a codec key to a JSON file.

    The key file is essential for decoding - share it securely with the intended
    recipient along with the encoded message.

    Args:
        key: The CodecKey to save.
        path: File path where the key will be written (overwrites if exists).

    Example:
        >>> key = encode_data_to_text(data, config, model, tokenizer)[1]
        >>> save_codec_key(key, "my_key.json")
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(key.to_dict(), f, indent=2)
        f.write("\n")


def load_codec_key(path: str) -> CodecKey:
    """Load a codec key from a JSON file.

    The loaded key can be used with decode_text_to_data() to recover the original
    payload from steganographic text.

    Args:
        path: Path to the JSON key file.

    Returns:
        A CodecKey instance with parameters loaded from the file.

    Raises:
        FileNotFoundError: If the key file doesn't exist.
        ValueError: If the key file contains invalid or missing required fields.

    Example:
        >>> key = load_codec_key("my_key.json")
        >>> decoded = decode_text_to_data(text, key, prompt, model, tokenizer, device)
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return CodecKey.from_dict(raw)


def set_deterministic(seed: int = 0) -> None:
    """Configure PyTorch for deterministic operation.

    Encoding and decoding must produce identical results given the same inputs.
    This function seeds random number generators and configures PyTorch/cuDNN
    to use deterministic algorithms.

    Note:
        For full determinism with CUDA, also set the environment variable
        CUBLAS_WORKSPACE_CONFIG=:4096:8 before importing torch.

    Args:
        seed: Random seed for reproducibility. Default is 0.

    Example:
        >>> set_deterministic(42)
        >>> # Now encoding/decoding will be reproducible
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _parse_torch_dtype(dtype: Optional[str]) -> Optional[Union[str, torch.dtype]]:
    if dtype is None:
        return None
    lowered = dtype.lower()
    if lowered == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if lowered not in mapping:
        raise ValueError(
            "torch-dtype must be one of: auto, float16/fp16/half, "
            "bfloat16/bf16, float32/fp32"
        )
    return mapping[lowered]


def load_model_and_tokenizer(
    model_name_or_path: str,
    device: str,
    torch_dtype: Optional[str] = None,
    seed: int = 0,
):
    """Load a HuggingFace causal language model and tokenizer for encoding/decoding.

    This function loads the model in evaluation mode with deterministic settings.
    The same model (including version and precision) must be used for both
    encoding and decoding to ensure correct round-trip behavior.

    Args:
        model_name_or_path: HuggingFace model identifier (e.g., "meta-llama/Llama-3.1-8B")
            or path to a local model directory.
        device: Computation device ("cuda", "cpu", or specific like "cuda:0").
        torch_dtype: Model precision. Options: "float16"/"fp16", "bfloat16"/"bf16",
            "float32"/"fp32", or "auto". If None, uses the model's default dtype.
        seed: Random seed for deterministic behavior. Default is 0.

    Returns:
        A tuple of (tokenizer, model) ready for use with encode/decode functions.

    Raises:
        ValueError: If torch_dtype is not a recognized precision string.

    Note:
        Loading large models requires significant memory. For Llama-3.1-8B with
        bfloat16, expect ~16GB GPU memory usage.

    Example:
        >>> tokenizer, model = load_model_and_tokenizer(
        ...     "meta-llama/Llama-3.1-8B-Instruct",
        ...     device="cuda",
        ...     torch_dtype="bf16"
        ... )
    """
    set_deterministic(seed)
    resolved_dtype = _parse_torch_dtype(torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {}
    if resolved_dtype is not None:
        model_kwargs["torch_dtype"] = resolved_dtype
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.to(device)
    model.eval()
    return tokenizer, model


def bytes_to_base_digits(data: bytes, base: int) -> List[int]:
    """Convert bytes to a list of digits in the specified base.

    Interprets the bytes as a big-endian unsigned integer, then converts
    to the given base. Used internally for fixed-base (v1) encoding.

    Args:
        data: Binary data to convert.
        base: The radix for conversion. Must be >= 2.

    Returns:
        A list of digits in the specified base, most significant first.
        Returns an empty list for empty input.

    Raises:
        ValueError: If base < 2.

    Example:
        >>> bytes_to_base_digits(b"\\x01\\x00", base=256)
        [1, 0]
        >>> bytes_to_base_digits(b"\\xff", base=16)
        [15, 15]
    """
    if base < 2:
        raise ValueError("base must be >= 2")
    if len(data) == 0:
        return []
    n = int.from_bytes(data, byteorder="big", signed=False)
    digits: List[int] = []
    while n > 0:
        n, rem = divmod(n, base)
        digits.append(rem)
    digits.reverse()
    return digits


def base_digits_to_bytes(
    digits: Iterable[int], base: int, length: Optional[int] = None
) -> bytes:
    """Convert a list of base-N digits back to bytes.

    This is the inverse of bytes_to_base_digits(). Used internally for
    fixed-base (v1) decoding.

    Args:
        digits: Sequence of digits in the specified base, most significant first.
        base: The radix of the digit representation. Must be >= 2.
        length: If provided, pads/truncates the result to exactly this many bytes.
            Required to correctly reconstruct payloads with leading zero bytes.

    Returns:
        The reconstructed bytes as a big-endian unsigned integer.

    Raises:
        ValueError: If base < 2 or any digit is out of range [0, base).
        OverflowError: If length is specified but the value doesn't fit.

    Example:
        >>> base_digits_to_bytes([1, 0], base=256)
        b'\\x01\\x00'
        >>> base_digits_to_bytes([15, 15], base=16)
        b'\\xff'
        >>> base_digits_to_bytes([0], base=256, length=3)
        b'\\x00\\x00\\x00'
    """
    if base < 2:
        raise ValueError("base must be >= 2")
    digit_list = list(digits)
    n = 0
    for d in digit_list:
        if d < 0 or d >= base:
            raise ValueError(f"digit {d} out of range for base {base}")
        n = n * base + d
    if length is not None:
        if n == 0 and length == 0:
            return b""
        return n.to_bytes(length, byteorder="big", signed=False)
    if n == 0:
        return b"" if len(digit_list) == 0 else b"\x00"
    length_bytes = (n.bit_length() + 7) // 8
    return n.to_bytes(length_bytes, byteorder="big", signed=False)


def _check_context_length(
    input_ids: torch.Tensor, max_context_length: Optional[int]
) -> None:
    if max_context_length is not None and input_ids.shape[1] > max_context_length:
        raise ValueError(
            f"Context length {input_ids.shape[1]} exceeds max_context_length={max_context_length}"
        )


def _prepare_sorted_indices(logits: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
    # Order tokens by descending likelihood and optionally clip to the allowed band
    sorted_indices = torch.argsort(logits, descending=True)
    if top_k is not None:
        sorted_indices = sorted_indices[:top_k]
    return sorted_indices


def _select_rank_band(
    logits: torch.Tensor, top_k: Optional[int], top_p: float, min_base: int = 2
) -> Tuple[torch.Tensor, int]:
    sorted_indices = torch.argsort(logits, descending=True)
    if top_k is not None:
        sorted_indices = sorted_indices[:top_k]

    candidate_logits = logits[sorted_indices]
    if candidate_logits.numel() < min_base:
        raise ValueError(
            f"Not enough candidate tokens ({candidate_logits.numel()}) to satisfy min_base={min_base}; "
            "increase top_k or adjust prompt_prefix"
        )

    probs = torch.softmax(candidate_logits, dim=0)
    cumulative = torch.cumsum(probs, dim=0)
    cutoff = torch.searchsorted(
        cumulative, torch.tensor(top_p, device=cumulative.device), right=True
    )
    base = int(cutoff.item()) + 1
    base = max(min_base, min(base, candidate_logits.numel()))
    return sorted_indices, base


def mixed_radix_digits_to_bytes(
    digits: Iterable[int], bases: Iterable[int], length: Optional[int] = None
) -> bytes:
    """Convert mixed-radix digits back to bytes.

    In mixed-radix (variable-base) encoding, each position can have a different
    base. This is used in v2 encoding where the base varies dynamically based
    on the model's confidence at each token position.

    The conversion treats the digit sequence as a mixed-radix number and
    reconstructs the original integer value, then converts to bytes.

    Args:
        digits: Sequence of digits, one per position. Each digit[i] must be
            in range [0, bases[i]).
        bases: Sequence of bases corresponding to each digit position.
            All bases must be >= 2.
        length: If provided, pads/truncates the result to exactly this many bytes.
            Required to correctly reconstruct payloads with leading zero bytes.

    Returns:
        The reconstructed bytes as a big-endian unsigned integer.

    Raises:
        ValueError: If digits and bases have different lengths, any base < 2,
            or any digit is out of range for its corresponding base.
        OverflowError: If length is specified but the value doesn't fit.

    Example:
        >>> mixed_radix_digits_to_bytes([1, 2, 3], [4, 5, 6])  # 1*(5*6) + 2*6 + 3
        b'\\x2d'  # 45 in decimal
    """
    digit_list = list(digits)
    base_list = list(bases)
    if len(digit_list) != len(base_list):
        raise ValueError("digits and bases must have the same length")
    n = 0
    for digit, base in zip(reversed(digit_list), reversed(base_list)):
        if base < 2:
            raise ValueError("All bases must be >= 2")
        if digit < 0 or digit >= base:
            raise ValueError(f"digit {digit} out of range for base {base}")
        n = n * base + digit
    if length is not None:
        if n == 0 and length == 0:
            return b""
        return n.to_bytes(length, byteorder="big", signed=False)
    if n == 0:
        return b"" if len(digit_list) == 0 else b"\x00"
    length_bytes = (n.bit_length() + 7) // 8
    return n.to_bytes(length_bytes, byteorder="big", signed=False)


def _encode_variable_base_digits(
    data: bytes,
    cfg: CodecConfig,
    model,
    tokenizer,
) -> Tuple[torch.Tensor, CodecKey]:
    try:
        vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    except TypeError as exc:
        raise ValueError("tokenizer must provide vocab_size or __len__") from exc
    if cfg.top_k is not None:
        if cfg.top_k < 2:
            raise ValueError("top_k must be >= 2 for variable-base encoding")
        max_base = min(cfg.top_k, vocab_size)
    else:
        max_base = vocab_size
    if max_base < 2:
        raise ValueError("tokenizer must have at least 2 tokens in its vocabulary")

    payload_bits = len(data) * 8

    prompt_ids = tokenizer(cfg.prompt_prefix, return_tensors="pt").input_ids.to(
        cfg.device
    )
    input_ids = prompt_ids
    _check_context_length(input_ids, cfg.max_context_length)

    remaining = int.from_bytes(data, byteorder="big", signed=False)
    step = 0

    while remaining > 0:
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :].squeeze(0).cpu()

        sorted_indices, base = _select_rank_band(logits, cfg.top_k, cfg.top_p)
        digit = remaining % base
        remaining //= base

        next_token_id = sorted_indices[digit].unsqueeze(0).unsqueeze(0).to(cfg.device)
        input_ids = torch.cat([input_ids, next_token_id], dim=1)
        _check_context_length(input_ids, cfg.max_context_length)
        step += 1

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, -1, :].squeeze(0).cpu()

    sorted_indices, base = _select_rank_band(logits, cfg.top_k, cfg.top_p)
    if base >= len(sorted_indices):
        raise ValueError(
            "Dynamic base exhausted the available candidate tokens; "
            "lower top_p or raise top_k to leave room for a terminator"
        )
    terminator_token_id = sorted_indices[base].unsqueeze(0).unsqueeze(0).to(cfg.device)
    input_ids = torch.cat([input_ids, terminator_token_id], dim=1)
    _check_context_length(input_ids, cfg.max_context_length)

    key = CodecKey(
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        prompt_prefix=cfg.prompt_prefix,
        model_name_or_path=cfg.model_name_or_path if cfg.store_model_in_key else None,
        device=cfg.device,
        torch_dtype=cfg.torch_dtype,
        version="v2",
        payload_length=len(data),
    )
    return input_ids, key


def encode_data_to_text(
    data: bytes,
    cfg: CodecConfig,
    model,
    tokenizer,
) -> Tuple[str, CodecKey]:
    """Encode binary data into steganographic text.

    This is the main encoding function. It takes arbitrary binary data and
    produces natural-looking text that hides the data in token selection patterns.
    The returned key is required to decode the message later.

    The encoding process:
    1. Converts the payload to a big-endian integer
    2. Generates text token-by-token, selecting each token's rank to encode
       a digit of the payload in a variable base determined by top_p/top_k
    3. Appends a terminator token (at rank = base) to mark the end
    4. Returns the full text including the prompt prefix

    Args:
        data: Binary payload to hide in the generated text. Can be any bytes.
        cfg: CodecConfig with model, device, prompt, and encoding parameters.
        model: Pre-loaded HuggingFace causal language model (from load_model_and_tokenizer).
        tokenizer: Corresponding tokenizer for the model.

    Returns:
        A tuple of (encoded_text, key) where:
        - encoded_text: The steganographic text containing the hidden payload.
          Starts with cfg.prompt_prefix followed by generated content.
        - key: A CodecKey required for decoding. Save this securely!

    Raises:
        ValueError: If top_p is invalid, context length is exceeded, or the
            model can't provide enough token candidates for encoding.

    Example:
        >>> config = CodecConfig(
        ...     model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
        ...     device="cuda",
        ...     prompt_prefix="Here's a story: ",
        ...     top_k=16,
        ...     top_p=0.9
        ... )
        >>> text, key = encode_data_to_text(b"secret", config, model, tokenizer)
        >>> save_codec_key(key, "key.json")
    """
    cfg_top_p = _validate_top_p(cfg.top_p)
    cfg = dataclasses.replace(cfg, top_p=cfg_top_p)
    input_ids, key = _encode_variable_base_digits(data, cfg, model, tokenizer)
    full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return full_text, key


def _decode_fixed_base(
    encoded_text: str,
    key: CodecKey,
    prompt_prefix: str,
    model,
    tokenizer,
    device: str,
    max_context_length: Optional[int],
) -> bytes:
    if key.base is None:
        raise ValueError("codec key missing base for v1 decoding")

    body_ids = tokenizer(encoded_text, return_tensors="pt").input_ids.to(device)
    prompt_ids = tokenizer(prompt_prefix, return_tensors="pt").input_ids.to(device)

    generated_ids = body_ids[:, prompt_ids.shape[1] :]
    input_ids = prompt_ids.clone()
    digits: List[int] = []
    terminated = False

    for next_token_id in generated_ids[0]:
        _check_context_length(input_ids, max_context_length)
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :].squeeze(0).cpu()
        sorted_indices = _prepare_sorted_indices(logits, key.top_k)
        positions = (sorted_indices == int(next_token_id)).nonzero(as_tuple=True)[0]
        if len(positions) == 0:
            raise ValueError(
                "Generated token not found in sorted indices; mismatched parameters?"
            )
        rank = int(positions.item())
        if rank >= key.base:
            terminated = True
            break
        digits.append(rank)
        input_ids = torch.cat([input_ids, next_token_id.view(1, 1).to(device)], dim=1)

    if not terminated:
        raise ValueError("Termination token not found in encoded text")

    return base_digits_to_bytes(digits, key.base, length=key.payload_length)


def _decode_variable_base(
    encoded_text: str,
    key: CodecKey,
    prompt_prefix: str,
    model,
    tokenizer,
    device: str,
    max_context_length: Optional[int],
) -> bytes:
    top_p = _validate_top_p(key.top_p)

    body_ids = tokenizer(encoded_text, return_tensors="pt").input_ids.to(device)
    prompt_ids = tokenizer(prompt_prefix, return_tensors="pt").input_ids.to(device)

    generated_ids = body_ids[:, prompt_ids.shape[1] :]
    input_ids = prompt_ids.clone()
    digits: List[int] = []
    bases: List[int] = []
    terminated = False

    for next_token_id in generated_ids[0]:
        _check_context_length(input_ids, max_context_length)
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :].squeeze(0).cpu()
        sorted_indices, base = _select_rank_band(logits, key.top_k, top_p)
        positions = (sorted_indices == int(next_token_id)).nonzero(as_tuple=True)[0]
        if len(positions) == 0:
            raise ValueError(
                "Generated token not found in sorted indices; mismatched parameters?"
            )
        rank = int(positions.item())
        if rank >= base:
            terminated = True
            break
        digits.append(rank)
        bases.append(base)
        input_ids = torch.cat([input_ids, next_token_id.view(1, 1).to(device)], dim=1)

    if not terminated:
        raise ValueError("Termination token not found in encoded text")

    return mixed_radix_digits_to_bytes(digits, bases, length=key.payload_length)


def decode_text_to_data(
    encoded_text: str,
    key: CodecKey,
    prompt_prefix: str,
    model,
    tokenizer,
    device: str,
    max_context_length: Optional[int] = None,
) -> bytes:
    """Decode steganographic text back to the original binary payload.

    This is the main decoding function. Given encoded text and the corresponding
    codec key, it recovers the exact original bytes that were hidden during encoding.

    The decoding process:
    1. Locates and strips the prompt prefix from the encoded text
    2. Re-runs the model to reconstruct token probability distributions
    3. Determines which rank was selected at each position to recover digits
    4. Stops when the terminator token (rank >= base) is encountered
    5. Converts the digit sequence back to bytes using payload_length from key

    The decoder is robust to trailing text after the encoded message, since it
    stops at the terminator token.

    Args:
        encoded_text: The steganographic text to decode. Must contain the prompt
            prefix followed by the encoded content.
        key: CodecKey from the original encoding (from encode_data_to_text or
            loaded via load_codec_key).
        prompt_prefix: The prompt prefix used during encoding. Must match exactly.
        model: Pre-loaded HuggingFace causal language model. Must be the same
            model (including version and dtype) used for encoding.
        tokenizer: Corresponding tokenizer for the model.
        device: Computation device ("cuda", "cpu", etc.).
        max_context_length: Optional context limit. If None, uses model default.

    Returns:
        The original binary payload, byte-for-byte identical to what was encoded.

    Raises:
        ValueError: If the prompt prefix doesn't match, isn't found in the text,
            the terminator token is missing, or parameters don't match encoding.

    Example:
        >>> key = load_codec_key("key.json")
        >>> decoded = decode_text_to_data(
        ...     encoded_text=message,
        ...     key=key,
        ...     prompt_prefix=key.prompt_prefix,
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     device="cuda"
        ... )
        >>> assert decoded == original_secret
    """
    if key.prompt_prefix is not None and key.prompt_prefix != prompt_prefix:
        raise ValueError("Prompt prefix does not match codec key")

    prefix_index = encoded_text.find(prompt_prefix)
    if prefix_index == -1:
        raise ValueError("prompt_prefix not found in encoded_text")
    encoded_text = encoded_text[prefix_index:]

    if key.version == "v1":
        return _decode_fixed_base(
            encoded_text,
            key=key,
            prompt_prefix=prompt_prefix,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_context_length=max_context_length,
        )

    return _decode_variable_base(
        encoded_text,
        key=key,
        prompt_prefix=prompt_prefix,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_context_length=max_context_length,
    )


__all__ = [
    "CodecConfig",
    "CodecKey",
    "set_deterministic",
    "load_model_and_tokenizer",
    "load_codec_key",
    "save_codec_key",
    "bytes_to_base_digits",
    "base_digits_to_bytes",
    "encode_data_to_text",
    "decode_text_to_data",
    "mixed_radix_digits_to_bytes",
]
