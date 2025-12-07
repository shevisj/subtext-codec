import dataclasses
import json
from typing import Iterable, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclasses.dataclass
class CodecConfig:
    model_name_or_path: str
    device: str
    base: int
    prompt_prefix: str
    max_new_tokens: int
    max_context_length: Optional[int] = None
    top_k: Optional[int] = None
    torch_dtype: Optional[str] = None
    store_model_in_key: bool = False


@dataclasses.dataclass
class CodecKey:
    base: int
    top_k: Optional[int]
    prompt_prefix: Optional[str] = None
    model_name_or_path: Optional[str] = None
    device: Optional[str] = None
    torch_dtype: Optional[str] = None
    version: str = "v1"

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "base": self.base,
            "top_k": self.top_k,
            "prompt_prefix": self.prompt_prefix,
            "model_name_or_path": self.model_name_or_path,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CodecKey":
        version = data.get("version", "v1")
        if version != "v1":
            raise ValueError(f"Unsupported codec key version: {version}")
        base = int(data["base"])
        if base < 2:
            raise ValueError("base must be >= 2")
        top_k_raw = data.get("top_k")
        if isinstance(top_k_raw, str) and top_k_raw.lower() == "none":
            top_k = None
        else:
            top_k = None if top_k_raw is None else int(top_k_raw)
        prompt_prefix = data.get("prompt_prefix")
        model_name_or_path = data.get("model_name_or_path")
        device = data.get("device")
        torch_dtype = data.get("torch_dtype")
        # Ignore any legacy termination metadata in existing key files
        return cls(
            base=base,
            top_k=top_k,
            prompt_prefix=prompt_prefix,
            model_name_or_path=model_name_or_path,
            device=device,
            torch_dtype=torch_dtype,
            version=version,
        )


def save_codec_key(key: CodecKey, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(key.to_dict(), f, indent=2)
        f.write("\n")


def load_codec_key(path: str) -> CodecKey:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return CodecKey.from_dict(raw)


def set_deterministic(seed: int = 0) -> None:
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
):
    resolved_dtype = _parse_torch_dtype(torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {}
    if resolved_dtype is not None:
        model_kwargs["dtype"] = resolved_dtype
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.to(device)
    model.eval()
    return tokenizer, model


def bytes_to_base_digits(data: bytes, base: int) -> List[int]:
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


def base_digits_to_bytes(digits: Iterable[int], base: int) -> bytes:
    if base < 2:
        raise ValueError("base must be >= 2")
    n = 0
    for d in digits:
        if d < 0 or d >= base:
            raise ValueError(f"digit {d} out of range for base {base}")
        n = n * base + d
    if n == 0:
        return b"" if len(digits) == 0 else b"\x00"
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


def encode_data_to_text(
    data: bytes,
    cfg: CodecConfig,
    model,
    tokenizer,
) -> Tuple[str, CodecKey]:
    digits = bytes_to_base_digits(data, cfg.base)
    prompt_ids = tokenizer(cfg.prompt_prefix, return_tensors="pt").input_ids.to(
        cfg.device
    )
    input_ids = prompt_ids
    _check_context_length(input_ids, cfg.max_context_length)
    digit_idx = 0

    while digit_idx < len(digits):
        if input_ids.shape[1] - prompt_ids.shape[1] >= cfg.max_new_tokens:
            raise ValueError(
                f"Reached max_new_tokens={cfg.max_new_tokens} before consuming all digits "
                f"({digit_idx}/{len(digits)})"
            )

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :].squeeze(0).cpu()

        sorted_indices = _prepare_sorted_indices(logits, cfg.top_k)
        if cfg.base > len(sorted_indices):
            raise ValueError(
                f"base {cfg.base} exceeds available candidate tokens ({len(sorted_indices)}); "
                "increase top_k or lower base"
            )

        digit = digits[digit_idx]
        if digit >= cfg.base or digit < 0:
            raise ValueError(f"Digit {digit} out of range for base {cfg.base}")

        # Choose the token whose rank matches the next digit
        next_token_id = sorted_indices[digit].unsqueeze(0).unsqueeze(0).to(cfg.device)
        input_ids = torch.cat([input_ids, next_token_id], dim=1)
        _check_context_length(input_ids, cfg.max_context_length)

        digit_idx += 1

    # Append a single terminator token using the first rank outside the base band
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, -1, :].squeeze(0).cpu()

    sorted_indices = _prepare_sorted_indices(logits, cfg.top_k)
    if cfg.base >= len(sorted_indices):
        raise ValueError(
            f"base {cfg.base} requires at least {cfg.base + 1} candidate tokens for termination; "
            "increase top_k or lower base"
        )
    terminator_token_id = (
        sorted_indices[cfg.base].unsqueeze(0).unsqueeze(0).to(cfg.device)
    )
    input_ids = torch.cat([input_ids, terminator_token_id], dim=1)
    _check_context_length(input_ids, cfg.max_context_length)

    full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    key = CodecKey(
        base=cfg.base,
        top_k=cfg.top_k,
        prompt_prefix=cfg.prompt_prefix,
        model_name_or_path=cfg.model_name_or_path if cfg.store_model_in_key else None,
        device=cfg.device,
        torch_dtype=cfg.torch_dtype,
    )
    return full_text, key


def decode_text_to_data(
    encoded_text: str,
    key: CodecKey,
    prompt_prefix: str,
    model,
    tokenizer,
    device: str,
    max_context_length: Optional[int] = None,
) -> bytes:
    if key.prompt_prefix is not None and key.prompt_prefix != prompt_prefix:
        raise ValueError("Prompt prefix does not match codec key")

    # Allow for extra text before the prompt prefix
    prefix_index = encoded_text.find(prompt_prefix)
    if prefix_index == -1:
        raise ValueError("prompt_prefix not found in encoded_text")
    encoded_text = encoded_text[prefix_index:]

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
            print(next_token_id)
            print(sorted_indices)
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

    return base_digits_to_bytes(digits, key.base)


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
]
