import dataclasses
import re
from typing import Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


HEADER_PATTERN = re.compile(
    r"^\[LLM-CODEC v1; base=(\d+); length=(\d+)(?:; top_k=([^\];]+))?\]\s*(.*)$",
    re.DOTALL,
)


@dataclasses.dataclass
class CodecConfig:
    model_name_or_path: str
    device: str
    base: int
    prompt_prefix: str
    max_new_tokens: int
    max_context_length: Optional[int] = None
    top_k: Optional[int] = None


def set_deterministic(seed: int = 0) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_model_and_tokenizer(model_name_or_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
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


def base_digits_to_bytes(digits: Iterable[int], base: int, length_bytes: int) -> bytes:
    if base < 2:
        raise ValueError("base must be >= 2")
    n = 0
    for d in digits:
        if d < 0 or d >= base:
            raise ValueError(f"digit {d} out of range for base {base}")
        n = n * base + d
    data = n.to_bytes(length_bytes, byteorder="big", signed=False)
    if len(data) > length_bytes:
        data = data[-length_bytes:]
    elif len(data) < length_bytes:
        data = b"\x00" * (length_bytes - len(data)) + data
    return data


def _check_context_length(
    input_ids: torch.Tensor, max_context_length: Optional[int]
) -> None:
    if max_context_length is not None and input_ids.shape[1] > max_context_length:
        raise ValueError(
            f"Context length {input_ids.shape[1]} exceeds max_context_length={max_context_length}"
        )


def _prepare_sorted_indices(logits: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
    sorted_indices = torch.argsort(logits, descending=True)
    if top_k is not None:
        sorted_indices = sorted_indices[:top_k]
    return sorted_indices


def encode_data_to_text(
    data: bytes,
    cfg: CodecConfig,
    model,
    tokenizer,
):
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

        next_token_id = sorted_indices[digit].unsqueeze(0).unsqueeze(0).to(cfg.device)
        input_ids = torch.cat([input_ids, next_token_id], dim=1)
        _check_context_length(input_ids, cfg.max_context_length)

        digit_idx += 1

    full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    header = (
        f"[LLM-CODEC v1; base={cfg.base}; length={len(data)}; "
        f"top_k={cfg.top_k if cfg.top_k is not None else 'none'}]"
    )
    return header + "\n" + full_text


def _parse_header(encoded_text: str) -> Tuple[int, int, Optional[int], str]:
    match = HEADER_PATTERN.match(encoded_text.strip())
    if not match:
        raise ValueError("Encoded text missing or malformed LLM-CODEC header")
    base = int(match.group(1))
    length_bytes = int(match.group(2))
    raw_top_k = match.group(3)
    top_k = None if raw_top_k is None or raw_top_k.lower() == "none" else int(raw_top_k)
    body_text = match.group(4)
    return base, length_bytes, top_k, body_text


def decode_text_to_data(
    encoded_text: str,
    prompt_prefix: str,
    model,
    tokenizer,
    device: str,
    top_k_override: Optional[int] = None,
    max_context_length: Optional[int] = None,
) -> bytes:
    base, length_bytes, header_top_k, body_text = _parse_header(encoded_text)
    effective_top_k = top_k_override if top_k_override is not None else header_top_k

    body_ids = tokenizer(body_text, return_tensors="pt").input_ids.to(device)
    prompt_ids = tokenizer(prompt_prefix, return_tensors="pt").input_ids.to(device)

    if not body_text.startswith(prompt_prefix):
        raise ValueError("Body text does not start with provided prompt_prefix")

    generated_ids = body_ids[:, prompt_ids.shape[1] :]
    input_ids = prompt_ids.clone()
    digits: List[int] = []

    for next_token_id in generated_ids[0]:
        _check_context_length(input_ids, max_context_length)
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :].squeeze(0).cpu()
        sorted_indices = _prepare_sorted_indices(logits, effective_top_k)
        positions = (sorted_indices == int(next_token_id)).nonzero(as_tuple=True)[0]
        if len(positions) == 0:
            raise ValueError(
                "Generated token not found in sorted indices; mismatched parameters?"
            )
        digit = int(positions.item())
        digits.append(digit)
        input_ids = torch.cat([input_ids, next_token_id.view(1, 1).to(device)], dim=1)

    return base_digits_to_bytes(digits, base, length_bytes)


__all__ = [
    "HEADER_PATTERN",
    "CodecConfig",
    "set_deterministic",
    "load_model_and_tokenizer",
    "bytes_to_base_digits",
    "base_digits_to_bytes",
    "encode_data_to_text",
    "decode_text_to_data",
]
