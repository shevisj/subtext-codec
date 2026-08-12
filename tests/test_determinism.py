"""Device-level reproducibility.

The codec's correctness reduces to one property: two independent passes over
the same tokens, in the same environment, must produce the same candidate band
at every step. Everything downstream of the forward pass -- the sort, the
nucleus cut, the stability filter -- runs on CPU in float32 regardless of where
the model lives, so this is the only place a device can change the answer.

These parametrize over every device present, so they cover CPU everywhere and
additionally cover CUDA on a machine that has it. Run them on the hardware you
intend to encode with before trusting a message to it.
"""

import functools

import pytest
import torch

import subtext_codec
from subtext_codec import CodecConfig, decode_text_to_data, encode_data_to_text
from subtext_codec.codec import _step_distribution, _Stepper

MODEL = "sshleifer/tiny-gpt2"
PROMPT = "Once upon a time, "

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

# bf16/fp16 halve the mantissa, which leaves less margin between adjacent
# logits; they are worth covering precisely because they are the tight case.
DTYPES = ["float32"] + (["bfloat16", "float16"] if torch.cuda.is_available() else [])


@functools.lru_cache(maxsize=None)
def load(device: str, dtype: str):
    try:
        return subtext_codec.load_model_and_tokenizer(MODEL, device, dtype)
    except Exception as exc:  # noqa: BLE001 - offline or hub outage
        pytest.skip(f"{MODEL} unavailable on {device}/{dtype} ({exc})")


def step_through(model, tokenizer, device, ids, count):
    """Drive a fresh stepper over `ids`, collecting logits and coding tables."""
    stepper = _Stepper(model, device, ids)
    context = list(ids)
    logits, tables = [], []
    for _ in range(count):
        logits.append(stepper.logits.clone())
        alphabet, cum = _step_distribution(
            stepper.logits, context, tokenizer, top_k=32, temperature=1.5
        )
        tables.append((tuple(alphabet), tuple(cum)))
        token = alphabet[0]
        context.append(token)
        stepper.advance(token)
    return logits, tables


@pytest.mark.slow
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_logits_are_bit_identical_across_runs(device, dtype):
    """Two independent passes must agree exactly, not merely closely."""
    tokenizer, model = load(device, dtype)
    ids = tokenizer(PROMPT)["input_ids"]

    first, _ = step_through(model, tokenizer, device, ids, count=12)
    second, _ = step_through(model, tokenizer, device, ids, count=12)

    for step, (a, b) in enumerate(zip(first, second)):
        assert torch.equal(a, b), f"logits diverged at step {step} on {device}/{dtype}"


@pytest.mark.slow
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_coding_tables_are_identical_across_runs(device, dtype):
    """The frequency table is what encode and decode must agree on exactly.

    A single differing frequency changes which symbol the arithmetic coder
    selects, and every subsequent bit with it.
    """
    tokenizer, model = load(device, dtype)
    ids = tokenizer(PROMPT)["input_ids"]

    _, first = step_through(model, tokenizer, device, ids, count=12)
    _, second = step_through(model, tokenizer, device, ids, count=12)

    assert first == second


@pytest.mark.slow
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_round_trip_on_device(device, dtype):
    tokenizer, model = load(device, dtype)
    cfg = CodecConfig(
        model_name_or_path=MODEL,
        device=device,
        prompt_prefix=PROMPT,
        top_k=32,
        temperature=1.5,
        torch_dtype=dtype,
        store_model_in_key=True,
    )
    payload = b"\x00\x00device round trip\xff"

    text, key = encode_data_to_text(payload, cfg, model, tokenizer)
    decoded = decode_text_to_data(
        text,
        key=key,
        prompt_prefix=PROMPT,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    assert decoded == payload


@pytest.mark.slow
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rolling_round_trip_on_device(device, dtype):
    """A payload that outgrows the window resets its context mid-stream. Encode
    and decode must reset identically, on every device and dtype."""
    tokenizer, model = load(device, dtype)
    cfg = CodecConfig(
        model_name_or_path=MODEL,
        device=device,
        prompt_prefix=PROMPT,
        top_k=32,
        temperature=1.5,
        torch_dtype=dtype,
        max_context_length=32,  # small, so the payload forces resets
        store_model_in_key=True,
    )
    payload = b"\x00\x00rolling across a reset\xff"

    text, key = encode_data_to_text(payload, cfg, model, tokenizer)
    assert key.window == 32  # the message rolled and recorded its window
    decoded = decode_text_to_data(
        text, key=key, prompt_prefix=PROMPT, model=model,
        tokenizer=tokenizer, device=device, max_context_length=32,
    )
    assert decoded == payload


@pytest.mark.slow
@pytest.mark.parametrize("dtype", DTYPES)
def test_cached_stepping_matches_a_full_forward(dtype):
    """The KV cache must not change what the model predicts.

    Encode and decode both step through the cache so they agree with each other
    regardless; this checks the cache is not also silently shifting the logits
    away from an uncached pass, which would make the band depend on how the
    sequence was fed rather than on the sequence itself.
    """
    device = DEVICES[-1]  # prefer CUDA when present
    tokenizer, model = load(device, dtype)
    ids = tokenizer(PROMPT)["input_ids"]

    with torch.no_grad():
        full = model(
            input_ids=torch.tensor([ids], device=device)
        ).logits[0, -1, :].float().cpu()

    stepper = _Stepper(model, device, ids[:-1])
    stepper.advance(ids[-1])
    stepped = stepper.logits.float().cpu()

    # bf16/fp16 accumulate differently between a batched prefill and a
    # single-token step, so this is a closeness check, not an equality one.
    tolerance = 1e-4 if dtype == "float32" else 5e-2
    assert torch.allclose(full, stepped, atol=tolerance), (
        f"cached and uncached logits diverge on {device}/{dtype}: "
        f"max delta {(full - stepped).abs().max().item():.3e}"
    )


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cross_device_decode_is_never_silently_wrong():
    """Crossing devices is unsupported, but it must not corrupt quietly.

    Whether a CPU decode of a GPU encode succeeds depends on the model and
    dtype -- small fp32 models often do match bit for bit, larger bf16 ones
    will not. Either outcome is acceptable; returning plausible wrong bytes is
    not. That is what the framing check exists to prevent, and what this pins.
    """
    tokenizer, gpu_model = load("cuda", "float32")
    _, cpu_model = load("cpu", "float32")

    cfg = CodecConfig(
        model_name_or_path=MODEL,
        device="cuda",
        prompt_prefix=PROMPT,
        top_k=32,
        temperature=1.5,
        torch_dtype="float32",
    )
    payload = b"cross device payload that is long enough to diverge"
    text, key = encode_data_to_text(payload, cfg, gpu_model, tokenizer)

    try:
        decoded = decode_text_to_data(
            text,
            key=key,
            prompt_prefix=PROMPT,
            model=cpu_model,
            tokenizer=tokenizer,
            device="cpu",
        )
    except ValueError:
        return  # detected, which is the required behaviour
    assert decoded == payload, "cross-device decode returned wrong bytes silently"
