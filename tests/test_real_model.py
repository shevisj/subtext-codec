"""Round trips against a real tokenizer and model.

These exercise the thing the fakes cannot: an actual byte-level BPE vocabulary,
where writing tokens out and reading them back genuinely does not always return
the same ids. Marked slow because they download a checkpoint.
"""

import pytest
import torch

import subtext_codec
from subtext_codec import CodecConfig, decode_text_to_data, encode_data_to_text

MODEL = "sshleifer/tiny-gpt2"  # gpt2 tokenizer, ~2MB of random weights
PROMPT = "Once upon a time, "


@pytest.fixture(scope="module")
def real_components():
    try:
        return subtext_codec.load_model_and_tokenizer(MODEL, "cpu", "float32")
    except Exception as exc:  # noqa: BLE001 - offline, gated, or hub outage
        pytest.skip(f"{MODEL} unavailable ({exc})")


def make_config(**overrides) -> CodecConfig:
    params = dict(
        model_name_or_path=MODEL,
        device="cpu",
        prompt_prefix=PROMPT,
        top_k=32,
        temperature=1.5,
        store_model_in_key=True,
    )
    params.update(overrides)
    return CodecConfig(**params)


@pytest.mark.slow
@pytest.mark.parametrize(
    "payload",
    [b"", b"\x00\x00secret", b"attack at dawn", bytes(range(48))],
    ids=["empty", "leading-zeros", "text", "binary"],
)
def test_real_model_round_trip(payload, real_components):
    tokenizer, model = real_components
    cfg = make_config()

    text, key = encode_data_to_text(payload, cfg, model, tokenizer)
    assert text.startswith(PROMPT)

    decoded = decode_text_to_data(
        text,
        key=key,
        prompt_prefix=PROMPT,
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )
    assert decoded == payload


@pytest.mark.slow
def test_real_model_text_is_tokenizer_stable(real_components):
    """The property that made the pre-v3 codec corrupt messages at random.

    A byte-level BPE vocabulary will happily merge an emitted token into its
    neighbours; when that happens the decoder reads a different id stream than
    the encoder wrote. Every message this codec produces must survive being
    written out and read back.
    """
    tokenizer, model = real_components
    cfg = make_config()

    for trial in range(4):
        payload = bytes([trial]) * 24
        text, _ = encode_data_to_text(payload, cfg, model, tokenizer)
        ids = tokenizer(text)["input_ids"]
        assert tokenizer(tokenizer.decode(ids))["input_ids"] == ids


@pytest.mark.slow
def test_real_model_ignores_surrounding_noise(real_components):
    tokenizer, model = real_components
    cfg = make_config()
    payload = b"hidden in plain sight"

    text, key = encode_data_to_text(payload, cfg, model, tokenizer)
    noisy = "Forwarded message:\n\n" + text + "\n\n-- sent from my phone"

    decoded = decode_text_to_data(
        noisy,
        key=key,
        prompt_prefix=PROMPT,
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )
    assert decoded == payload


@pytest.mark.slow
def test_real_model_rejects_an_edited_message(real_components):
    """Editing the text must fail loudly rather than return plausible bytes."""
    tokenizer, model = real_components
    cfg = make_config()

    text, key = encode_data_to_text(b"do not tamper with this", cfg, model, tokenizer)
    body_start = len(PROMPT)
    tampered = text[:body_start] + " zzz" + text[body_start:]

    with pytest.raises(ValueError):
        decode_text_to_data(
            tampered,
            key=key,
            prompt_prefix=PROMPT,
            model=model,
            tokenizer=tokenizer,
            device="cpu",
        )


@pytest.mark.slow
def test_cached_and_uncached_prefill_agree(real_components):
    """The KV cache must not change the logits the codec reads.

    Encode and decode both step through the cache, so they agree with each
    other by construction; this checks the cache is not also silently changing
    what the model predicts.
    """
    tokenizer, model = real_components
    ids = tokenizer(PROMPT)["input_ids"]

    with torch.no_grad():
        full = model(input_ids=torch.tensor([ids])).logits[0, -1, :]

        cached = model(
            input_ids=torch.tensor([ids[:-1]]), use_cache=True, past_key_values=None
        )
        stepped = model(
            input_ids=torch.tensor([ids[-1:]]),
            past_key_values=cached.past_key_values,
            use_cache=True,
        ).logits[0, -1, :]

    assert torch.allclose(full, stepped, atol=1e-4)
