"""The entropy coder on its own, with no model or tokenizer involved."""

import pytest
import torch

from subtext_codec.arithmetic import (
    ENCODER_TAIL,
    FREQ_TOTAL,
    ArithmeticDecoder,
    ArithmeticEncoder,
    cumulative,
    from_bits,
    quantize_frequencies,
    to_bits,
)


def random_table(generator, alphabet_size):
    probs = torch.rand(alphabet_size, generator=generator, dtype=torch.float64) + 1e-3
    return cumulative(quantize_frequencies(probs))


# --------------------------------------------------------------------------
# frequency quantization
# --------------------------------------------------------------------------


@pytest.mark.parametrize("size", [2, 3, 17, 64])
def test_frequencies_are_well_formed(size):
    generator = torch.Generator().manual_seed(size)
    for _ in range(25):
        probs = torch.rand(size, generator=generator, dtype=torch.float64)
        freqs = quantize_frequencies(probs)
        assert sum(freqs) == FREQ_TOTAL
        assert all(f >= 1 for f in freqs)
        assert len(freqs) == size


def test_frequencies_survive_an_extremely_peaked_distribution():
    """Every symbol must stay reachable, or the coder cannot emit it."""
    probs = torch.tensor([1.0] + [1e-12] * 40, dtype=torch.float64)
    freqs = quantize_frequencies(probs)
    assert sum(freqs) == FREQ_TOTAL
    assert all(f >= 1 for f in freqs)


def test_frequencies_are_deterministic():
    probs = torch.rand(48, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    assert quantize_frequencies(probs) == quantize_frequencies(probs.clone())


# --------------------------------------------------------------------------
# bit packing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("payload", [b"", b"\x00", b"\xff", b"hello", bytes(range(64))])
def test_bit_packing_round_trip(payload):
    assert from_bits(to_bits(payload)) == payload


# --------------------------------------------------------------------------
# coder round trips
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alphabet_size", [2, 3, 17, 64])
@pytest.mark.parametrize("length", [1, 5, 200])
def test_symbols_round_trip(alphabet_size, length):
    """Symbols -> bits -> symbols, with the table changing every step."""
    generator = torch.Generator().manual_seed(alphabet_size * 1000 + length)
    tables = [random_table(generator, alphabet_size) for _ in range(length)]
    symbols = [
        int(torch.randint(alphabet_size, (1,), generator=generator).item())
        for _ in range(length)
    ]

    encoder = ArithmeticEncoder()
    for symbol, cum in zip(symbols, tables):
        encoder.encode(symbol, cum)
    bits = encoder.finish()

    decoder = ArithmeticDecoder(bits)
    assert [decoder.decode(cum) for cum in tables] == symbols


@pytest.mark.parametrize("payload_bytes", [1, 7, 64, 400])
def test_payload_round_trips_through_symbols(payload_bytes):
    """The direction the codec uses: bits -> symbols -> bits.

    Emitting symbols consumes payload bits; re-encoding those symbols must
    reproduce the payload. Only bits emitted during renormalization are pinned
    down by the symbols, so the stopping rule is "the mirror emitted enough",
    not "enough input was consumed" -- the encoder's flush bits are a free
    choice inside the final interval and need not match.
    """
    generator = torch.Generator().manual_seed(payload_bytes)
    payload = bytes(
        int(x) for x in torch.randint(256, (payload_bytes,), generator=generator)
    )
    bits = to_bits(payload)

    decoder = ArithmeticDecoder(bits + ENCODER_TAIL)
    mirror = ArithmeticEncoder()
    symbols, tables = [], []
    stagnant = 0
    while len(mirror.bits) < len(bits):
        emitted = len(mirror.bits)
        cum = random_table(generator, 40)
        symbol = decoder.decode(cum)
        mirror.encode(symbol, cum)
        tables.append(cum)
        symbols.append(symbol)
        stagnant = 0 if len(mirror.bits) > emitted else stagnant + 1
        assert stagnant < 100, (
            f"stalled at {len(mirror.bits)}/{len(bits)} bits "
            f"(pending={mirror.pending}, value={decoder.value})"
        )

    encoder = ArithmeticEncoder()
    for symbol, cum in zip(symbols, tables):
        encoder.encode(symbol, cum)
    assert encoder.finish()[: len(bits)] == bits


def test_zero_tail_starves_the_last_bits():
    """Why ENCODER_TAIL exists.

    Reading zeros past the payload parks the value register on the interval
    midpoint, where the coder underflows forever: pending grows without bound
    and the payload's final bits are never emitted. Regression guard, because
    the failure is silent -- a short message, not a crash.
    """
    generator = torch.Generator().manual_seed(99)
    payload = bytes(int(x) for x in torch.randint(256, (64,), generator=generator))
    bits = to_bits(payload)

    decoder = ArithmeticDecoder(bits)  # deliberately no tail
    mirror = ArithmeticEncoder()
    for _ in range(400):
        cum = random_table(generator, 40)  # one table per step, both halves
        mirror.encode(decoder.decode(cum), cum)
        if len(mirror.bits) >= len(bits):
            break

    assert len(mirror.bits) < len(bits), "expected the zero tail to starve the coder"
    assert mirror.pending > 100, "expected pending bits to grow without bound"
    assert decoder.value == 1 << 31, "expected the value register on the midpoint"


# --------------------------------------------------------------------------
# capacity behaviour
# --------------------------------------------------------------------------


def test_confident_distributions_cost_almost_nothing():
    """The property rank coding could not have: a near-certain step is ~free."""
    cum = cumulative(quantize_frequencies(
        torch.tensor([0.999, 0.0005, 0.0005], dtype=torch.float64)
    ))
    generator = torch.Generator().manual_seed(7)
    payload = bytes(int(x) for x in torch.randint(256, (64,), generator=generator))

    decoder = ArithmeticDecoder(to_bits(payload))
    mirror = ArithmeticEncoder()
    for _ in range(50):
        mirror.encode(decoder.decode(cum), cum)

    assert len(mirror.bits) < 15


def test_flat_distributions_carry_full_bits():
    """The other end: a flat 16-way distribution carries ~4 bits per step."""
    cum = cumulative(quantize_frequencies(torch.ones(16, dtype=torch.float64)))
    generator = torch.Generator().manual_seed(11)
    payload = bytes(int(x) for x in torch.randint(256, (128,), generator=generator))

    decoder = ArithmeticDecoder(to_bits(payload))
    mirror = ArithmeticEncoder()
    for _ in range(50):
        mirror.encode(decoder.decode(cum), cum)

    assert 190 <= len(mirror.bits) <= 210
