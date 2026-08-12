"""Integer arithmetic coding, in the CACM87 form.

This is the entropy coder the codec is built on, kept free of any model or
tokenizer so it can be tested exhaustively on its own. The two halves are exact
inverses: feeding a bitstream through :class:`ArithmeticDecoder` yields symbols
distributed according to the supplied frequencies, and running those symbols
back through :class:`ArithmeticEncoder` reproduces the bitstream.

Emitting a symbol the model gives probability ``p`` costs ``-log2(p)`` bits, so
a confident step carries almost nothing and an uncertain one carries a lot.
That is what makes the generated text follow the model's own distribution
rather than a flattened version of it.
"""

from __future__ import annotations

import bisect
from typing import List, Sequence

import torch

#: Width of the coder's interval registers.
PRECISION = 32
TOP = 1 << PRECISION
HALF = TOP >> 1
QUARTER = TOP >> 2
THREE_QUARTER = 3 * QUARTER

#: Frequencies are quantized to this total. 16 bits is far finer than the
#: differences between candidate tokens that survive the stability filter.
FREQ_BITS = 16
FREQ_TOTAL = 1 << FREQ_BITS

# Reading zeros past the end of a payload drives the decoder's value register
# onto the interval midpoint exactly, where it sits in permanent underflow: the
# coder accumulates pending bits forever and never emits the payload's final
# two. A non-degenerate tail breaks that symmetry. Its content is irrelevant --
# only the leading bits, which the payload pins down, are ever read back.
ENCODER_TAIL = [1, 0] * 48


class ArithmeticEncoder:
    """Symbols in, bits out.

    Bits appended to :attr:`bits` during renormalization are final: they are the
    leading bits of every value the interval still admits. The one or two bits
    added by :meth:`finish` are not -- they are a free choice inside the final
    interval, so a caller that needs exact bits must ensure enough were emitted
    by renormalization alone.
    """

    def __init__(self) -> None:
        self.low = 0
        self.high = TOP - 1
        self.pending = 0
        self.bits: List[int] = []

    def _emit(self, bit: int) -> None:
        self.bits.append(bit)
        while self.pending:
            self.bits.append(1 - bit)
            self.pending -= 1

    def encode(self, symbol: int, cum: Sequence[int], total: int = FREQ_TOTAL) -> None:
        span = self.high - self.low + 1
        self.high = self.low + (span * cum[symbol + 1]) // total - 1
        self.low = self.low + (span * cum[symbol]) // total
        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self) -> List[int]:
        self.pending += 1
        self._emit(0 if self.low < QUARTER else 1)
        return self.bits


class ArithmeticDecoder:
    """Bits in, symbols out."""

    def __init__(self, bits: Sequence[int]) -> None:
        self.src = list(bits)
        self.pos = 0
        self.low = 0
        self.high = TOP - 1
        self.value = 0
        for _ in range(PRECISION):
            self.value = (self.value << 1) | self._next()

    def _next(self) -> int:
        bit = self.src[self.pos] if self.pos < len(self.src) else 0
        self.pos += 1
        return bit

    def decode(self, cum: Sequence[int], total: int = FREQ_TOTAL) -> int:
        span = self.high - self.low + 1
        scaled = ((self.value - self.low + 1) * total - 1) // span
        symbol = bisect.bisect_right(cum, scaled) - 1
        self.high = self.low + (span * cum[symbol + 1]) // total - 1
        self.low = self.low + (span * cum[symbol]) // total
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.value -= HALF
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.value -= QUARTER
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.value = (self.value << 1) | self._next()
        return symbol

    @property
    def consumed(self) -> int:
        return self.pos


def quantize_frequencies(probs: torch.Tensor, total: int = FREQ_TOTAL) -> List[int]:
    """Integer frequencies, each at least 1, summing exactly to ``total``.

    Every step of this is deterministic -- float64 throughout, stable sorts --
    because encoder and decoder must derive byte-identical tables.

    The alphabet cannot exceed ``total``: every frequency is forced to at least
    1, so more than ``total`` symbols cannot sum to ``total``. Reject that
    rather than spin forever trying to reclaim frequencies that are all already
    at the floor.
    """
    if len(probs) > total:
        raise ValueError(
            f"cannot build a frequency table for {len(probs)} symbols in "
            f"{total} units; the alphabet is larger than FREQ_TOTAL. Lower top_k."
        )
    p = probs.double()
    p = p / p.sum()
    raw = p * total
    freqs = torch.clamp(raw.floor(), min=1.0)

    deficit = total - int(freqs.sum().item())
    if deficit > 0:
        # Hand the surplus to the largest fractional parts.
        order = torch.argsort(raw - raw.floor(), descending=True, stable=True)
        for i in range(deficit):
            freqs[order[i % len(order)]] += 1
    elif deficit < 0:
        # Reclaim from the largest frequencies, never below 1.
        order = torch.argsort(freqs, descending=True, stable=True)
        i = 0
        while deficit < 0:
            index = int(order[i % len(order)])
            if freqs[index] > 1:
                freqs[index] -= 1
                deficit += 1
            i += 1
    return [int(f) for f in freqs]


def cumulative(freqs: Sequence[int]) -> List[int]:
    """Cumulative table of length ``len(freqs) + 1``, starting at 0."""
    out = [0]
    for f in freqs:
        out.append(out[-1] + f)
    return out


def to_bits(data: bytes) -> List[int]:
    return [(byte >> shift) & 1 for byte in data for shift in range(7, -1, -1)]


def from_bits(bits: Sequence[int]) -> bytes:
    usable = len(bits) - len(bits) % 8
    return bytes(
        sum(bit << (7 - i) for i, bit in enumerate(bits[base : base + 8]))
        for base in range(0, usable, 8)
    )


__all__ = [
    "ENCODER_TAIL",
    "FREQ_TOTAL",
    "PRECISION",
    "ArithmeticDecoder",
    "ArithmeticEncoder",
    "cumulative",
    "from_bits",
    "quantize_frequencies",
    "to_bits",
]
