# subtext-codec
### Steganographic data encoding in natural language via arithmetic coding against an LLM

**subtext-codec** hides arbitrary binary data inside seemingly normal LLM-generated text. The payload drives an arithmetic coder whose probability model is the language model's own next-token distribution, so the text it produces is distributed exactly as ordinary sampling would be while secretly encoding bytes. With the same model, tokenizer, prompt prefix and parameters, the process is fully reversible.

---

## How it works

The payload drives an **arithmetic decoder whose probability model is the language model itself**. Emitting a token the model assigns probability `p` consumes `-log2(p)` payload bits, so a confident step carries almost nothing and an uncertain one carries a lot.

Two properties follow, and they are the entire point of the design:

- The generated text is distributed **exactly** as ordinary sampling at the chosen temperature. Nothing is ever forced off-distribution.
- Capacity per token equals the distribution's entropy, which is the information-theoretic ceiling for text that stays indistinguishable from sampling.

### Encoding
1. The payload is framed with a length and a CRC32, then read as a bitstream.
2. At each step the model's logits are clipped to `top_k`, filtered to tokens that are **safe to emit** (see below), and turned into an integer frequency table at `temperature`.
3. The arithmetic decoder consumes payload bits and selects a token from that table.
4. Generation stops as soon as the emitted tokens pin down every payload bit.

### Decoding
1. Start from the same prompt prefix and model, and rebuild the same table at each step.
2. Run the arithmetic *encoder* over the observed tokens, which reproduces the bitstream.
3. Stop once the declared length has been recovered, then check the CRC.

Decoding needs the text, the prompt prefix, the same model and tokenizer, and the codec parameters -- all of which the key file carries. No terminator token is required, so text after the message is simply ignored.

### Why the encoder checks itself

The coder re-encodes each symbol as it emits it and refuses to return a message whose bits it cannot reproduce. That is not belt-and-braces: feeding the arithmetic decoder a zero-filled tail past the end of the payload parks its value register exactly on the interval midpoint, where it underflows forever and the payload's **last two bits are never emitted**. The result is a message that looks fine and silently fails to decode. The check is what turns that class of bug into a loud failure.

### Why the safety filter exists

The encoder chooses *token ids*, but the decoder is handed *text*. Nothing guarantees that writing tokens out and reading them back returns the same ids: a byte-level BPE vocabulary will merge an emitted token into its neighbours, so `'an'` written after `'Kanz'` reads back as `'anz'` and every subsequent step decodes against the wrong prefix.

So a candidate is only usable if appending it leaves the surrounding text re-tokenizing unchanged. Both sides apply that same filter to the same prefix, so they stay in step, and the encoder verifies the finished text tokenizes back to exactly what it built before handing it over. A message that cannot satisfy this raises at encode time rather than becoming an artifact that silently fails to decode.

---

## Features

- Entropy-optimal capacity -- each token carries `-log2(p)` bits, the theoretical maximum for output that still looks like sampling
- Output distributed exactly as the model's own sampling, so no per-token statistical tell
- Temperature as a principled capacity dial, replacing rank truncation
- Tokenizer-stable candidate selection, verified before any message is returned
- Self-inverse check at encode time -- the coder re-encodes as it goes and refuses to emit a message it cannot read back
- Framed payload with a CRC32, so a wrong key, prompt or model is detected rather than yielding plausible wrong bytes
- Incremental KV-cached stepping, so cost is linear in message length rather than quadratic
- Deterministic throughout; verified bit-identical on CPU and CUDA across fp32/bf16/fp16
- Hugging Face Transformers backend -- works with most causal LMs

---

## Installation

From PyPI (or via `uv`):

```bash
uv pip install subtext-codec
# or: pip install subtext-codec
```

From source:

```bash
git clone https://github.com/shevisj/subtext-codec
cd subtext-codec
uv venv --python 3.13
uv pip install -r requirements.txt
uv pip install -e .
```

`torch` and `transformers` are the only runtime dependencies. The published package sets lower bounds only, so it installs alongside whatever torch build you already have; `requirements.txt` pins an exact, CPU-only development environment and is the one to use when reproducing a decode.

---

## Usage

The CLI exposes `encode` and `decode`. Shared flags:

- `--key` -- path to the codec key file (required)
- `--model-name-or-path` -- HuggingFace model id or local path; optional if stored in the key
- `--prompt-prefix` -- prefix used for both encode and decode; taken from the key if present
- `--device` -- e.g. `cpu` or `cuda` (falls back to the key, then `cpu`)
- `--torch-dtype` -- weight dtype (`auto`, fp16, bf16, fp32)
- `--max-context-length` -- cap on sequence length; defaults to the model's own limit
- `--seed` -- deterministic seed (default: 0)
- `--quiet` -- suppress the progress line on stderr

### Encode bytes into text

```bash
subtext-codec encode \
  --model-name-or-path gpt2 \
  --prompt-prefix "Once upon a time, " \
  --input-bytes secret.txt \
  --output-text message.txt \
  --key key.json \
  --temperature 1.5
```

Encode-only flags:

- `--temperature` -- the capacity dial (default: 1.5). Higher packs more payload per token and shortens the cover text, at the cost of more erratic prose. See [Choosing a temperature](#choosing-a-temperature).
- `--top-k` -- candidates considered per step (default: 64). This bounds the stability filter's work; it is not a capacity dial. Must be at least 2.
- `--max-new-tokens` -- fail rather than generate more than this many tokens
- `--no-store-model` -- keep the model id out of the key
- `--verify` -- decode the result before writing it, confirming the round trip end to end

The output text is just the generated story, with no metadata header. `key.json` records `temperature`, `top_k`, the prompt prefix, the device, the dtype and (unless `--no-store-model`) the model id.

If the path given to `--key` already exists, its values are reused as defaults and any CLI overrides are written back, so a second message needs only the key:

```bash
subtext-codec encode --key key.json --input-bytes secret.bin --output-text message.txt
```

### Decode text back into bytes

```bash
subtext-codec decode \
  --input-text message.txt \
  --key key.json \
  --output-bytes decoded.bin
```

Parameters come from the key unless overridden for the run. **Decoding never modifies the key file** -- the one artifact a message cannot be recovered without is not rewritten by a read operation.

Decoding stops as soon as the payload's declared length has been recovered, so trailing text is harmless. Text before the prompt prefix is skipped too. If the message was altered, or the key, prompt or model do not match, the CRC32 catches it rather than returning plausible wrong bytes.

### Sample artifacts

`samples/secret.txt` is a payload to play with. Build the rest of the fixture from it:

```bash
python samples/regenerate.py
```

That encodes it with `gpt2` on CPU -- a ~500MB download, no GPU or gated checkpoint needed -- and writes `message.txt`, `key.json` and `decoded.txt` alongside it, verifying the round trip before writing anything. Encoding is deterministic, so it reproduces the same message every time.

Then decode it back the normal way:

```bash
subtext-codec decode \
  --input-text samples/message.txt \
  --key samples/key.json \
  --output-bytes /tmp/decoded.txt

diff samples/secret.txt /tmp/decoded.txt && echo "exact match"
```

(Before 1.0 the fixture was checked in, but it was a `v2` message against a gated 16GB Llama checkpoint -- unreadable by this version and unrunnable by most people. Generating it locally is both smaller and honest.)

### Programmatic usage

See the [demo notebook](/demo.ipynb).

---

## Choosing a temperature

`temperature` is the capacity dial. Raising it flattens the distribution, so each token carries more payload and the cover text gets shorter -- but the text becomes more erratic, exactly as sampling at that temperature would. `top_k` is *not* a capacity dial here; it only bounds how much work the stability filter does.

Measured on the 445-byte `samples/secret.txt`, Qwen2.5-7B (bf16) on an RTX 5090, `top_k=64`:

| Temperature | Cover tokens | Bits/token | Character |
| --- | --- | --- | --- |
| 1.0 | 2914 | 1.22 | best prose, but far too long to be plausible |
| **1.5** | **894** | **3.98** | **the knee -- reads like a real, rambling review** |
| 2.0 | 709 | 5.02 | densest, but visibly erratic and prone to invention |

Mean surprisal of the emitted tokens matches capacity at every setting (1.23 / 4.02 / 5.07 bits), which is the confirmation that the coder is running at the distribution's entropy rather than leaving capacity on the table.

For reference, the rank-coding scheme this replaced in 1.0 managed 4.35 bits/token at its best setting and 1.57 at its most conservative, and its output was *never* model-distributed. Arithmetic coding at temperature 1.5 matches its best capacity, and at 2.0 exceeds it, while staying on-distribution throughout.

**A bigger model does not buy more capacity.** A stronger model concentrates probability mass, which *lowers* entropy and therefore bits per token. It buys better prose at a given temperature, not shorter text. Keep payloads to a few hundred bytes, and pick a prompt whose natural continuation is long-form so the required length is not conspicuous.

**Encrypt or compress first if it matters.** The indistinguishability argument assumes the payload bits look uniform. Feeding raw ASCII biases the emitted text.

---

## Determinism and devices

The codec reduces to one requirement: encode and decode must derive the same frequency table at every step, down to the last integer. Everything after the forward pass -- the sort, the quantization, the stability filter -- runs on **CPU in float64 no matter where the model lives**, so the only device-sensitive component is the logits themselves. Encode and decode also drive the model identically (prefill the prompt, then one token at a time), so they hit the same kernels at the same shapes.

This has been measured, not just reasoned about. On an RTX 5090 (Blackwell, `sm_120`, torch 2.13 + CUDA 13), repeated passes over the same tokens produce **bit-identical** logits in fp32, bf16 and fp16, and all three round-trip on GPU. GPU is as reliable as CPU.

What matters is that the *environment* matches between encode and decode:

| Factor | Effect |
| --- | --- |
| Different dtype (bf16 ↔ fp32) | Breaks. The key records the dtype; keep it. |
| TF32 on vs off | Pinned off by `set_deterministic`, so it no longer varies. |
| Different attention backend (SDPA / FlashAttention / eager) | Can break across machines. Same environment, same choice. |
| Different torch / transformers version | Can break; pin them (`requirements.txt`) if a message must survive an upgrade. |
| Different device (`cuda` ↔ `cpu`) | Unreliable, but not automatically fatal -- see below. |

Crossing devices is **not supported, though it is not guaranteed to fail**: gpt2 in fp32 does decode on CPU from a CUDA encode, because the two happen to agree bit for bit. Larger models and bf16 will not be so lucky. Do not rely on it. What *is* guaranteed is that a mismatch is caught rather than returning plausible wrong bytes -- that is the payload framing's job.

`set_deterministic` also sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` (via package import, before torch loads) and disables cuDNN autotuning. It holds fp32 matmuls at full precision, which costs some speed on Ampere and later; a deliberate trade, since a mismatch costs the entire message.

bf16 and fp16 work but leave less margin -- half the mantissa means adjacent logits sit closer together. fp32 is safer when you can afford it. Exact ties are fine in any precision: the sort is stable, so ties resolve by token id.

To confirm on your own hardware before trusting a message to it:

```bash
uv run pytest tests/test_determinism.py -v
```

Those parametrize over every device and dtype present, so on a GPU box they additionally check bit-identical logits on CUDA, matching bands, and fp32/bf16/fp16 round trips.

---

## Compatibility

1.0 reads and writes one wire format, `v1`, and nothing else. Every pre-1.0 format is gone along with the code that decoded them: they used rank coding, which corrupted roughly a third of its own messages and never produced model-distributed output.

Pre-1.0 also numbered a format `v1`, but that one carried `base`/`top_p` and no `temperature`, so the collision is detectable -- such a key fails with an explanation rather than a generic error. To read a message encoded before 1.0, install `subtext-codec~=0.2.0`; to keep it, re-encode the payload with this version.

---

## Limitations

* **Brittle to edits**: changing a single token of the output breaks decoding. This is an encoding scheme, not an error-correcting one.
* **Model-dependent**: requires the exact same weights, tokenizer *and* dtype. Decoding a bf16 encode in fp32 will not work.
* **Not confidential on its own**: the key is metadata, not a cipher. Encrypt the payload before encoding it -- both for secrecy and because the indistinguishability argument assumes uniform payload bits.
* **Context length**: large payloads may exceed the model's context window; there is no chunking.
* **Capacity**: the distribution's entropy, typically 1-5 bits per token depending on temperature. Expect a few hundred tokens per hundred bytes.
* **Length is the remaining tell**: the per-token statistics are right, but a 900-token hotel review is still an odd thing to find. Pick a prompt whose natural continuation is long-form.
* **"Indistinguishable" is relative to the temperature you chose**: output at `temperature=1.5` is exactly temperature-1.5 sampling, which is only unremarkable to someone who has no expectation about the setting. An adversary who knows you would have sampled at 1.0 can distinguish it in aggregate. Lower temperature closes that gap and lengthens the text.

This project is a **research prototype**, not a secure or production steganography system.

---

## Testing

```bash
uv run pytest              # everything
uv run pytest -m "not slow"  # skip tests that download a checkpoint
```

The slow tests use `sshleifer/tiny-gpt2` and skip themselves if the Hub is unreachable.

---

## License

MIT
