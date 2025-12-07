# subtext-codec  
### Steganographic data encoding in natural language using LLM logit-rank steering

**subtext-codec** is a proof-of-concept codec that hides arbitrary binary data inside seemingly normal LLM-generated text.  
It works by steering a language model’s next-token choices using the **rank** of each token in the model’s logit distribution.  
With the same model, tokenizer, prefix, and parameters, the process is fully reversible — enabling text that reads naturally while secretly encoding bytes.

---

## How it works

### Encoding
1. Input bytes are converted into digits in a configurable base (e.g., 2, 4, 8, 16).  
2. At each generation step, the LLM’s logits for the next token are sorted by probability.  
3. The next digit selects which ranked token to emit.  
4. This continues until all digits are consumed, producing a natural-language text containing the hidden payload.

### Decoding
Decoding replays the forward pass deterministically:
1. Start with the same prefix and model.  
2. For each generated token, compute the logits again and find its rank.  
3. Recover the digit stream and convert it back into bytes.

Decoding only requires:
- The generated text  
- The original prompt prefix  
- The same model + tokenizer  
- The codec parameters (base, top-k if used)

---

## Features

- **Configurable base** — trade off naturalness vs. capacity  
- **Deterministic next-token steering** using logits, no randomness  
- **Full round-trip encode/decode** for arbitrary byte payloads  
- **Single-token terminator** — decoder stops at the first token ranked outside the base  
- **Hugging Face Transformers backend** — works with most causal LMs  
- **Readable, compact implementation** designed for experimentation  
- **External key file** captures encode-time metadata for reliable decoding  

---

## Installation

From PyPI (or via `uv`):

```bash
uv pip install subtext-codec
# or: pip install subtext-codec
```

From source:

```bash
git clone https://github.com/shevisj/subtext
cd subtext
uv venv .env --python 3.13
uv pip install -r requirements.txt
```

`torch` and `transformers` are the only real runtime dependencies, but to reduce decoding errors all package versions are pinned in `requirements.txt`.

Tests use `pytest`.

---

## Usage
The CLI exposes `encode` and `decode` subcommands. Shared flags:

- `--model-name-or-path` – Hugging Face model name or local path (causal LM); optional on decode if stored in the key file
- `--prompt-prefix` – prefix text used for both encode and decode (defaults from key if present)
- `--device` – e.g. `cpu` or `cuda` (falls back to the key value or defaults to `cpu`)
- `--torch-dtype` – optional weight dtype (`auto`, fp16, bf16, fp32)
- `--max-context-length` – optional guardrail; defaults to model limit
- `--seed` – deterministic seeding (default: 0)

### Encode bytes into text

```bash
subtext-codec encode \
  --model-name-or-path gpt2 \
  --base 8 \
  --prompt-prefix "Once upon a time, " \
  --input-bytes secret.txt \
  --output-text message.txt \
  --key key.json \
  --max-new-tokens 512 \
  --top-k 16
```

The output text is just the generated story (no metadata header). The accompanying `codec.key.json` captures `base`, `top_k`, the prompt prefix used to generate the message, the `device`, and the `torch_dtype`.
The encoder automatically appends a single terminator token whose rank is the first index outside the chosen base; the decoder stops at that token and ignores any trailing text.

You can also reuse an existing key instead of re-entering parameters:

```bash
subtext-codec encode \
  --key codec.key.json \
  --input-bytes secret.bin \
  --output-text message.txt \
  --include-model-in-key
```

If the path you pass to `--key` already exists, its values (base, top-k, prompt prefix, device, model name if saved) are reused; any CLI overrides are written back to the same file. If it does not exist, the encoder creates it after generation. The model name you supply is persisted so you can decode without re-specifying it; the legacy `--include-model-in-key` flag remains for compatibility. The key also stores `torch_dtype` so you can replay the same loading setup.

### Decode text back into bytes

```bash
subtext-codec decode \
  --input-text message.txt \
  --key codec.key.json \
  --output-bytes decoded.bin
```

The base and `top_k` are read from the key file. If the key already stores the model name from encode time you can omit `--model-name-or-path` here. The prompt prefix is taken from the key unless you explicitly pass `--prompt-prefix` (it must still match the encode run).
Decoding stops as soon as it encounters a generated token whose rank is **outside** the configured base (i.e., rank `>= base`), discarding that terminator and any text after it.
Any CLI overrides you provide for prompt/device/model during decode are saved back into the key for future reuse.
Decoded bytes are reconstructed from the digit stream without persisting the original payload length, so leading zero bytes are intentionally stripped.

### Sample artifacts

If you just want to poke at the codec without generating new data, there is a small fixture set under `samples/`:

- `samples/message.txt` — generated text that hides `samples/secret.txt`
- `samples/codec.key.json` — matching base/top-k/prompt/model metadata
- `samples/decoded.txt` — expected decode output for comparison

To decode the included example back into bytes:

```bash
subtext-codec decode \
  --input-text samples/message.txt \
  --key samples/codec.key.json \
  --output-bytes samples/decoded.txt
```

The key was created with the prompt prefix `"Once upon a time, "` and a Llama 3.1 8B model; decoding requires access to the same model and tokenizer.

---

## Limitations (for now)

* **Brittle to edits**: modifying even a single output token breaks decoding
* **Model-dependent**: requires the exact same weights + tokenizer
* **Floating-point sensitivity**: extreme ties in logits may reorder ranks
* **Context length**: large payloads may exceed model context without chunking
* **Parameter drift**: mismatching base, top-k, or prompt prefix will break decoding

This project is a **research prototype**, not a secure or production steganography system.

---

## Testing

```
python -m pytest
```

The slow round-trip test uses `sshleifer/tiny-gpt2`; if the model cannot be downloaded (e.g., offline), the test is skipped.

---

## License

MIT
