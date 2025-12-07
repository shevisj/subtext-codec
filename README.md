# subtext  
### Steganographic data encoding in natural language using LLM logit-rank steering

**subtext** is a proof-of-concept codec that hides arbitrary binary data inside seemingly normal LLM-generated text.  
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
- **Hugging Face Transformers backend** — works with most causal LMs  
- **Readable, compact implementation** designed for experimentation  
- **Metadata header** embedded in output for reliable decoding  

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

- `--model-name-or-path` – Hugging Face model name or local path (causal LM)
- `--prompt-prefix` – prefix text used for both encode and decode
- `--device` – e.g. `cpu` or `cuda`
- `--top-k` – optional cap on candidate tokens; must satisfy `base <= top_k`
- `--max-context-length` – optional guardrail; defaults to model limit
- `--seed` – deterministic seeding (default: 0)

### Encode bytes into text

```bash
subtext encode \
  --model-name-or-path gpt2 \
  --base 8 \
  --prompt-prefix "Once upon a time, " \
  --input-bytes secret.bin \
  --output-text message.txt \
  --max-new-tokens 512 \
  --top-k 16
```

The output text begins with a metadata header, e.g.:

```
[LLM-CODEC v1; base=8; length=42; top_k=16]
Once upon a time, ...
```

### Decode text back into bytes

```bash
subtext decode \
  --model-name-or-path gpt2 \
  --prompt-prefix "Once upon a time, " \
  --input-text message.txt \
  --output-bytes decoded.bin \
  --top-k 16
```

The base, payload length, and `top_k` are pulled from the header; you still need to supply the prompt prefix and model path.

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
