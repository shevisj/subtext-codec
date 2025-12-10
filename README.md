# subtext-codec
### Steganographic data encoding in natural language using LLM logit-rank steering

**subtext-codec** is a proof-of-concept codec that hides arbitrary binary data inside seemingly normal LLM-generated text. It steers a language model's next-token choices using the **rank** of each token in the model's logit distribution. With the same model, tokenizer, prefix, and parameters, the process is fully reversible -- enabling text that reads naturally while secretly encoding bytes.

---

## How it works

### Encoding
1. Input bytes are treated as a big integer.  
2. For each generation step, the model's logits for the next token are sorted (optionally truncated with `top_k`) and softmaxed.  
3. Tokens are kept until their cumulative probability exceeds `top_p`; that count is the active base for the step.  
4. The payload integer is divided by the active base to pick the ranked token to emit; the quotient is carried forward to the next step with a newly computed base.  
5. Once the payload is exhausted, a single terminator token is emitted using the first rank outside the active base.

### Decoding
Decoding replays the forward pass deterministically:
1. Start with the same prefix and model.  
2. For each generated token, recompute the logits, rebuild the dynamic base from `top_p`/`top_k`, and find the token's rank.  
3. Collect the rank stream until a token falls outside the active base (the terminator), then reconstruct the original bytes from the mixed-radix digits.

Decoding only requires:
- The generated text  
- The original prompt prefix  
- The same model + tokenizer  
- The codec parameters (`top_p`, `top_k` if used; legacy `v1` keys that store a fixed base still decode via the old path)

---

## Features

- Adaptive base per token -- capacity rises and falls with the model's confidence using `top_p` + optional `top_k`  
- Deterministic next-token steering -- logits only, no randomness  
- Mixed-radix payload reconstruction -- handles variable bases without length metadata  
- Single-token terminator -- chosen as the first rank outside the active base  
- Hugging Face Transformers backend -- works with most causal LMs  
- Readable, compact implementation designed for experimentation  
- External key file captures encode-time metadata for reliable decoding  

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
uv venv .env --python 3.13
uv pip install -r requirements.txt
```

`torch` and `transformers` are the only real runtime dependencies, but to reduce decoding errors all package versions are pinned in `requirements.txt`.

Tests use `pytest`.

---

## Usage
The CLI exposes `encode` and `decode` subcommands. Shared flags:

- `--model-name-or-path` -- Hugging Face model name or local path (causal LM); optional on decode if stored in the key file
- `--prompt-prefix` -- prefix text used for both encode and decode (defaults from key if present)
- `--device` -- e.g. `cpu` or `cuda` (falls back to the key value or defaults to `cpu`)
- `--torch-dtype` -- optional weight dtype (`auto`, fp16, bf16, fp32)
- `--max-context-length` -- optional guardrail; defaults to model limit
- `--seed` -- deterministic seeding (default: 0)

### Encode bytes into text

```bash
subtext-codec encode \
  --model-name-or-path gpt2 \
  --prompt-prefix "Once upon a time, " \
  --input-bytes secret.txt \
  --output-text message.txt \
  --key key.json \
  --max-new-tokens 512 \
  --top-k 16 \
  --top-p 0.9
```

The output text is just the generated story (no metadata header). The accompanying `key.json` captures `top_p`, `top_k`, the prompt prefix used to generate the message, the `device`, and the `torch_dtype`. The model name can also be stored for reuse. `top_p` defaults to `0.9` if not provided.  
The encoder automatically appends a single terminator token whose rank is the first index outside the active base at the final step; the decoder stops at that token and ignores any trailing text.

You can also reuse an existing key instead of re-entering parameters:

```bash
subtext-codec encode \
  --key key.json \
  --input-bytes secret.bin \
  --output-text message.txt \
  --include-model-in-key
```

If the path you pass to `--key` already exists, its values (top-p, top-k, prompt prefix, device, model name if saved) are reused; any CLI overrides are written back to the same file. The model name you supply is persisted so you can decode without re-specifying it; the legacy `--include-model-in-key` flag remains for compatibility. The key also stores `torch_dtype` so you can replay the same loading setup.

### Decode text back into bytes

```bash
subtext-codec decode \
  --input-text message.txt \
  --key key.json \
  --output-bytes decoded.bin
```

The `top_p`/`top_k` parameters are read from the key file (unless you override them for the current run). If the key already stores the model name from encode time you can omit `--model-name-or-path` here. The prompt prefix is taken from the key unless you explicitly pass `--prompt-prefix` (it must still match the encode run).  
Decoding stops as soon as it encounters a generated token whose rank is **outside** the active base for that step, discarding that terminator and any text after it.  
Any CLI overrides you provide for prompt/device/model during decode are saved back into the key for future reuse.  
Decoded bytes are reconstructed from the digit stream without persisting the original payload length, so leading zero bytes are intentionally stripped.

### Sample artifacts

If you just want to poke at the codec without generating new data, there is a small fixture set under `samples/`:

- `samples/message.txt` -- generated text that hides `samples/secret.txt`
- `samples/key.json` -- v2 key using dynamic bases with `top_p=0.9` and `top_k=16` (prompt/model metadata included)
- `samples/decoded.txt` -- expected decode output for comparison

To decode the included example back into bytes:

```bash
subtext-codec decode \
  --input-text samples/message.txt \
  --key samples/key.json \
  --output-bytes samples/decoded.txt
```

The key was created with the prompt prefix `"Once upon a time, "` and a Llama 3.1 8B model; decoding requires access to the same model and tokenizer. The sample key is v2 and uses the variable-base codec; legacy v1 keys are still decoded via the fixed-base path if encountered.

---

## Limitations (for now)

* Brittle to edits: modifying even a single output token breaks decoding
* Model-dependent: requires the exact same weights + tokenizer
* Floating-point sensitivity: extreme ties in logits may reorder ranks
* Context length: large payloads may exceed model context without chunking
* Parameter drift: mismatching top-p/top-k or prompt prefix will break decoding

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
