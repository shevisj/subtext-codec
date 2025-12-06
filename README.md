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

```bash
git clone https://github.com/<your-username>/subtext
cd subtext
pip install -r requirements.txt
```

Requirements (typical):

```text
torch
transformers
```

---

## Usage

### Encode bytes into text

```bash
python llm_codec.py encode \
  --model-name-or-path gpt2 \
  --base 8 \
  --prompt-prefix "Once upon a time, " \
  --input-bytes secret.bin \
  --output-text message.txt \
  --max-new-tokens 512
```

### Decode text back into bytes

```bash
python llm_codec.py decode \
  --model-name-or-path gpt2 \
  --prompt-prefix "Once upon a time, " \
  --input-text message.txt \
  --output-bytes decoded.bin
```

---

## Limitations (for now)

* **Brittle to edits**: modifying even a single output token breaks decoding
* **Model-dependent**: requires the exact same weights + tokenizer
* **Floating-point sensitivity**: extreme ties in logits may reorder ranks
* **Context length**: large payloads may exceed model context without chunking

This project is a **research prototype**, not a secure or production steganography system.

---

## License

MIT
