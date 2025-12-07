Here’s a concrete build spec you can hand to an AI agent (or a human) to implement the PoC.

---

## 0. Goal

Implement a **round-trip codec** that:

* **Encodes arbitrary binary data** into a **natural-language-looking text** by steering an LLM’s next-token choices using the **rank of tokens in the logit distribution**.
* **Decodes** that text back into the original data **using only**:

  * the same model weights + tokenizer,
  * the same initial prompt/prefix,
  * the same “base” and other agreed parameters.

We don’t care about speed or polish. We care about: **correctness**, **reproducibility**, and a clear separation of concerns.

---

## 1. High-Level Design

### 1.1 Core idea

At each generation step:

1. Run the LLM on the current context to get logits over the vocab.
2. Sort tokens by descending logit (most likely first).
3. Take the next digit from an input number stream (base `B`).
4. Use that digit as an **index** into the sorted token list, pick that token, append it to the text.
5. Repeat until all digits are consumed (or we hit some stopping condition).

Decoding:

1. Start from the same prompt and model.
2. For each generated token:

   * Compute logits given the current context.
   * Sort tokens by descending logit.
   * Find the **rank** of the actual next token in this sorted list.
   * That rank is the next base-`B` digit.
3. Recover the digit string, then turn it back into bytes.

---

## 2. Implementation Scope

Implement a **single Python CLI script**:

* `__main__.py`
* Using:

  * `transformers` (Hugging Face)
  * `torch`
  * Standard library only otherwise.

CLI:

* `encode` mode: bytes → digit stream → LLM-generated text.
* `decode` mode: text → digit stream → bytes.
* Simple config via CLI flags and/or small YAML/JSON config file.

---

## 3. Configuration / Parameters

These must be configurable either via CLI or config file:

* `--model-name-or-path` (e.g. `"gpt2"` or local path)
* `--device` (e.g. `"cuda"` or `"cpu"`)
* `--base` (integer `B >= 2`, e.g. 2, 4, 8, 16, 32)
* `--prompt-prefix` (string) – initial text for the model.
* `--max-new-tokens` – safety guard.
* `--max-context-length` – from model config, but allow overriding.
* `--top-k` (optional) – if set, restrict selection to top-k tokens and enforce `base <= k`.
* `--input-bytes` – path to file to encode (or `stdin`).
* `--output-text` – path to write encoded text (or `stdout`).
* `--input-text` – path to text to decode.
* `--output-bytes` – path to write decoded bytes.

Also record and/or accept these at decode time:

* `--base`
* `--top-k`
* `--prompt-prefix`
* `--model-name-or-path`

Decoding must **not** guess these; they must match.

---

## 4. Data Representation

### 4.1 Bytes → base-B digit string

Implement helper functions:

```python
def bytes_to_base_digits(data: bytes, base: int) -> list[int]:
    """Return list of digits in [0, base-1] representing data."""
```

Procedure:

1. Interpret bytes as a **big integer**:

   ```python
   n = int.from_bytes(data, byteorder="big", signed=False)
   ```
2. Special case: if `len(data) == 0`, return an empty digit list.
3. Convert `n` to base-`B` digits (most significant digit first):

   ```python
   digits = []
   while n > 0:
       n, rem = divmod(n, base)
       digits.append(rem)
   digits.reverse()
   ```
4. To make decoding simpler, **store the original byte length** separately and include it in the encoded text metadata (see §4.3).

### 4.2 Base-B digit string → bytes

```python
def base_digits_to_bytes(digits: list[int], base: int, length_bytes: int) -> bytes:
    """Inverse of bytes_to_base_digits given the original length."""
```

1. Compute integer:

   ```python
   n = 0
   for d in digits:
       n = n * base + d
   ```
2. Convert back to bytes with the known length:

   ```python
   data = n.to_bytes(length_bytes, byteorder="big", signed=False)
   ```
3. If `len(data) > length_bytes`, truncate from the left; if less, pad with leading zeros (this should not happen if encoding/decoding is consistent but guard anyway).

### 4.3 Metadata encoding

We need to carry **metadata** through in the text so decoding knows:

* Original byte length (`length_bytes`).
* Base (`B`).
* Optional: a small version marker.

For this PoC, keep it simple:

* Prepend a short **header line** before the natural-language body, for example:

  ```
  [LLM-CODEC v1; base=16; length=1234]
  <generated text continues here...>
  ```

* Use a stable regex to parse this in decode mode.

* The actual stego/stego-clean version can hide this later; for now we want it reliable.

---

## 5. LLM Integration Details

### 5.1 Model loading

Use `AutoTokenizer` and `AutoModelForCausalLM`:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
model.eval()
```

* Disable gradient computation (`torch.no_grad()`).
* Set `pad_token` to `eos_token` if needed.

### 5.2 Determinism

Requirements:

* Use a **fixed random seed** and set all relevant PyTorch flags for determinism.
* Use **no sampling** in generation – we override token selection with a deterministic index anyway.
* The only source of variation between different runs should be the **input data**.

---

## 6. Encoding Algorithm (Bytes → Text)

Implement:

```python
def encode_data_to_text(
    data: bytes,
    base: int,
    model,
    tokenizer,
    prompt_prefix: str,
    max_new_tokens: int,
    top_k: int | None = None,
) -> str:
    ...
```

### 6.1 Steps

1. **Convert data to digits** using `bytes_to_base_digits`.

2. **Prepare initial context**:

   ```python
   text = prompt_prefix
   input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
   ```

3. Maintain an index `digit_idx = 0`.

4. Loop while `digit_idx < len(digits)` and `generated_tokens < max_new_tokens`:

   * Run the model forward:

     ```python
     with torch.no_grad():
         outputs = model(input_ids=input_ids)
         logits = outputs.logits[:, -1, :]  # shape [1, vocab_size]
     ```

   * Convert `logits` to CPU and sort indices by descending score:

     ```python
     logits = logits.squeeze(0).cpu()
     sorted_indices = torch.argsort(logits, descending=True)
     ```

   * If `top_k` is not `None`, truncate:

     ```python
     if top_k is not None:
         sorted_indices = sorted_indices[:top_k]
     ```

   * **Check base vs candidate count**:

     ```python
     if base > len(sorted_indices):
         raise ValueError("base > number of candidate tokens (top_k).")
     ```

   * Take current digit `d = digits[digit_idx]`.

     * If `d >= base`: this should never happen; treat as error.
     * Index into `sorted_indices[d]` to get `next_token_id`.

   * Append `next_token_id` to `input_ids`:

     ```python
     next_token_id = sorted_indices[d].unsqueeze(0).unsqueeze(0).to(device)
     input_ids = torch.cat([input_ids, next_token_id], dim=1)
     ```

   * Increment `digit_idx` and `generated_tokens`.

   * Optional: **avoid EOS** early in generation:

     * If `next_token_id` == `tokenizer.eos_token_id` and `digit_idx < len(digits)`:

       * For PoC: either allow it or skip; simplest is **allow EOS** but keep going; the model will just be generating post-EOS garbage, which is fine here.

5. After loop: decode full sequence:

   ```python
   full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
   ```

6. Prepend metadata header:

   ```python
   header = f"[LLM-CODEC v1; base={base}; length={len(data)}]"
   return header + "\n" + full_text
   ```

7. Add logging / debug info (e.g. number of tokens generated, number of digits consumed) to stdout.

---

## 7. Decoding Algorithm (Text → Bytes)

Implement:

```python
def decode_text_to_data(
    encoded_text: str,
    model,
    tokenizer,
) -> bytes:
    ...
```

### 7.1 Steps

1. **Parse metadata header**:

   * Use a regex like:

     ```text
     ^\[LLM-CODEC v1; base=(\d+); length=(\d+)\]\s*(.*)$
     ```

     capturing `base`, `length_bytes`, and `body_text`.

2. **Reconstruct prompt vs generated text**

   We need to split:

   * `prompt_prefix` (known externally, must be provided at CLI).
   * `generated_body` = text after the prompt.

   For PoC, make this simple:

   * Require that decoding is always called with **the same `prompt_prefix` string** passed as CLI arg.
   * Then, inside `decode_text_to_data`, after stripping the header, assert that `body_text` starts with `prompt_prefix` or, if that’s not robust, simply ignore that check and reconstruct context as:

     ```python
     context_text = prompt_prefix
     generated_text = body_text[len(prompt_prefix):]
     ```

   Note: If whitespace/tokenization issues arise, we can instead:

   * Tokenize the entire `body_text`.
   * Tokenize `prompt_prefix`.
   * Drop the first `len(prompt_ids)` tokens from `body_ids` to get `generated_ids`.

3. **Tokenization**

   ```python
   body_ids = tokenizer(body_text, return_tensors="pt").input_ids.to(device)
   prompt_ids = tokenizer(prompt_prefix, return_tensors="pt").input_ids.to(device)

   # generated_ids: drop prompt part
   generated_ids = body_ids[:, prompt_ids.shape[1]:]
   ```

4. **Iteratively recover digits**

   Initialize:

   ```python
   input_ids = prompt_ids.clone()
   digits = []
   ```

   For each `next_token_id` in `generated_ids[0]`:

   * Run model on current context:

     ```python
     with torch.no_grad():
         outputs = model(input_ids=input_ids)
         logits = outputs.logits[:, -1, :]  # [1, vocab_size]
         logits = logits.squeeze(0).cpu()
         sorted_indices = torch.argsort(logits, descending=True)
     ```

   * If `top_k` was used during encoding, decode must know it and apply the same truncation.

   * Find the **rank** of `next_token_id` in `sorted_indices`:

     ```python
     next_id = int(next_token_id)
     positions = (sorted_indices == next_id).nonzero(as_tuple=True)[0]
     if len(positions) == 0:
         # This should not happen; treat as error.
         raise ValueError("Generated token not found in sorted indices")
     d = int(positions.item())
     ```

   * Append `d` to `digits`.

   * Append `next_token_id` to `input_ids` and continue.

   * Optional early stop: Once we have enough digits to reconstruct `length_bytes` and possibly some known max bit length, we can stop. For PoC, easiest is to **consume all generated tokens**, then reconstruct.

5. **Convert digits back to bytes**

   * Use `base_digits_to_bytes(digits, base, length_bytes)`.

6. Return bytes.

---

## 8. CLI Structure

Use `argparse` with subcommands:

* `encode`:

  * `--model-name-or-path`
  * `--base`
  * `--prompt-prefix`
  * `--input-bytes`
  * `--output-text`
  * `--top-k`
  * `--max-new-tokens`
  * `--device`

* `decode`:

  * `--model-name-or-path`
  * `--prompt-prefix`
  * `--input-text`
  * `--output-bytes`
  * `--top-k`
  * `--device`

Implementation outline:

```python
def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # encode subparser ...
    # decode subparser ...

    args = parser.parse_args()

    tokenizer, model = load_model(args.model_name_or_path, args.device)

    if args.command == "encode":
        data = open(args.input_bytes, "rb").read()
        text = encode_data_to_text(...)

        with open(args.output_text, "w", encoding="utf-8") as f:
            f.write(text)

    elif args.command == "decode":
        text = open(args.input_text, "r", encoding="utf-8").read()
        data = decode_text_to_data(...)

        with open(args.output_bytes, "wb") as f:
            f.write(data)
```

---

## 9. Testing Plan

### 9.1 Unit tests (simple)

Write a small `tests/` module or inline tests:

1. **Base conversion:**

   * Generate random `bytes` of lengths [1, 16, 100, 1024].
   * For each base in `[2, 4, 8, 16, 32]`, verify:

     ```python
     digits = bytes_to_base_digits(data, base)
     data2 = base_digits_to_bytes(digits, base, len(data))
     assert data2 == data
     ```

2. **Round-trip through LLM codec:**

   * For a small model (e.g. `"sshleifer/tiny-gpt2"`), pick random bytes of 32–64 bytes.
   * Run `encode` then `decode`, ensure output bytes equal input bytes.
   * Do this for multiple bases.

### 9.2 Manual sanity checks

* Use a small prompt prefix like `"Once upon a time, "`.
* Encode a short message ("hello world" bytes).
* Inspect the generated text: ensure it looks at least locally fluent.
* Try different bases and observe fluency vs length tradeoff.

---

## 10. Known Limitations / Edge Cases

The implementation should note (in comments / README):

* **Model & prompt sensitivity**: any change in model weights, tokenizer, or prompt prefix will break decoding.
* **Floating-point stability**: tiny numeric differences could theoretically change rank order; for this PoC we assume stable ordering across runs on the same hardware and environment.
* **Context limits**: if data is large, the number of digits may require more tokens than the context window; for now, we assume small payloads and do not implement chunking.
* **EOS tokens**: we don’t special-case them beyond what’s described; in practice, might want to avoid them for “cleaner” prose.

