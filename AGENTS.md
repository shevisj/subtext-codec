
## System info
OS: Ubuntu 24.04.3 LTS x86_64 
CPU: AMD Ryzen 9 9950X3D
GPU: NVIDIA RTX 5090
RAM: 128 GB

## Project summary
- Repository: subtext — PoC codec that hides binary data inside LLM-generated text by steering token selection via logit rank.
-- Core script: `__main__.py` (Python, torch + transformers). Provides `encode`/`decode` CLI with deterministic replay.
- Metadata header format: `[LLM-CODEC v1; base=<B>; length=<bytes>; top_k=<value|none>]` followed by the generated body (which must include the provided prompt prefix).

## Working notes
- Use the virtual env suggested in README (`uv venv .env --python 3.13`; `uv pip install -r requirements.txt`).
- Keep changes ASCII. Preserve existing instructions and metadata header shape unless intentionally versioning.
- When decoding, parameters (model, prompt prefix, base/top_k) must match the encoding run; mismatches raise errors.
- Tests are under `tests/`; `python -m pytest` runs them. The tiny-model round-trip test skips if the model is unavailable offline.
