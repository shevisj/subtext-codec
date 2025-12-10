## Project summary
- Repository: subtext-codec -- PoC codec that hides binary data inside LLM-generated text by steering token selection via logit rank.
-- Core script: `__main__.py` (Python, torch + transformers). Provides `encode`/`decode` CLI with deterministic replay.
- Dynamic base: per-token base derived from `top_p` (cumulative logit mass after optional `top_k` truncation). Legacy v1 keys still store a fixed `base`.

## Working notes
- Use the virtual env suggested in README (`uv venv .env --python 3.13`; `uv pip install -r requirements.txt`).
- Keep changes ASCII. Preserve existing instructions and metadata header shape unless intentionally versioning.
- When decoding, parameters (model, prompt prefix, top_p/top_k or legacy base/top_k) must match the encoding run; mismatches raise errors.
- Tests are under `tests/`; `python -m pytest` runs them. The tiny-model round-trip test skips if the model is unavailable offline.

## Repo-wide style and guardrails
- Prefer standard library and existing dependencies; if a new dependency is unavoidable, add it to `pyproject.toml`, regenerate `uv.lock`, and pin it in `requirements.txt`.
- Keep Python modules import-safe: avoid heavyweight work at import time, gate side effects behind `if __name__ == "__main__":`, and preserve deterministic seeding helpers for reproducibility.
- Maintain explicit typing and small, pure helpers where possible; raise clear `ValueError`/`RuntimeError` for user-facing misconfiguration instead of silent fallbacks.
- For model usage, stay in inference mode (`model.eval()`, `torch.no_grad()`), respect `top_p`/`top_k` constraints (and legacy `base` for v1 keys), and ensure prompt-prefix handling remains symmetrical between encode/decode.
- Default tests should stay fast and offline-friendly; mark network/model downloads as `@pytest.mark.slow` and skip gracefully when prerequisites are missing.
- Documentation and samples should keep the prompt prefix and metadata header in sync with the code paths, and any new examples should round-trip with the current CLI flags.
