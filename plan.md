# Regression Testing Plan: Transformers 5.1.0 + HuggingFace Hub Migration

Below is a change-by-change plan. Each section explains **what changed, why, and how to test it**. Tests are ordered from quickest smoke tests to longer end-to-end runs.

---

## Change 1: `CLIPFeatureExtractor` → `CLIPImageProcessor`

**File:** `invokeai/backend/stable_diffusion/diffusers_pipeline.py`
**Why:** `CLIPFeatureExtractor` was removed in transformers 5.x in favour of `CLIPImageProcessor`.

| # | Test | How |
|---|------|-----|
| 1a | **Import smoke test** | `python -c "from invokeai.backend.stable_diffusion.diffusers_pipeline import StableDiffusionGeneratorPipeline"` — should not raise `ImportError` |
| 1b | **SD 1.5 text-to-image** | In the UI, load any SD 1.5 model and generate an image with a simple prompt (e.g. `"a cat on a couch"`). Confirm the image generates without errors. This exercises the full `StableDiffusionGeneratorPipeline` including the feature extractor type. |

---

## Change 2: `AutoFeatureExtractor` → `AutoImageProcessor` + removed `safe_serialization`

**File:** `invokeai/backend/image_util/safety_checker.py`
**Why:** All vision `FeatureExtractor` classes were removed in transformers 5.x. The `safe_serialization` parameter was also removed from `save_pretrained` (safetensors is now the only format).

| # | Test | How |
|---|------|-----|
| 2a | **Import smoke test** | `python -c "from invokeai.backend.image_util.safety_checker import SafetyChecker"` |
| 2b | **NSFW checker first-time download** | Delete the local cache at `<root>/models/core/convert/stable-diffusion-safety-checker/` if it exists. Enable the NSFW checker in config (`nsfw_checker: true`). Generate any SD 1.5 image. Confirm the safety checker downloads, saves to disk (no `safe_serialization` error), and the image either passes or is correctly blurred. |
| 2c | **NSFW checker cached load** | With the cache from 2b still present, generate another image. Confirm it loads from local path via `AutoImageProcessor.from_pretrained()` without re-downloading. |

---

## Change 3: `T5TokenizerFast` → `T5Tokenizer` (4 files)

**Files:**
- `invokeai/app/invocations/flux_text_encoder.py`
- `invokeai/app/invocations/sd3_text_encoder.py`
- `invokeai/backend/model_manager/load/model_util.py`
- `invokeai/backend/model_manager/load/model_loaders/flux.py`

**Why:** Transformers 5.x unified slow/fast tokenizers — `T5TokenizerFast` no longer exists as a separate class. `T5Tokenizer` now uses the Rust backend by default.

| # | Test | How |
|---|------|-----|
| 3a | **Import smoke tests** | Run each: `python -c "from invokeai.app.invocations.flux_text_encoder import FluxTextEncoderInvocation"`, same for `sd3_text_encoder`, `model_util`, and the flux loader module. |
| 3b | **FLUX text-to-image** | Load a FLUX model (e.g. FLUX.1-dev or schnell). Generate an image with prompt `"a lighthouse on a cliff at sunset"`. This exercises `T5Tokenizer.from_pretrained()` in the loader and the `isinstance(t5_tokenizer, T5Tokenizer)` assertion in the text encoder invocation. |
| 3c | **FLUX with long prompt (truncation path)** | Use a very long prompt (500+ words) with a FLUX model. Check that: the image generates, and the console shows a truncation warning (this exercises the tokenizer's truncation detection logic). |
| 3d | **SD3 text-to-image** | Load an SD3 model. Generate an image with a simple prompt. This covers the SD3 text encoder's T5 tokenization path and its `batch_decode` for truncation warnings. |
| 3e | **Model size calculation** | Load a FLUX or SD3 model and check that the model manager correctly reports the tokenizer memory footprint in the logs (exercises the `isinstance(model, T5Tokenizer)` path in `model_util.py`). No crash = pass. |

---

## Change 4: `configure_http_backend` removed + session-aware metadata fetching

**Files:**
- `invokeai/backend/model_manager/metadata/metadata_base.py` — removed `configure_http_backend` import and call
- `invokeai/backend/model_manager/metadata/fetch/huggingface.py` — removed `configure_http_backend`, added `_model_info_via_session()` fallback, added `_has_custom_session` flag

**Why (root cause chain):**
1. `transformers>=5.1.0` pulls in `huggingface_hub>=1.0.0` as a dependency.
2. `huggingface_hub` 1.0 switched its HTTP backend from `requests` to `httpx` and removed the `configure_http_backend()` function entirely.
3. InvokeAI called `configure_http_backend(backend_factory=lambda: session)` in two places to inject a custom `requests.Session` — this was used in production for `download_urls()` and, critically, in tests to inject a `TestSession` with mock HTTP adapters so tests could run without real network calls.
4. Simply removing the calls fixed the import crash in production (since `HfApi()` now uses `httpx` internally and works fine for real HTTP). However, it broke the test suite: `HfApi().model_info()` now bypasses the mock `requests.TestSession` entirely and hits the real HuggingFace API, causing `RepositoryNotFoundError` for test-only repos like `InvokeAI-test/textual_inversion_tests`.
5. The fix: `HuggingFaceMetadataFetch` now tracks whether a custom session was injected (`_has_custom_session`). When true, `from_id()` calls a new `_model_info_via_session()` method that uses the injected `requests.Session` to query the HF API directly (matching the URL patterns the test mocks expect). When false (production), it uses `HfApi()` as before.

| # | Test | How |
|---|------|-----|
| 4a | **Import smoke test** | `python -c "from invokeai.backend.model_manager.metadata.fetch import HuggingFaceMetadataFetch"` |
| 4b | **Automated test suite** | `pytest tests/app/services/model_install/test_model_install.py -x -v` — all 19 tests should pass, especially `test_heuristic_import_with_type`, `test_huggingface_install`, and `test_huggingface_repo_id` which depend on mock HF API responses via the injected session. |
| 4c | **Install a model from HuggingFace** | In the UI's Model Manager, add a model by HuggingFace repo ID (e.g. `stabilityai/sd-turbo`). Confirm the metadata (name, description, tags) is correctly fetched and displayed, and the model downloads successfully. This exercises the production `HfApi().model_info()` path, plus `hf_hub_url()` and `download_urls()`. |
| 4d | **Browse HF model metadata** | If the UI has a model info/details view, open it for an already-installed HF model and confirm metadata fields are populated. |

---

## Change 5: `HfFolder` → `huggingface_hub.get_token()`

**File:** `invokeai/app/services/model_install/model_install_default.py`
**Why:** `HfFolder` was removed in `huggingface_hub` 1.0+. The replacement is the top-level `get_token()` function.

| # | Test | How |
|---|------|-----|
| 5a | **Import smoke test** | `python -c "from invokeai.app.services.model_install.model_install_default import ModelInstallService"` |
| 5b | **Install gated model (with token)** | If you have a HuggingFace account with an access token cached (`huggingface-cli login`), try installing a gated model (e.g. `black-forest-labs/FLUX.1-dev`). Confirm the token is automatically injected and the download succeeds. |
| 5c | **Install public model (no token)** | Without explicit token, install a public model. Confirm `get_token()` returns `None` gracefully and the install proceeds. |

---

## Change 6: `transformers>=5.1.0` override for compel

**File:** `pyproject.toml` — `override-dependencies`
**Why:** `compel==2.1.1` requires `transformers ~= 4.25` (`<5.0`). The uv override forces past this constraint.

| # | Test | How |
|---|------|-----|
| 6a | **SD 1.5 prompt weights** | Generate an image with weighted prompts: `"a (red:1.5) car on a (blue:0.5) road"`. Compare to an unweighted `"a red car on a blue road"`. The weighted version should show noticeably more red and less blue. This is the core compel functionality. |
| 6b | **SD 1.5 negative prompts** | Generate with prompt `"a photo of a dog"` and negative prompt `"blurry, low quality"`. Confirm it generates without crash. |
| 6c | **SDXL prompt weights** | Same as 6a but with an SDXL model. SDXL uses a different compel path (`SDXLCompelPromptInvocation`). |
| 6d | **Prompt blending (compel syntax)** | Try compel blend syntax if supported: `"a photo of a cat".blend("a photo of a dog", 0.5)` or `("a cat", "a dog").blend(0.5, 0.5)`. This exercises deeper compel internals. |

---

## Change 7: T5 shared-weight assertion → `model.tie_weights()`

**Files:**
- `invokeai/backend/model_manager/load/model_loaders/flux.py` — `_load_state_dict_into_t5()` classmethod
- `invokeai/backend/quantization/scripts/quantize_t5_xxl_bnb_llm_int8.py` — `load_state_dict_into_t5()` function

**Why (root cause chain):**
1. T5 models have a shared weight: `model.shared.weight` and `model.encoder.embed_tokens.weight` should refer to the same tensor.
2. In **transformers 4.x**, this sharing was implemented as a Python object alias — both attributes literally pointed to the same `nn.Parameter` object, so `a is b` was `True`.
3. In **transformers 5.x**, weight tying is implemented at the **parameter level** via `_tie_weights()` / `tie_weights()`. The two attributes may be distinct `nn.Parameter` objects that are kept in sync by the framework, so `a is b` can be `False`.
4. InvokeAI calls `model.load_state_dict(state_dict, strict=False, assign=True)`. The `assign=True` flag replaces parameters in-place rather than copying data into existing tensors. This severs even the parameter-level tie that transformers 5.x establishes.
5. The old code then asserted `model.encoder.embed_tokens.weight is model.shared.weight`, which was guaranteed `True` in 4.x but fails in 5.x after `assign=True`.
6. **Fix:** Replace the identity assertion with `model.tie_weights()`, which re-establishes the tie regardless of how it is internally implemented. This is forward-compatible and is the officially recommended approach.

| # | Test | How |
|---|------|-----|
| 7a | **Import smoke test** | `python -c "from invokeai.backend.model_manager.load.model_loaders.flux import FluxBnbQuantizednf4bCheckpointModel"` |
| 7b | **FLUX text-to-image** | Load a FLUX model and generate an image. This is the primary code path that calls `_load_state_dict_into_t5()`. The generation should complete without `AssertionError`. (Same as test 3b — this change and Change 3 are both exercised together.) |
| 7c | **FLUX BnB quantized model** | If you have a BnB-quantized FLUX model, load and generate with it. This exercises the `FluxBnbQuantizednf4bCheckpointModel` loader which also calls `_load_state_dict_into_t5()`. |
| 7d | **Quantize script (manual)** | If you need to re-quantize a T5 model: run `quantize_t5_xxl_bnb_llm_int8.py` and confirm it completes without assertion errors. |

---

## Change 8: `HFTokenHelper.get_status()` — null token guard for `get_token_permission()`

**File:** `invokeai/app/api/routers/model_manager.py`
**Why (root cause chain):**
1. `HFTokenHelper.get_status()` calls `huggingface_hub.get_token_permission(huggingface_hub.get_token())` to check whether a valid HF token is present.
2. When no token is configured, `get_token()` returns `None`.
3. In **huggingface_hub <1.0**, `get_token_permission(None)` returned a falsy value, so the code fell through to `return HFTokenStatus.INVALID` — correct behavior.
4. In **huggingface_hub 1.0+**, `get_token_permission(None)` **raises an exception** (it now validates the input and rejects `None`).
5. The `except Exception` catch returned `HFTokenStatus.UNKNOWN`, which the frontend interprets as a network error, showing the misleading message: *"Unable to Verify HF Token — Unable to verify HuggingFace token. This is likely due to a network error."*
6. **Fix:** Check `get_token()` for `None` first and return `INVALID` immediately, before ever calling `get_token_permission()`. This restores the correct "no token" UI message.

| # | Test | How |
|---|------|-----|
| 8a | **No token → INVALID status** | Remove/rename your HF token file (`~/.cache/huggingface/token`), clear `$env:HF_TOKEN`, restart InvokeAI. The UI should show the proper "no token" message, **not** the "unable to verify / network error" message. |
| 8b | **Valid token → VALID status** | Restore your token (`huggingface-cli login`), restart InvokeAI. The UI should show the token as valid. |
| 8c | **Install gated model without token** | With no token, try to install a gated model (e.g. `black-forest-labs/FLUX.1-dev`). The UI should clearly indicate a token is needed, not a network error. |

---

## Automated Test Suite

| # | Command | What it covers |
|---|---------|----------------|
| A1 | `pytest ./tests -x -m "not slow"` | Run the full fast test suite. Any existing tests that touch model loading, metadata, or imports will catch regressions. The `-x` flag stops on first failure for quick feedback. |
| A2 | `pytest ./tests -x -m "slow"` | Run slow tests (if you have models available). These likely include integration tests. |

---

## Quick Smoke Test Script (all imports at once)

Run this to verify none of the changed files crash on import:

```python
python -c "
from invokeai.backend.stable_diffusion.diffusers_pipeline import StableDiffusionGeneratorPipeline
from invokeai.backend.image_util.safety_checker import SafetyChecker
from invokeai.app.invocations.flux_text_encoder import FluxTextEncoderInvocation
from invokeai.app.invocations.sd3_text_encoder import Sd3TextEncoderInvocation
from invokeai.backend.model_manager.load.model_util import calc_model_size_by_data
from invokeai.backend.model_manager.load.model_loaders.flux import FluxBnbQuantizednf4bCheckpointModel
from invokeai.backend.model_manager.metadata.metadata_base import HuggingFaceMetadata
from invokeai.backend.model_manager.metadata.fetch import HuggingFaceMetadataFetch
from invokeai.app.services.model_install.model_install_default import ModelInstallService
print('All imports OK')
"
```

If this prints `All imports OK`, you've passed the baseline. Then proceed to the UI-based tests in priority order: **6a → 3b/7b → 3d → 4b → 5c → 2b → 1b**.
