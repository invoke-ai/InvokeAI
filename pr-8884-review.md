# Code Review: PR #8884 - External Models (Gemini & OpenAI GPT Image)

**PR:** https://github.com/invoke-ai/InvokeAI/pull/8884
**Author:** CypherNaught-0x
**Head SHA:** `7671304195518fe010b6e9b080f7a30b752015ca`
**Scope:** 79 files changed, +4619 / -143 lines

---

## Summary

This PR adds support for external image generation provider APIs (Google Gemini and OpenAI GPT Image). It includes:
- A new `ExternalGenerationService` with provider adapters for Gemini and OpenAI
- New `ExternalApiModelConfig` model type and install flow via `external://` source URIs
- API key management endpoints and config persistence
- Frontend graph builder, model selection UI, and provider badge display
- Tests and documentation

---

## Issues Found

### High Confidence (score >= 80)

**1. Debug file dump writes request payloads and images to disk unconditionally** (score: 85)

In `invokeai/app/services/external_generation/providers/gemini.py`, the methods `_dump_debug_payload` and `_dump_debug_image` are called on **every single Gemini API request** with no flag or config option to disable them. Both methods are annotated with `TODO: remove debug payload dump once Gemini is stable` / `TODO: remove debug image dump once Gemini is stable`.

They write:
- Full JSON request/response payloads (including base64-encoded image data) to `outputs/external_debug/gemini/{label}_{uuid}.json`
- Decoded PNG images to `outputs/external_debug/gemini/decoded_{uuid}.png`

This causes **unbounded disk growth** and silently persists all user generation data (prompts, images, API responses) to disk with no user knowledge or consent. There is no way to disable this behavior -- it runs whenever `outputs_path` is set, which is always the case in normal operation.

https://github.com/invoke-ai/InvokeAI/blob/7671304195518fe010b6e9b080f7a30b752015ca/invokeai/app/services/external_generation/providers/gemini.py (methods `_dump_debug_payload` and `_dump_debug_image`)

---

### Moderate Confidence (score 50-79)

**2. Service layer imports from API layer -- inverted dependency** (score: 70)

In `invokeai/app/services/external_generation/external_generation_default.py`, the method `_refresh_model_capabilities` does:

```python
from invokeai.app.api.dependencies import ApiDependencies
record = ApiDependencies.invoker.services.model_manager.store.get_model(request.model.key)
```

No other service in the codebase imports from `invokeai.app.api.dependencies`. All other services receive their dependencies via constructor injection through `InvocationServices`. This is an architectural violation that makes the service harder to test in isolation and creates a hidden coupling between the service and API layers.

**3. `ExternalModelSource` incorrectly mapped to `ModelSourceType.Url`** (score: 50)

In `invokeai/app/services/model_install/model_install_common.py`:

```python
MODEL_SOURCE_TO_TYPE_MAP = {
    ...
    ExternalModelSource: ModelSourceType.Url,
}
```

`ExternalModelSource` is not a URL source. There is no `ModelSourceType.External` enum value in `taxonomy.py`. This means external models get recorded as `Url`-type sources in the database, which is semantically incorrect and could cause issues in any code that branches on `source_type`.

**4. `_apply_external_provider_update` directly mutates `model_fields_set`** (score: 50)

In `invokeai/app/api/routers/app_info.py`:

```python
for config in (runtime_config, file_config):
    config.update_config(updates)
    for field_name, value in updates.items():
        if value is None:
            config.model_fields_set.discard(field_name)
```

This directly mutates the `model_fields_set` of the global singleton `InvokeAIAppConfig`, bypassing Pydantic's field-tracking internals. Concurrent requests to `set_external_provider_config` or `reset_external_provider_config` could race on this shared mutable set.

**5. Duplicate key conflict on reinstall of external model** (score: 50)

In `invokeai/app/services/model_install/model_install_default.py`, `_register_external_model` generates a deterministic key via `slugify(f"{provider_id}-{provider_model_id}")`. Installing the same external model twice produces the same key. While the DB layer catches this with `DuplicateModelException`, there is no proactive check or update-if-exists logic, resulting in an unhelpful error for the user.

**6. `setattr` used on Pydantic models instead of `model_copy`** (score: 50)

In `invokeai/app/api/routers/model_manager.py`, `list_model_records` uses `setattr(model, "capabilities", ...)` and `setattr(model, "default_settings", ...)` on Pydantic model instances. Pydantic v2 models may not support direct attribute mutation without `validate_assignment = True`. The PR itself uses `model_copy(update=...)` correctly in other places (e.g., `_apply_starter_overrides` in `external_generation_default.py`), so this is inconsistent.

---

### Low Confidence / Nitpicks (score < 50)

**7. API key potentially leaked in error messages** (score: 40)

In `gemini.py`, the Gemini API key is passed as a URL query parameter (`params={"key": api_key}`), and error handling includes raw `response.text` in exception messages. If the API echoes back the request URL in error responses, the key could be exposed in logs or UI.

**8. Duplicate ratio utility functions** (score: 40)

Functions `_parse_ratio`, `_gcd`, `_format_aspect_ratio`, `_select_closest_ratio` are duplicated between `external_generation_default.py` and `providers/gemini.py`.

**9. `InvokeAIAppConfig` docstring not updated** (score: 25)

The class has an exhaustive `Attributes:` docstring listing all 40+ config fields. The 4 new fields (`external_gemini_api_key`, `external_openai_api_key`, `external_gemini_base_url`, `external_openai_base_url`) are not added to it.

**10. Missing section comment in `factory.py` `AnyModelConfig` union** (score: 25)

Every group of model configs in the union has a section comment (e.g., `# Main (Pipeline) - diffusers format`). `ExternalApiModelConfig` is added without one.

---

### False Positives Identified

- **`positivePrompt` node value never set in `buildExternalGraph`**: FALSE POSITIVE. The prompt value is injected via `prepareLinearUIBatch` batch data mechanism, same as all other graph builders (SD1, FLUX, Z-Image, etc.). The `string` node intentionally starts empty and gets its value from batch processing.

- **`reidentify_model` `hasattr`/`setattr` dead code for external models**: Low concern. While `from_model_on_disk` raises for external models before reaching this code, the defensive guard doesn't cause harm and protects against future changes.
