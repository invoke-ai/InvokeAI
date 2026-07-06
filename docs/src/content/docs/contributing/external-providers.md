---
title: External Provider Integration
---

This guide covers:

1. Adding a new **external model** (most common; existing provider).
2. Adding a brand-new **external provider** (adapter + config + UI wiring).

## 1) Add a New External Model (Existing Provider)

For provider-backed models (for example, OpenAI or Gemini), the source of truth is
`invokeai/backend/model_manager/starter_models.py`.

### Required model fields

Define a `StarterModel` with:

- `base=BaseModelType.External`
- `type=ModelType.ExternalImageGenerator`
- `format=ModelFormat.ExternalApi`
- `source="external://<provider_id>/<provider_model_id>"`
- `name`, `description`
- `capabilities=ExternalModelCapabilities(...)`
- optional `default_settings=ExternalApiModelDefaultSettings(...)`

Example:

```python
new_external_model = StarterModel(
    name="Provider Model Name",
    base=BaseModelType.External,
    source="external://openai/my-model-id",
    description=(
        "Provider model (external API). "
        "Requires a configured OpenAI API key and may incur provider usage costs."
    ),
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img", "img2img", "inpaint"],
        supports_negative_prompt=False,
        supports_seed=False,
        supports_guidance=False,
        supports_steps=False,
        supports_reference_images=True,
        max_images_per_request=4,
    ),
    default_settings=ExternalApiModelDefaultSettings(
        width=1024,
        height=1024,
        num_images=1,
    ),
)
```

Then append it to `STARTER_MODELS`.

### Required description text

External starter model descriptions must clearly state:

- an API key is required
- usage may incur provider-side costs

### Capabilities must be accurate

These flags directly control UI visibility and request payload fields:

- `supports_negative_prompt`
- `supports_seed`
- `supports_guidance`
- `supports_steps`
- `supports_reference_images`

`supports_steps` is especially important: if `False`, steps are hidden for that model and `steps` is sent as `null`.

### Source string stability

Starter overrides are matched by `source` (`external://provider/model-id`). Keep this stable:

- runtime capability/default overrides depend on it
- installation detection in starter-model APIs depends on it

`STARTER_MODELS` enforces unique `source` values with an assertion.

### Install behavior notes

- External starter models are managed in **External Providers** setup (not the regular Starter Models tab).
- External starter models auto-install when a provider is configured.
- Removing a provider API key removes installed external models for that provider.

## 2) Credentials and Config

External provider API keys are stored separately from `invokeai.yaml`:

- default file: `~/invokeai/api_keys.yaml`
- resolved path: `<INVOKEAI_ROOT>/api_keys.yaml`

Non-secret provider settings (for example base URL overrides) stay in `invokeai.yaml`.

Environment variables are still supported, e.g.:

- `INVOKEAI_EXTERNAL_GEMINI_API_KEY`
- `INVOKEAI_EXTERNAL_OPENAI_API_KEY`

## 3) Add a New Provider (Only If Needed)

If your model uses a provider that is not already integrated:

1. Add config fields in `invokeai/app/services/config/config_default.py`
   `external_<provider>_api_key` and optional `external_<provider>_base_url`.
2. Add provider field mapping in `invokeai/app/api/routers/app_info.py`
   (`EXTERNAL_PROVIDER_FIELDS`).
3. Implement provider adapter in `invokeai/app/services/external_generation/providers/`
   by subclassing `ExternalProvider`.
4. Register the provider in `invokeai/app/api/dependencies.py` when building
   `ExternalGenerationService`.
5. Add starter model entries using `source="external://<provider>/<model-id>"`.
6. Optional UI ordering tweak:
   `invokeai/frontend/web/src/features/modelManagerV2/subpanels/AddModelPanel/ExternalProviders/ExternalProvidersForm.tsx`
   (`PROVIDER_SORT_ORDER`).

## 4) Optional Manual Installation

You can also install external models directly via:

`POST /api/v2/models/install?source=external://<provider_id>/<provider_model_id>`

If omitted, `path`, `source`, and `hash` are auto-populated for external model configs.
Set capabilities conservatively; the external generation service enforces capability checks at runtime.
