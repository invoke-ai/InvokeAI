# External Provider Integration

This guide covers how to add new external image generation providers and model configs.

## Provider Adapter Steps

1) Create a provider adapter in `invokeai/app/services/external_generation/providers/` that inherits from `ExternalProvider`.
2) Implement `is_configured()` using `InvokeAIAppConfig` fields, and `generate()` to map `ExternalGenerationRequest` to the provider API.
3) Use helpers from `invokeai/app/services/external_generation/image_utils.py` for image encoding/decoding.
4) Raise `ExternalProviderRequestError` on non-200 responses or empty payloads.
5) Register the provider in `invokeai/app/api/dependencies.py` when building the `ExternalGenerationService` registry.

## Config + Env Vars

Add provider API keys to `InvokeAIAppConfig` with the `INVOKEAI_` prefix:

- `INVOKEAI_EXTERNAL_GEMINI_API_KEY`
- `INVOKEAI_EXTERNAL_OPENAI_API_KEY`

These can also be set in `invokeai.yaml` under `external_gemini_api_key` and `external_openai_api_key`.

## Example External Model Config

External models are stored in the model manager like any other config. This example can be used as the `config` payload
for `POST /api/v2/models/install?source=external://openai/gpt-image-1`:

```json
{
  "key": "openai_gpt_image_1",
  "name": "OpenAI GPT-Image-1",
  "base": "external",
  "type": "external_image_generator",
  "format": "external_api",
  "provider_id": "openai",
  "provider_model_id": "gpt-image-1",
  "capabilities": {
    "modes": ["txt2img", "img2img", "inpaint"],
    "supports_negative_prompt": true,
    "supports_seed": true,
    "supports_guidance": true,
    "supports_reference_images": false,
    "max_images_per_request": 1
  },
  "default_settings": {
    "width": 1024,
    "height": 1024,
    "steps": 30
  },
  "tags": ["external", "openai"],
  "is_default": false
}
```

Notes:

- `path`, `source`, and `hash` will auto-populate if omitted.
- Set `capabilities` conservatively; the external generation service enforces them at runtime.
