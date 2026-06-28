# Second GPU Text Encoder Feature

## Goal

Add an optional runtime setting that keeps the main generation model on the app's configured CUDA GPU while running text encoders on the other CUDA GPU.

Example behavior:

- App device `cuda:0` -> text encoders use `cuda:1`
- App device `cuda:1` -> text encoders use `cuda:0`
- App device `cuda` -> normalized by InvokeAI, then the opposite CUDA device is selected
- Fewer than 2 CUDA GPUs -> setting is unavailable/disabled

This is intentionally a small model-loader/cache feature, not a full multi-GPU scheduler.

## User-Facing Behavior

- Advanced settings shows a toggle: `Use Second GPU for Text Encoder`
- Toggle is disabled unless the backend reports at least two CUDA GPUs and the app is using CUDA
- No CPU option is added
- Setting is persisted through `/api/v1/app/runtime_config`
- Manual `Clear Model Cache` still clears everything

## First Upstream Concern: Dynamic Toggle Behavior

The toggle should actively change the loaded model state instead of only changing future model loads.

Expected behavior:

- Toggle on:
  - Persist `use_second_gpu_for_text_encoder=true`
  - Drop any cached text encoder entries that were loaded on the main device
  - Trigger/prewarm loading of the currently selected text encoder onto the second CUDA device, if the active workflow/model selection exposes a current encoder
  - Keep the denoise model resident on the main CUDA device when possible

- Toggle off:
  - Persist `use_second_gpu_for_text_encoder=false`
  - Drop protected second-GPU text encoder cache entries
  - Let the next generation reload the encoder through the normal stock device path
  - Ideally unload the second-GPU encoder immediately so the GPU memory visibly frees

Implementation options:

- Minimum safe version:
  - On runtime config update, backend drops cached text encoder cache records affected by the setting change
  - UI shows the changed setting immediately
  - Next generation reloads the encoder onto the correct device

- Better interactive version:
  - Backend exposes a small endpoint to refresh/prewarm current text encoder models after the setting changes
  - Frontend calls it after the toggle saves
  - If no current encoder can be inferred, fallback to cache-drop-only behavior

This should be addressed before proposing upstream, because users will expect the toggle to have immediate, visible effect.

## Backend Config/API Changes

### `invokeai/app/services/config/config_default.py`

Add:

```py
use_second_gpu_for_text_encoder: bool = Field(
    default=False,
    description="When at least two CUDA GPUs are available, run text encoder models on the CUDA device that is not the main execution device.",
)
```

This belongs near the existing `device` setting.

### `invokeai/app/api/routers/app_info.py`

Expose CUDA device count through `get_app_deps()`:

```py
deps["CUDA Devices"] = str(cuda_device_count)
deps[f"CUDA Device {device_index}"] = torch.cuda.get_device_name(device_index)
```

Allow runtime updates:

```py
use_second_gpu_for_text_encoder: bool | None = Field(default=None, ...)
```

## Model Routing Changes

### `invokeai/backend/model_manager/load/load_default.py`

Add text encoder detection for:

- `SubModelType.TextEncoder`
- `SubModelType.TextEncoder2`
- `SubModelType.TextEncoder3`
- `ModelType.CLIPEmbed`
- `ModelType.T5Encoder`
- `ModelType.Qwen3Encoder`
- `ModelType.QwenVLEncoder`
- `ModelType.TextLLM`

When the setting is enabled and two CUDA GPUs exist, return the CUDA device whose index differs from the app's main execution device.

The current local version also protects CUDA-resident models from automatic eviction while this mode is active:

```py
prevent_auto_evict = (
    self._app_config.use_second_gpu_for_text_encoder and effective_execution_device.type == "cuda"
)
```

## Cache Changes Needed

### `invokeai/backend/model_manager/load/model_cache/cache_record.py`

Add:

```py
prevent_auto_evict: bool = False
```

Protected entries are skipped by automatic cleanup, but explicit model reloads and manual cache clearing may still remove them.

### `invokeai/backend/model_manager/load/model_cache/model_cache.py`

Add `prevent_auto_evict` to `ModelCache.put()`.

Automatic VRAM offload skips protected entries.

Automatic RAM cache eviction skips protected entries.

Manual cache clear can bypass protection:

```py
make_room(bytes_needed, preserve_auto_evict_protected=False)
```

Make VRAM accounting device-aware so loading on `cuda:0` does not try to free models resident on `cuda:1`.

Optional split-GPU cache sizing:

- When split-GPU mode is enabled, calculate the RAM cache cap using total VRAM across CUDA devices
- Use a larger RAM fraction so paired denoise + encoder models can remain cached

### `invokeai/app/services/model_manager/model_manager_default.py`

Pass the config flag into `ModelCache`:

```py
use_multi_cuda_ram_cache=app_config.use_second_gpu_for_text_encoder
```

### `invokeai/app/api/routers/model_manager.py`

Manual empty-cache endpoint should bypass protection:

```py
ram_cache.make_room(1000 * 2**30, preserve_auto_evict_protected=False)
```

## Frontend Changes

### New Component

`invokeai/frontend/web/src/features/parameters/components/Advanced/ParamUseSecondGpuForTextEncoder.tsx`

Responsibilities:

- Read `CUDA Devices` from `useGetAppDepsQuery()`
- Read current runtime config from `useGetRuntimeConfigQuery()`
- Save changes with `useUpdateRuntimeConfigMutation()`
- Disable switch if:
  - runtime config is not loaded
  - user cannot edit runtime config
  - CUDA device count is below 2
  - app device is not CUDA

### Advanced Accordion

`invokeai/frontend/web/src/features/settingsAccordions/components/AdvancedSettingsAccordion/AdvancedSettingsAccordion.tsx`

Render the new toggle once globally in Advanced settings. Do not tie it to specific model picker visibility.

### API Schema

`invokeai/frontend/web/src/services/api/schema.ts`

Add `use_second_gpu_for_text_encoder` to:

- `InvokeAIAppConfig`
- `UpdateAppGenerationSettingsRequest`

If preparing for upstream, prefer regenerating OpenAPI/types if the official workflow supports it.

## Local-Only Change Not For Upstream

`update.bat` was changed locally so this repo can build with pnpm 10 even when PATH has pnpm 11.

Do not include that in an upstream PR unless upstream wants Windows helper changes.

## Testing Checklist

Test with app device set to `cuda:0`:

- Toggle off: behavior matches stock InvokeAI
- Toggle on: text encoder spikes and remains on GPU 1
- Denoise remains on GPU 0
- Running a second generation should reuse the resident encoder

Test with app device set to `cuda:1`:

- Toggle on: text encoder moves to GPU 0
- Denoise remains on GPU 1
- Loading encoder should not evict denoise
- Loading denoise should not evict encoder

Test encoder families:

- FLUX T5
- SD3 T5
- CLIP embed paths
- Qwen3 / Z-Image
- Qwen VL / Qwen Image
- Any text LLM path that uses `ModelType.TextLLM`

Test cache controls:

- Manual Clear Model Cache should remove both models
- Switching the toggle should require cache reload/new model loads to reflect the device assignment
- Fewer than two CUDA devices should disable the UI control

## Upstream PR Shape

For upstream, create a clean branch from `upstream/main` and bring over only:

- config/API setting
- model loader routing
- cache/device accounting/protection
- Advanced UI toggle
- generated schema/type updates

Leave out:

- Batch+ custom page changes
- local `update.bat`
- runtime/install wrapper changes
- unrelated object serializer cleanup
