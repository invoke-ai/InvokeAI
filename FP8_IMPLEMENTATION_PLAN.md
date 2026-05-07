# FP8 Layerwise Casting - Implementation

## Summary

Add per-model `fp8_storage` option to model default settings that enables diffusers' `enable_layerwise_casting()` to store weights in FP8 (`float8_e4m3fn`) while casting to fp16/bf16 during inference. This reduces VRAM usage by ~50% per model with minimal quality loss.

Supported: SD1/SD2/SDXL/SD3, Flux, Flux2, CogView4, Z-Image, VAE (diffusers-based), ControlNet, T2IAdapter.
Not applicable: Text Encoders, LoRA, GGUF, BnB, custom classes.

## Related Issues / Discussions

- https://github.com/invoke-ai/InvokeAI/issues/7148
- Based on approach from https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/14031
- Uses diffusers' native `enable_layerwise_casting()` (available in diffusers 0.36.0)

## QA Instructions

1. Set `fp8_storage: true` in a model's `default_settings` (via API or Model Manager UI)
2. Load the model and generate an image
3. Verify VRAM usage is reduced compared to normal loading
4. Verify image quality is acceptable (minimal degradation expected)
5. Verify Text Encoders are NOT affected (excluded by submodel type filter)
6. Verify non-CUDA devices gracefully ignore the setting

### Test Matrix

- [ ] SD1.5 Diffusers with `fp8_storage=true` - load and generate
- [ ] SDXL Diffusers with `fp8_storage=true` - load and generate
- [ ] Flux Diffusers with `fp8_storage=true` - load and generate
- [ ] Flux2 Diffusers with `fp8_storage=true` - load and generate
- [ ] CogView4 with `fp8_storage=true` - load and generate
- [ ] Z-Image Diffusers with `fp8_storage=true` - load and generate
- [ ] VAE with `fp8_storage=true` - check quality
- [ ] ControlNet with `fp8_storage=true` - load and generate
- [ ] VRAM comparison: with vs. without `fp8_storage`
- [ ] Image quality comparison: FP8 vs fp16/bf16
- [ ] MPS/CPU: verify `fp8_storage` is silently ignored
- [ ] Flux Checkpoint (custom class): verify FP8 is gracefully skipped (not a ModelMixin)
- [ ] Text Encoder submodels: verify FP8 is NOT applied
- [ ] GGUF/BnB models: verify FP8 is gracefully skipped

## Checklist

- [x] _The PR has a short but descriptive title, suitable for a changelog_
- [ ] _Tests added / updated (if applicable)_
- [ ] _Changes to a redux slice have a corresponding migration_
- [ ] _Documentation added / updated (if applicable)_
- [ ] _Updated `What's New` copy (if doing a release after this PR)_

---

## All Changed Files

### Backend - Model Configs

#### `invokeai/backend/model_manager/configs/main.py` (Modified)

Added `fp8_storage` field to `MainModelDefaultSettings`:

```diff
     cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")
+    fp8_storage: bool | None = Field(
+        default=None,
+        description="Store weights in FP8 to reduce VRAM usage (~50% savings). Weights are cast to compute dtype during inference.",
+    )
```

#### `invokeai/backend/model_manager/configs/controlnet.py` (Modified)

Added `fp8_storage` field to `ControlAdapterDefaultSettings`:

```diff
 class ControlAdapterDefaultSettings(BaseModel):
     preprocessor: str | None
+    fp8_storage: bool | None = Field(
+        default=None,
+        description="Store weights in FP8 to reduce VRAM usage (~50% savings). Weights are cast to compute dtype during inference.",
+    )
     model_config = ConfigDict(extra="forbid")
```

### Backend - Model Loading

#### `invokeai/backend/model_manager/load/load_default.py` (Modified)

Added two helper methods to `ModelLoader` base class:

- `_should_use_fp8(config, submodel_type)` - Checks if FP8 should be applied:
  - Returns `False` if not CUDA
  - Returns `False` for excluded submodel types (TextEncoder, TextEncoder2, TextEncoder3, Tokenizer, Tokenizer2, Tokenizer3, Scheduler, SafetyChecker)
  - Returns `True` if `config.default_settings.fp8_storage is True`

- `_apply_fp8_layerwise_casting(model, config, submodel_type)` - Applies FP8 if conditions met:
  - Checks `_should_use_fp8()` first
  - Only applies to `diffusers.ModelMixin` instances (gracefully skips custom classes)
  - Calls `model.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=self._torch_dtype)`
  - Logs info message on success

#### `invokeai/backend/model_manager/load/model_loaders/generic_diffusers.py` (Modified)

Added `_apply_fp8_layerwise_casting` call after `from_pretrained()`. This covers T2IAdapter and other generic diffusers models.

```diff
+        result = self._apply_fp8_layerwise_casting(result, config, submodel_type)
         return result
```

#### `invokeai/backend/model_manager/load/model_loaders/stable_diffusion.py` (Modified)

Added FP8 call in `_load_model()` after `from_pretrained()`:

```diff
+        result = self._apply_fp8_layerwise_casting(result, config, submodel_type)
         return result
```

Added FP8 calls in `_load_from_singlefile()` for both the requested submodel and cached submodels:

```diff
             if submodel := getattr(pipeline, subtype.value, None):
+                self._apply_fp8_layerwise_casting(submodel, config, subtype)
                 self._ram_cache.put(get_model_cache_key(config.key, subtype), model=submodel)
-        return getattr(pipeline, submodel_type.value)
+        result = getattr(pipeline, submodel_type.value)
+        result = self._apply_fp8_layerwise_casting(result, config, submodel_type)
+        return result
```

#### `invokeai/backend/model_manager/load/model_loaders/flux.py` (Modified)

Added FP8 calls to:
- `FluxDiffusersModel._load_model()` - after `from_pretrained()`
- `Flux2DiffusersModel._load_model()` - after `from_pretrained()`
- `Flux2VAEDiffusersLoader._load_model()` - after `from_pretrained()`
- `Flux2VAELoader._load_model()` - after VAE loading

NOT applied to (gracefully skipped via `isinstance(model, ModelMixin)` check):
- `FluxCheckpointModel` - custom Flux class
- `FluxGGUFCheckpointModel` - GGUF format
- `FluxBnbQuantizednf4bCheckpointModel` - BnB quantized
- `FluxVAELoader` - custom AutoEncoder class

#### `invokeai/backend/model_manager/load/model_loaders/cogview4.py` (Modified)

Added FP8 call after `from_pretrained()`:

```diff
+        result = self._apply_fp8_layerwise_casting(result, config, submodel_type)
         return result
```

#### `invokeai/backend/model_manager/load/model_loaders/controlnet.py` (Modified)

Added FP8 call after `from_single_file()` for checkpoint ControlNets:

```diff
-            return ControlNetModel.from_single_file(
+            result = ControlNetModel.from_single_file(
                 config.path,
                 torch_dtype=self._torch_dtype,
             )
+            result = self._apply_fp8_layerwise_casting(result, config, submodel_type)
+            return result
```

#### `invokeai/backend/model_manager/load/model_loaders/vae.py` (Modified)

Added FP8 call after `from_single_file()` for checkpoint VAEs:

```diff
-            return AutoencoderKL.from_single_file(
+            result = AutoencoderKL.from_single_file(
                 config.path,
                 torch_dtype=self._torch_dtype,
             )
+            result = self._apply_fp8_layerwise_casting(result, config, submodel_type)
+            return result
```

#### `invokeai/backend/model_manager/load/model_loaders/z_image.py` (Modified)

Added FP8 call after `from_pretrained()` in `ZImageDiffusersModel`:

```diff
+        result = self._apply_fp8_layerwise_casting(result, config, submodel_type)
         return result
```

### Frontend - API Schema

#### `invokeai/frontend/web/src/services/api/schema.ts` (Modified)

Added `fp8_storage` field to both TypeScript type definitions:

- `ControlAdapterDefaultSettings.fp8_storage?: boolean | null`
- `MainModelDefaultSettings.fp8_storage?: boolean | null`

### Frontend - Hooks

#### `invokeai/frontend/web/src/features/modelManagerV2/hooks/useMainModelDefaultSettings.ts` (Modified)

Added `fp8Storage` to form defaults:

```typescript
fp8Storage: {
  isEnabled: !isNil(modelConfig?.default_settings?.fp8_storage),
  value: modelConfig?.default_settings?.fp8_storage ?? false,
},
```

#### `invokeai/frontend/web/src/features/modelManagerV2/hooks/useControlAdapterModelDefaultSettings.ts` (Modified)

Added same `fp8Storage` defaults pattern.

### Frontend - Components

#### `invokeai/frontend/web/src/features/modelManagerV2/subpanels/ModelPanel/MainModelDefaultSettings/DefaultFp8Storage.tsx` (New)

New component with `SettingToggle` + `Switch`, following `DefaultVaePrecision` pattern. Renders an FP8 Storage toggle with `InformationalPopover`.

#### `invokeai/frontend/web/src/features/modelManagerV2/subpanels/ModelPanel/ControlAdapterModelDefaultSettings/DefaultFp8StorageControlAdapter.tsx` (New)

Same pattern as `DefaultFp8Storage` but typed for `ControlAdapterModelDefaultSettingsFormData`.

#### `invokeai/frontend/web/src/features/modelManagerV2/subpanels/ModelPanel/MainModelDefaultSettings/MainModelDefaultSettings.tsx` (Modified)

- Added `DefaultFp8Storage` import
- Added `fp8Storage: FormField<boolean>` to `MainModelDefaultSettingsFormData`
- Added `fp8_storage: data.fp8Storage.isEnabled ? data.fp8Storage.value : null` to `onSubmit` body
- Added `<DefaultFp8Storage control={control} name="fp8Storage" />` to render

#### `invokeai/frontend/web/src/features/modelManagerV2/subpanels/ModelPanel/ControlAdapterModelDefaultSettings/ControlAdapterModelDefaultSettings.tsx` (Modified)

- Added `DefaultFp8StorageControlAdapter` import
- Added `fp8Storage: FormField<boolean>` to form data type
- Added `fp8_storage` to `onSubmit` body
- Added component to render grid

### Frontend - Translations & Popover

#### `invokeai/frontend/web/public/locales/en.json` (Modified)

Added translation key:
```json
"fp8Storage": "FP8 Storage (Save VRAM)"
```

Added InformationalPopover content:
```json
"fp8Storage": {
    "heading": "FP8 Storage",
    "paragraphs": [
        "Stores model weights in FP8 format in VRAM, reducing memory usage by approximately 50% compared to FP16.",
        "During inference, weights are cast layer-by-layer to the compute precision (FP16/BF16), so image quality is preserved. Works on all CUDA GPUs."
    ]
}
```

#### `invokeai/frontend/web/src/common/components/InformationalPopover/constants.ts` (Modified)

Added `'fp8Storage'` to `Feature` union type.

---

## How It Works

1. User sets `fp8_storage: true` in a model's default settings (via Model Manager UI)
2. On load, `_should_use_fp8()` checks: CUDA device? Not a text encoder? `fp8_storage` enabled?
3. `_apply_fp8_layerwise_casting()` calls `model.enable_layerwise_casting(storage_dtype=float8_e4m3fn)`
4. Weights are stored in FP8 in VRAM (~50% of fp16), cast layer-by-layer to fp16/bf16 during forward pass
5. Only one layer is in full precision at a time, so overhead is minimal (~1-3% VRAM)

## Not Modified (by design)

These model types cannot use `enable_layerwise_casting()`:

| Model Type | Reason |
|-----------|--------|
| Text Encoders (CLIP, T5, Qwen3) | `transformers` library, not diffusers `ModelMixin` |
| LoRA | State-dict patches, not a standalone model |
| IP-Adapter | Custom `RawModel` wrapper |
| Textual Inversion | Custom wrapper |
| Spandrel (Upscaler) | Not diffusers-based |
| ONNX | ONNX runtime, not torch |
| Flux Checkpoint (custom Flux class) | Not diffusers `ModelMixin` |
| GGUF models | Already quantized |
| BnB models | Already quantized |

## Follow-up Work

- Standalone VAE: Add `fp8_storage` field directly to VAE config classes
- Manual FP8 for custom classes: Implement FP8 storage for Flux Checkpoint (`Flux` class) without `enable_layerwise_casting()`
- Native FP8 compute: For RTX 40xx+ GPUs, use actual FP8 tensor cores via TorchAO
