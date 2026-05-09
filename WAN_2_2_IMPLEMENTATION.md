# Wan 2.2 Image Generation — Implementation Plan

**Branch:** `lstein/feature/wan-image-2-2`
**Status:** Planning
**Owner:** Lincoln Stein

## 0. Naming and Ground Rules

- New base: `BaseModelType.Wan = "wan"` (single base for both A14B and TI2V-5B; variants distinguish them).
- Backend module path: `invokeai/backend/wan/` (mirrors `invokeai/backend/anima/`, `invokeai/backend/flux/`).
- Invocations: prefix `wan_*` (e.g. `wan_model_loader`, `wan_text_encoder`, `wan_denoise`, `wan_lora_loader`, `wan_image_to_latents`, `wan_latents_to_image`, `wan_controlnet`, `wan_ref_image`).
- Submodel layout (per Diffusers `WanPipeline` / `WanImageToVideoPipeline`): `transformer/` (A14B has both `transformer/` and `transformer_2/`), `text_encoder/` (UMT5-XXL), `tokenizer/`, `vae/`, `scheduler/`.
- Diffusers 0.37.0 already in `pyproject.toml` and exposes `WanPipeline`, `WanImageToVideoPipeline`, `WanTransformer3DModel`, `AutoencoderKLWan`. **No diffusers bump required.**

## 1. Model Architecture Reality Check (verified against Diffusers 0.37.0)

These shape and signature facts shape every later design decision:

- `WanTransformer3DModel.__init__` defaults: `patch_size=(1,2,2)`, `text_dim=4096` (UMT5-XXL hidden), `in_channels=16`, `num_layers=40`, `num_attention_heads=40`, `attention_head_dim=128`. So a `text_dim` of 4096 is the strongest UMT5-XXL fingerprint.
- `WanTransformer3DModel.forward(hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_image=None, ...)` — text via `encoder_hidden_states`, optional CLIP image embedding via `encoder_hidden_states_image` (this is the I2V path; we will not feed it for pure T2I but **will** for "reference image at frame 1").
- `WanPipeline.__call__(prompt, ..., num_frames, guidance_scale, guidance_scale_2, ...)` — Diffusers already handles the two-expert swap when `transformer_2` is loaded; `guidance_scale` is for the high-noise expert and `guidance_scale_2` is for the low-noise expert.
- `AutoencoderKLWan.__init__` default: `z_dim=16`, `scale_factor_temporal=4`, `scale_factor_spatial=8`. **Standard Wan VAE used by A14B.**
- TI2V-5B uses a larger Wan2.2-VAE with `z_dim=48`. Latent channels are the strongest discriminator on disk.
- For `num_frames=1`, the temporal patch dimension collapses, but Wan still expects `[B, C, T=1, H, W]` 5D tensors. Latents-to-image will need to squeeze T just like Anima already does.
- A14B carries **two transformers** (high-noise + low-noise expert), shipped as separate `transformer/` and `transformer_2/` subfolders. Each is ~14B params — drives every VRAM and quantization decision.

## 2. Phasing Summary

| Phase | Goal | Independent? |
|---|---|---|
| 0 | Probe + taxonomy + base type | foundational (gate for all others) |
| 1 | Diffusers-format MVP T2I (TI2V-5B first) | depends on 0 |
| 2 | A14B dual-expert loader + denoise hooks + **Low VRAM mode** | depends on 1 |
| 3 | Standalone VAE + UMT5-XXL encoder configs | depends on 0; can run parallel to 1/2 |
| 4 | GGUF transformer (single-file) — both experts | depends on 2, 3 |
| 5 | LoRA (single + dual-expert pairing) | depends on 2 |
| 6 | ControlNet | depends on 2 |
| 7 | Reference image (frame-1 I2V conditioning) | depends on 2 |
| 8 | Inpaint | depends on 2 (uses `RectifiedFlowInpaintExtension`) |
| 9 | Frontend wiring (model picker, params slice, graph builder) | depends on 1 minimum |
| 10 | Starter models, docs | last |

Phases 5–8 can all run in parallel after Phase 2 lands. Phase 4 is the largest single unit of work.

---

## VRAM Targets and the Low VRAM Mode

Dev hardware: 16 GB VRAM card. Most InvokeAI users are at 16 GB or below, so the low-VRAM path is mandatory regardless.

| Config | Active VRAM (transformer only) | Verdict |
|---|---|---|
| TI2V-5B @ bf16 | ~10 GB | Comfortable native fit |
| A14B @ bf16 (one expert resident) | ~28 GB per expert | Won't fit; needs CPU offload |
| A14B @ Q8 GGUF (one expert) | ~14 GB | Tight; possible with offload of encoder/VAE |
| A14B @ Q4_K_M GGUF (one expert) | ~7 GB | Comfortable; realistic 16 GB path |

UMT5-XXL is ~5B params (~10 GB bf16) but only encodes once before denoise — it gets moved off GPU before the transformer runs.

**Low VRAM mode** (revised in Phase 2 implementation): InvokeAI's model cache already exposes partial loading via `InvokeAIAppConfig.enable_partial_loading` (default `True`). When a model exceeds the VRAM budget the cache loads what fits and streams the rest from RAM per forward pass. Combined with `_ExpertSwapper` (which keeps only one expert locked at a time, freeing the other for cache eviction), the A14B-at-bf16-on-16-GB scenario is **already handled by existing infrastructure** — no `low_vram` field on `wan_denoise` is required. Users with less VRAM than the model size get the slow-but-functional path automatically.

---

## Phase 0 — Foundation (taxonomy, base type, FE enum, probe scaffolding)

### Backend changes

- `invokeai/backend/model_manager/taxonomy.py`
  - Add `Wan = "wan"` to `BaseModelType`.
  - Add `class WanVariantType(str, Enum)` with `T2V_A14B = "t2v_a14b"` and `TI2V_5B = "ti2v_5b"`.
  - Add `WanVariantType` to the `AnyVariant` union and to `variant_type_adapter`.
- `invokeai/backend/model_manager/configs/main.py`
  - Add `MainModelDefaultSettings.from_base(BaseModelType.Wan, variant=...)`: A14B → `cls(steps=40, cfg_scale=4.0, width=1024, height=1024)`; TI2V-5B → `cls(steps=30, cfg_scale=5.0, width=1024, height=1024)`. Tune later.
- `invokeai/app/util/step_callback.py`
  - Add `BaseModelType.Wan` branch. 16-channel projection matrix for A14B; TI2V-5B's 48-channel preview is a TODO (non-blocking).
- `invokeai/app/services/shared/sqlite_migrator/migrations/migration_NN.py`
  - New migration to widen any base-model enum constraint (mirror Anima's `migration_26.py`). Bump `model_records_schema_version`.

### Frontend changes

- `invokeai/frontend/web/src/features/nodes/types/common.ts` — add `'wan'` to `zBaseModelType`/`zMainModelBase`; add `zWanVariantType`; include in `zAnyModelVariant`.
- `invokeai/frontend/web/src/features/parameters/types/constants.ts` — add `wan` to `CLIP_SKIP_MAP` with `maxClip=0`.
- `invokeai/frontend/web/src/features/nodes/types/constants.ts` — add `WanMainModelField` colour entry.

### Decisions

- **One base for both, or split?** One base (`Wan`) with two variants. They share text encoder (UMT5-XXL) and pipeline ergonomics. Splitting would double FE selectors for marginal gain.
- **Naming**: prefer `wan` over `wan-image` — Wan 2.3 is coming.

### Test surface

- `tests/backend/model_manager/configs/test_main_config.py` — migration adds new enum, existing rows still validate.

---

## Phase 1 — Diffusers Pipeline MVP (TI2V-5B, T2I single-frame)

Start with TI2V-5B because it's smaller (single transformer ~5B, fits ~16 GB), avoids the dual-expert complication, and validates the encoder/VAE/denoise path before adding the MoE layer.

### Probe / config

- `invokeai/backend/model_manager/configs/main.py`
  - Add `Main_Diffusers_Wan_Config(Diffusers_Config_Base, Main_Config_Base, Config_Base)`:
    - `base: Literal[BaseModelType.Wan]`, `variant: WanVariantType`.
    - `from_model_on_disk` accepts class names `{"WanPipeline", "WanImageToVideoPipeline", "WanTransformer3DModel"}`.
    - Variant detection: load `transformer/config.json`; if `in_channels` indicates 48-ch latents → TI2V-5B; if 16-ch and a sibling `transformer_2/` exists → A14B. Filename heuristic fallback.
    - `has_dual_expert: bool` field set at probe time.
- `invokeai/backend/model_manager/configs/factory.py` — add `Main_Diffusers_Wan_Config` to `AnyModelConfig` union.

### Loader

- `invokeai/backend/model_manager/load/model_loaders/wan.py` (new). Mirror `qwen_image.py`. Initial scope: TI2V-5B only.
  - Transformer: `WanTransformer3DModel.from_pretrained(model_path / "transformer", torch_dtype=bfloat16)`.
  - VAE: `AutoencoderKLWan.from_pretrained(model_path / "vae", torch_dtype=bfloat16)`.
  - Text encoder: standard `T5EncoderModel` / `T5TokenizerFast` from `text_encoder/` and `tokenizer/`. **Verify `model_type` in config.json — if `umt5`, use `UMT5EncoderModel` from transformers.**

### Invocation nodes (TI2V-5B only)

- `wan_model_loader.py` — outputs `transformer: TransformerField`, `vae: VAEField`, `text_encoder: WanTextEncoderField`.
- `model.py` — add `class WanTextEncoderField(BaseModel)` with `tokenizer`, `text_encoder`, `loras`.
- `wan_text_encoder.py` — runs UMT5-XXL, returns `WanConditioningField`. Output `WanConditioning` dataclass: `prompt_embeds: [seq_len, 4096]` + `prompt_attention_mask`. Add `WanConditioningInfo` to `invokeai/backend/stable_diffusion/diffusion/conditioning_data.py`.
- `fields.py` — add `WanConditioningField` and `wan_model` field-description string.
- `wan_image_to_latents.py` — VAE encode, mirroring `qwen_image_image_to_latents.py`. Wan VAE expects 5D `[B,3,1,H,W]`.
- `wan_latents_to_image.py` — VAE decode, squeeze T.
- `wan_denoise.py` — heart of the work for this phase.

### Denoise loop design

**Decision: bypass `WanPipeline.__call__` and drive the loop ourselves (Option A).** Same as every other InvokeAI backend — keeps LoRA / ControlNet / inpaint plumbing consistent.

For Phase 1 (single transformer):
- `invokeai/backend/wan/sampling_utils.py` — `get_noise(...)` returning 5D `[1, z_dim, 1, H/8, W/8]`; a `WanScheduler` (start with `FlowMatchEulerDiscreteScheduler` from Diffusers).
- Pseudocode:
  ```python
  latents = get_noise(...) or noised init
  for t in timesteps:
      noise_pred_cond = transformer(latents, t, prompt_embeds, ...)
      if cfg:
          noise_pred_uncond = transformer(latents, t, neg_embeds, ...)
          noise_pred = noise_pred_uncond + scale * (cond - uncond)
      latents = scheduler.step(noise_pred, t, latents)
      step_callback(...)
  return latents
  ```
- Reuse `RectifiedFlowInpaintExtension` from `invokeai.backend.rectified_flow.rectified_flow_inpaint_extension`.

### Open questions

- Does `WanPipeline` use `FlowMatchEulerDiscreteScheduler`? Confirm against `Wan-AI/Wan2.2-TI2V-5B-Diffusers/scheduler/scheduler_config.json`.
- New `WanT5EncoderConfig` rather than reuse of `T5Encoder_T5Encoder_Config`? **Yes** — UMT5-XXL is not bit-compatible with T5-XXL. See Phase 3.
- Does `WanTransformer3DModel` accept attention mask through `attention_kwargs`?

### Test surface

- `tests/app/invocations/test_wan_text_encoder.py` — output shape sanity.
- `tests/app/invocations/test_wan_denoise.py` (slow, gated by `INVOKEAI_HEAVY_TESTS=1`) — 4-step denoise on TI2V-5B at 256x256, assert non-NaN.
- `tests/backend/model_manager/configs/test_wan_config.py` — variant detection.

### Files touched in Phase 1

- `invokeai/backend/model_manager/taxonomy.py`
- `invokeai/backend/model_manager/configs/main.py`
- `invokeai/backend/model_manager/configs/factory.py`
- `invokeai/backend/model_manager/load/model_loaders/wan.py` (new)
- `invokeai/app/invocations/wan_model_loader.py` (new)
- `invokeai/app/invocations/wan_text_encoder.py` (new)
- `invokeai/app/invocations/wan_denoise.py` (new)
- `invokeai/app/invocations/wan_image_to_latents.py`, `wan_latents_to_image.py` (new)
- `invokeai/backend/wan/__init__.py`, `sampling_utils.py`, `conditioning_data.py` (new)

---

## Phase 2 — Dual-Expert MoE (Wan2.2-T2V-A14B) + Low VRAM Mode

### MoE detail

Wan 2.2 A14B runs two `WanTransformer3DModel` instances. `WanPipeline` swaps based on a noise threshold. `boundary_ratio` (default 0.875) lives on the scheduler config — the high-noise expert handles the first 12.5% of denoising, low-noise handles the rest.

### Loader changes

- Extend `SubModelType` with `Transformer2 = "transformer_2"`. Cleanest path: each expert is its own cacheable entity, `apply_smart_model_patches` LoRAs each independently, matches Diffusers folder layout. Mirror in FE `common.ts`.

### TransformerField split

- New `WanTransformerField`:
  ```python
  class WanTransformerField(BaseModel):
      transformer_high: ModelIdentifierField
      transformer_low: ModelIdentifierField | None
      loras_high: List[LoRAField] = []
      loras_low: List[LoRAField] = []
      boundary_ratio: float = 0.875
  ```
  In `invokeai/app/invocations/model.py`. Single explicit place where MoE-ness is encoded.
- `wan_model_loader.py` populates both. TI2V-5B leaves `transformer_low` as `None`.

### Denoise loop changes

- `wan_denoise.py`:
  ```python
  with ExitStack() as exit_stack:
      _, transformer_high = exit_stack.enter_context(context.models.load(field.transformer_high).model_on_device())
      transformer_low = None
      if field.transformer_low is not None:
          _, transformer_low = exit_stack.enter_context(context.models.load(field.transformer_low).model_on_device())
      apply_loras(transformer_high, field.loras_high)
      if transformer_low: apply_loras(transformer_low, field.loras_low)

      for i, t in enumerate(timesteps):
          model = transformer_high
          if transformer_low is not None and (t / t_max) < boundary_ratio:
              model = transformer_low
          noise_pred = model(...)
          # ...
  ```

### VRAM strategy (default mode)

- Both experts in **system RAM** (~28 GB at bf16; cheap in 2026).
- Only the active expert on **GPU**. Boundary crossing once per denoise → ~2s CPU↔GPU transfer overhead.
- Implementation: re-enter `model_on_device()` for the other expert after boundary crossing.

### Low VRAM mode (new — needed for 16 GB dev card and most users)

- New `low_vram: bool` field on `wan_denoise` (also a global setting).
- Mode A (default): RAM-resident, GPU-juggle on boundary as above.
- Mode B (low VRAM): wrap each transformer with `enable_model_cpu_offload()` semantics — model stays on CPU, individual layers move to GPU on forward call. Slow (~minutes/step at bf16, but seconds/step at Q4 GGUF). Let users render even when full active-expert won't fit.
- Mode B also useful for keeping the text encoder CPU-resident the whole time on tight VRAM.

### Dual-expert LoRA pairing

- Community releases ship paired files: `xxx_high_noise.safetensors` + `xxx_low_noise.safetensors`.
- New `wan_lora_loader` accepts either single LoRA (auto-applied to both, with quality warning) or explicit `lora_high` + `lora_low` pair.
- Probe identifies each as `LoRA_LyCORIS_Wan_Config(base=Wan)` with optional `expert: Literal["high","low"] | None` from filename heuristic (`"high_noise"`/`"low_noise"` substring).

### Files touched in Phase 2

- `invokeai/backend/model_manager/taxonomy.py` (add `Transformer2`)
- `invokeai/backend/model_manager/load/model_loaders/wan.py`
- `invokeai/app/invocations/model.py` (add `WanTransformerField`)
- `invokeai/app/invocations/wan_model_loader.py` (extend for dual)
- `invokeai/app/invocations/wan_denoise.py` (MoE swap + low VRAM mode)
- `invokeai/app/invocations/wan_lora_loader.py`
- `invokeai/frontend/web/src/features/nodes/types/common.ts` (Transformer2)

### Open questions

- `boundary_ratio` units in actual `scheduler_config.json` — timestep fraction vs sigma threshold? Read source of truth from disk.
- Expose `boundary_ratio` as advanced UI control? **Yes** — useful for experimentation, default from config.
- Expose `cfg_scale_low_noise` separately from `cfg_scale`? **Yes** as advanced override; default both to same value.

### Test surface

- Mock dual-expert load + boundary crossing: ensure correct expert called at each step. Fake transformer that records calls.
- Low VRAM mode smoke test against TI2V-5B (single-expert), confirm output matches default mode.

---

## Phase 3 — Standalone VAE + UMT5-XXL Encoder Configs

Makes the GGUF flow possible by allowing users to install only encoder + VAE + quantized transformer.

### VAE configs

- `invokeai/backend/model_manager/configs/vae.py`
  - `_is_wan_vae(state_dict)`: 5D conv weights and `decoder.conv_in.weight.shape[1] in {16, 48}`.
  - `VAE_Checkpoint_Wan_Config(Checkpoint_Config_Base, Config_Base)` with `base=Wan`, `latent_channels: Literal[16, 48]`. Detect via `state_dict["decoder.conv_in.weight"].shape[1]`.
  - Update `_validate_looks_like_vae` to exclude Wan VAEs (mirror Qwen Image / FLUX.2 exclusion at lines 113-118).
  - `VAE_Diffusers_Wan_Config` for diffusers-format Wan VAE (`AutoencoderKLWan`).
- `factory.py` — add both new VAE configs to `AnyModelConfig`.

### VAE loader

- `wan.py` — register `(base=Wan, type=VAE, format=Checkpoint)` and `format=Diffusers`.
- For checkpoint: build `AutoencoderKLWan(z_dim=...)` based on detected latent channels, then `model.load_state_dict(sd, assign=True)`. **VAE in fp16 is broken — use bf16** (FluxVAELoader pattern).

### UMT5-XXL encoder

- `invokeai/backend/model_manager/configs/wan_t5_encoder.py` (new) — `WanT5Encoder_Diffusers_Config` and `WanT5Encoder_Checkpoint_Config`.
- New config class **rather than reuse** of `T5Encoder_T5Encoder_Config`:
  - UMT5-XXL has `model_type: "umt5"` in transformers.
  - Different vocabulary — InvokeAI shouldn't let users wire a FLUX T5 into the Wan slot.
- New `ModelType.WanT5Encoder = "wan_t5_encoder"` and `ModelFormat.WanT5Encoder = "wan_t5_encoder"`. Add to taxonomy + FE enum.

### Standalone-encoder loader

- New class in `wan.py`: `(base=Any, type=WanT5Encoder, format=...)`. Loads `UMT5EncoderModel` for TextEncoder, `T5TokenizerFast` for Tokenizer. Mirror `T5EncoderLoader` in `flux.py:426-505`.

### Component-source loader pattern

- `wan_model_loader.py` follows `qwen_image_model_loader.py` pattern: optional standalone `vae_model` and `wan_t5_encoder_model` inputs override main model's submodels. Required when main model is single-file GGUF.

### Files touched in Phase 3

- `invokeai/backend/model_manager/configs/vae.py`
- `invokeai/backend/model_manager/configs/wan_t5_encoder.py` (new)
- `invokeai/backend/model_manager/configs/factory.py`
- `invokeai/backend/model_manager/load/model_loaders/wan.py`
- `invokeai/backend/model_manager/taxonomy.py`
- FE: `isWanVAEModelConfig`, `isWanT5EncoderModelConfig` type guards in `services/api/types.ts`; `useWanVAEModels`, `useWanT5EncoderModels` hooks in `services/api/hooks/modelsByType.ts`.

### Open questions

- A14B and TI2V-5B ship the same UMT5-XXL `text_encoder/`? Verify; if yes, one encoder config covers both.

---

## Phase 4 — GGUF Quantization for Both Experts

Highest user impact: brings Wan 2.2 A14B onto consumer hardware.

### Probe / config

- `invokeai/backend/model_manager/configs/main.py`
  - `Main_GGUF_Wan_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base)` with `base=Wan`, `format=GGUFQuantized`, `variant: WanVariantType`, `expert: Literal["high","low","none"] = "none"`.
  - Detection: GGML tensors + Wan-specific keys (`blocks.0.attn1.to_q.weight`, `attn2.to_k.weight` shape `[head_dim*heads, 4096]` for UMT5 cross-attn).
  - Expert from filename: `"high_noise"` / `"low_noise"` substring; fall back to `"none"`. **User must confirm** when ambiguous.

### Loader

- `wan.py` — `(base=Wan, type=Main, format=GGUFQuantized)`. Mirror `QwenImageGGUFCheckpointModel`:
  1. `gguf_sd_loader(model_path, compute_dtype=bfloat16)`
  2. Strip ComfyUI prefixes (`model.diffusion_model.`, `diffusion_model.`).
  3. Auto-detect arch (count `blocks.X.` keys → `num_layers`; `attn1.to_q.weight.shape[0]` → hidden dim).
  4. `with accelerate.init_empty_weights(): model = WanTransformer3DModel(**inferred_config)`
  5. `model.load_state_dict(sd, strict=False, assign=True)`.
- A14B's two GGUFs: same registration handles both — file alone is the unit, called twice by `wan_model_loader` invocation.

### Pairing in the model loader invocation

- UI sketch:
  ```
  Transformer (High Noise)  [GGUF or Diffusers]
  Transformer (Low Noise)   [GGUF or Diffusers, optional — empty for TI2V-5B]
  Component Source          [Diffusers, optional — for VAE/encoder]
  Standalone VAE            [optional]
  Standalone Wan T5 Encoder [optional]
  Low VRAM mode             [bool]
  ```
- Low Noise field hidden on FE when High Noise variant is TI2V-5B.

### Files touched in Phase 4

- `invokeai/backend/model_manager/configs/main.py`
- `invokeai/backend/model_manager/configs/factory.py`
- `invokeai/backend/model_manager/load/model_loaders/wan.py`
- `invokeai/app/invocations/wan_model_loader.py` (extend pickers)

### Open questions

- Reference GGUFs: `city96/Wan2.2-T2V-A14B-gguf`, `QuantStack/Wan2.2-TI2V-5B-GGUF`. Verify key naming matches Diffusers' `WanTransformer3DModel` exactly.
- If only one of the two A14B experts is GGUF'd, fall back to bf16 for the other (mixed quant within one denoise loop). Loader supports this — each transformer slot has independent format.

---

## Phase 5 — LoRA

### Probe / config

- `invokeai/backend/model_manager/configs/lora.py`
  - `_is_wan_lora(state_dict)`: keys like `blocks.0.attn1.to_q.lora_A.weight` / `lora_unet_blocks_0_attn1_to_q.lora_down.weight` / `transformer.blocks.0.attn1.to_q.lora_A.weight`. Exclude clashes with Anima (`cross_attn`/`self_attn`) and FLUX (`double_blocks`, `single_blocks`).
  - `LoRA_LyCORIS_Wan_Config(LoRA_LyCORIS_Config_Base, Config_Base)` with `base=Wan`, optional `expert: Literal["high","low"] | None`.
  - Register in `factory.py`.

### LoRA conversion

- `invokeai/backend/patches/lora_conversions/wan_lora_constants.py` (new) — `WAN_LORA_TRANSFORMER_PREFIX = "lora_transformer-"`.
- `invokeai/backend/patches/lora_conversions/wan_lora_conversion_utils.py` (new) — handle three formats:
  - **Kohya**: `lora_unet_blocks_X_...` → diffusers `blocks.X....`
  - **Diffusers PEFT**: `transformer.blocks.X.attn1.to_q.lora_A.weight` → strip `transformer.` prefix.
  - **Native diffusion_model**: `diffusion_model.blocks.X....` → strip prefix.
- Start from `qwen_image_lora_conversion_utils.py` and adjust prefixes/key-renaming.

### Loader integration

- `invokeai/backend/model_manager/load/model_loaders/lora.py` — add `BaseModelType.Wan` branch calling `lora_model_from_wan_state_dict(state_dict, alpha=None)`.

### Invocation node

- `invokeai/app/invocations/wan_lora_loader.py`:
  - Single LoRA mode (default): one picker, auto-applied to both experts.
  - Dual LoRA mode: two pickers (high / low). Validates bases are both Wan and at most one of each `expert`.
  - Mirrors `AnimaLoRALoaderInvocation` + `AnimaLoRACollectionLoader`.
- Output: `WanLoRALoaderOutput` containing the `WanTransformerField` with updated `loras_high` / `loras_low`.

### Denoise integration

- `wan_denoise.py` — when entering each transformer's `model_on_device()` context, apply `LayerPatcher.apply_smart_model_patches(model=transformer_high, patches=loras_high_iter, prefix=WAN_LORA_TRANSFORMER_PREFIX, ...)`. Pattern from `flux_denoise.py:434-443`.

### Files touched in Phase 5

- `invokeai/backend/model_manager/configs/lora.py`
- `invokeai/backend/model_manager/configs/factory.py`
- `invokeai/backend/model_manager/load/model_loaders/lora.py`
- `invokeai/backend/patches/lora_conversions/wan_lora_constants.py` (new)
- `invokeai/backend/patches/lora_conversions/wan_lora_conversion_utils.py` (new)
- `invokeai/app/invocations/wan_lora_loader.py` (new)
- `invokeai/app/invocations/wan_denoise.py`

---

## Phase 6 — ControlNet

Wan ControlNet ecosystem **less mature** than FLUX. Common community models target Wan2.1, with Wan2.2 ports trickling out. Treat with thrash risk.

### Approach

- `invokeai/backend/wan/controlnet/` mirroring `invokeai/backend/flux/controlnet/`. Two state-dict identifiers initially:
  - **InstantX-style**: `controlnet_x_embedder.` / `controlnet_blocks.` + `blocks.X.attn1.*` transformer keys.
  - **Diffusers Wan ControlNet** (if/when one exists): `WanControlNetModel`-style.
- Configs: `ControlNet_Checkpoint_Wan_Config`, `ControlNet_Diffusers_Wan_Config` in `invokeai/backend/model_manager/configs/controlnet.py`.
- Loader: extend `wan.py`.
- Extension: `invokeai/backend/wan/extensions/wan_controlnet_extension.py` — callable taking control-image, returning per-block residuals. Pattern from `flux/extensions/instantx_controlnet_extension.py`.
- Invocation: `invokeai/app/invocations/wan_controlnet.py` — defines `WanControlNetField` and picker node.
- Denoise: `wan_denoise.py` accepts `control: WanControlNetField | list[WanControlNetField] | None`.

### Risks

- If community ControlNet weights only target one expert, need conditional injection. Defer until reference model in hand.
- ControlNet may want a separate VAE-encoded conditioning image (FLUX denoise pattern).
- **Gate on ecosystem maturity**: ship v1 without ControlNet if Wan2.2-native models aren't ready; add as v2.

---

## Phase 7 — Reference Image (Frame-1 I2V Conditioning)

Wan 2.2's I2V variant takes an image and produces a video starting from it. At `num_frames=1`, becomes a reference image — analogous to FLUX Kontext.

### Decision: Path B — CLIP-vision conditioning via `encoder_hidden_states_image`

`WanTransformer3DModel.forward` accepts `encoder_hidden_states_image: Optional[Tensor]`. I2V pipeline preprocesses the ref image through CLIP-vision and feeds those features. We do the same with stock `CLIPVisionModelWithProjection` (already in `invokeai/backend/model_manager/load/model_loaders/clip_vision.py`).

Treats ref-image as conditioning rather than a different model. Simpler UI, no extra 30 GB checkpoint to install. Same approach as FLUX Kontext (`invokeai/backend/flux/extensions/kontext_extension.py`).

### Implementation

- `invokeai/backend/wan/extensions/wan_ref_image_extension.py` — encodes via CLIP vision, produces `image_embeds` for `encoder_hidden_states_image`.
- `wan_denoise.py` accepts `ref_image: WanRefImageConditioningField | None`.

### Open questions

- Wan2.2-T2V-A14B's `transformer/config.json` likely has `image_dim=None` (text-only); I2V variant has `image_dim != None`. **Ref-image path only works on I2V variants.** Either ship I2V as separate variant or detect and reject gracefully. Add `WanVariantType.I2V_A14B = "i2v_a14b"` if shipping. Probe via `transformer/config.json::image_dim`.

---

## Phase 8 — Inpaint

Inpaint = image-to-image with denoise mask. `RectifiedFlowInpaintExtension` already handles this for Anima and FLUX. Wan's flow-matching scheduler is mathematically identical; reuse should be straightforward.

### Implementation

- `wan_denoise.py` accepts `denoise_mask: DenoiseMaskField | None`.
- Reuse `RectifiedFlowInpaintExtension` from `invokeai.backend.rectified_flow.rectified_flow_inpaint_extension`. Anima needed `AnimaInpaintExtension` for shifted timesteps; for Wan, check if the scheduler shift introduces the same issue. If yes, subclass.

### Files touched in Phase 8

- `invokeai/app/invocations/wan_denoise.py` (mask branch)
- Possibly `invokeai/backend/wan/wan_inpaint_extension.py`

---

## Phase 9 — Frontend Wiring

### Type definitions

- `invokeai/frontend/web/src/services/api/types.ts` — `isWanMainModelConfig`, `isWanLoRAModelConfig`, `isWanVAEModelConfig`, `isWanT5EncoderModelConfig`, `isWanControlNetModelConfig`. Mirror Anima/Qwen Image at lines 286-322.
- `invokeai/frontend/web/src/services/api/hooks/modelsByType.ts` — `useWanMainModels`, `useWanVAEModels`, `useWanT5EncoderModels`, `useWanLoRAModels`, `useWanControlNetModels`. Mirror lines 105-113.

### Params slice

- `invokeai/frontend/web/src/features/controlLayers/store/paramsSlice.ts`
  - Selectors: `selectWanVaeModel`, `selectWanT5EncoderModel`, `selectWanScheduler`, `selectWanBoundaryRatio`, `selectWanLowVramMode`. Anima sets the precedent.
  - State: `wanVaeModel`, `wanT5EncoderModel`, etc.

### Graph builder

- `invokeai/frontend/web/src/features/nodes/util/graph/generation/buildWanGraph.ts` (new). Mirror `buildAnimaGraph.ts`. Differences:
  - Two transformer pickers when variant is A14B.
  - Dual-expert LoRA collection node.
  - Separate VAE / WanT5Encoder pickers (GGUF requires them).
  - Low VRAM toggle.
- `invokeai/frontend/web/src/features/nodes/util/graph/generation/addWanLoRAs.ts` (new).
- `invokeai/frontend/web/src/features/nodes/util/graph/types.ts` — register Wan in `GraphBuilderArg`.
- Graph dispatcher (`buildGenerationTabGraph.ts`) — add `case 'wan'`.

### UI

- ModelPicker, ControlLayer toolbox iterate over `BaseModelType` so adding `'wan'` should propagate. Audit `ModelPicker.tsx` for hardcoded base lists.

---

## Phase 10 — Starter Models, Migration, Docs

### Starter models

- `invokeai/backend/model_manager/starter_models.py` — append `# region Wan` block:
  ```python
  wan_t5_encoder = StarterModel(name="Wan T5 Encoder (UMT5-XXL)",
      base=BaseModelType.Any, source="Wan-AI/Wan2.2-T2V-A14B-Diffusers::text_encoder+tokenizer",
      type=ModelType.WanT5Encoder, format=ModelFormat.WanT5Encoder, ...)
  wan_vae = StarterModel(name="Wan VAE",
      base=BaseModelType.Wan, source="Wan-AI/Wan2.2-T2V-A14B-Diffusers::vae/diffusion_pytorch_model.safetensors",
      type=ModelType.VAE, format=ModelFormat.Checkpoint, ...)
  wan_vae_2_2 = StarterModel(name="Wan2.2 VAE",
      base=BaseModelType.Wan, source="Wan-AI/Wan2.2-TI2V-5B-Diffusers::vae/...",
      type=ModelType.VAE, ...)
  wan_t2v_a14b = StarterModel(name="Wan 2.2 T2V A14B",
      base=BaseModelType.Wan, source="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
      type=ModelType.Main, variant=WanVariantType.T2V_A14B, ...)
  wan_t2v_a14b_high_q4 = StarterModel(name="Wan 2.2 T2V A14B High Noise (Q4_K_M)",
      base=BaseModelType.Wan,
      source="https://huggingface.co/city96/Wan2.2-T2V-A14B-gguf/resolve/main/wan2.2-t2v-a14b-high-noise-Q4_K_M.gguf",
      ..., dependencies=[wan_t5_encoder, wan_vae])
  wan_t2v_a14b_low_q4 = ...
  wan_ti2v_5b = StarterModel(name="Wan 2.2 TI2V 5B",
      base=BaseModelType.Wan, source="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
      variant=WanVariantType.TI2V_5B, ...)
  ```
- Verify each `source` URL exists before merge.

### DB migration

- New `migration_NN.py` only if `BaseModelType` Enum constraint rejects unknown values. Inspect `migration_26.py` (Anima) for pattern.

### Docs

- Update `docs/` (model support tables, getting-started for Wan).

---

## Risk Register

| # | Risk / Unknown | Mitigation |
|---|---|---|
| 1 | `WanPipeline` Option A bypass — enough hooks? | Source confirms `WanTransformer3DModel.forward` is callable directly. Low risk. |
| 2 | Dual-expert VRAM blowup | Default RAM-resident + GPU-juggle on boundary. Low VRAM mode covers tighter cases. GGUF Q4 → ~7 GB/expert. Document expectations. |
| 3 | GGUF availability for both A14B experts | `city96/Wan2.2-T2V-A14B-gguf` advertises both. Verify before Phase 4. Mixed-quant denoise as fallback. |
| 4 | UMT5-XXL vs T5-XXL distinction | Strict probe via `model_type`. Separate `WanT5Encoder` type prevents cross-wiring. |
| 5 | Wan ControlNet ecosystem maturity | Phase 6 may slip — ship v1 without if Wan2.2-native models not ready, ControlNet as v2. |
| 6 | Single-frame inference is OOD | Empirically fine. Document as known property. |
| 7 | Boundary ratio variability | Read from `scheduler/scheduler_config.json::boundary_ratio` per-model. Default 0.875. |
| 8 | TI2V-5B's 48-channel VAE | Probe both 16/48 in `_is_wan_vae`. Denoise loop reads `z_dim` from VAE config, doesn't hardcode. |
| 9 | DB enum widening | Standard migration template (Anima's `migration_26.py`). Low risk. |
| 10 | Diffusers' modular `Wan22Blocks`/`WanModularPipeline` — use it? | No. Modular = extra moving part. Stick to `WanPipeline`/`WanTransformer3DModel`. |
| 11 | FE vitest tests for new base type | Mostly automatic via zod enum; audit `*.test.ts` mentioning `'anima'`. |
| 12 | Step preview latents for Wan | Reuse FLUX 16-channel matrix for A14B. TI2V-5B's 48-channel: degraded preview (slice 16) until proper RGB factors generated via `scripts/generate_vae_linear_approximation.py`. |

---

## Recommended Working Cadence

1. Phases 0 + 1 (TI2V-5B Diffusers MVP) — one PR, foundational, no user-visible features but unblocks everything.
2. Phase 2 (A14B dual-expert + Low VRAM mode) — second PR, first user-visible feature.
3. Phase 3 (standalone components) — third PR, parallelizable with Phase 2.
4. Phase 4 (GGUF) — fourth PR, the big VRAM win.
5. Phase 5 (LoRA) — fifth PR.
6. Phases 6, 7, 8 in parallel — small targeted PRs.
7. Phase 9 (FE) tracks each backend phase.
8. Phase 10 (starters) gates final release.

Total: ~4–6 weeks focused work. Schedule risk concentrated on Phase 6 (ControlNet) and Phase 4 (GGUF arch verification).
