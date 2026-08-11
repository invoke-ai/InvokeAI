# Edit Mode Feature Plan (External + Local Models)

## Background

Several models — both external API providers and local model architectures — support
**instruction-based image editing**: an input image plus a prompt that describes the
desired modification, with no denoise-strength slider. This is fundamentally different
from classical `img2img`, which scales noise against an init latent.

**External providers** with this capability: Gemini, OpenAI GPT Image, Seedream,
Qwen Image Edit Max. Currently partially exposed as `img2img` (OpenAI GPT Image,
Seedream) and partially hidden (Gemini — provider code accepts `init_image` but UI
is gated off).

**Local model architectures** with this capability:
- **FLUX.2 Klein** (all variants: Klein 4B, Klein 4B Base, Klein 9B, Klein 9B Base) —
  built-in editing via `kontext_conditioning` reference images, wired in
  [`flux2_denoise.py:134-139`](invokeai/app/invocations/flux2_denoise.py#L134-L139)
  and [`flux2_denoise.py:413-426`](invokeai/app/invocations/flux2_denoise.py#L413-L426).
- **Qwen Image Edit 2511** (diffusers + all GGUF variants Q2_K / Q4_K_M / Q6_K /
  Q8_0) — built-in editing via `init_latents` path in
  [`qwen_image_denoise.py:351-440`](invokeai/app/invocations/qwen_image_denoise.py#L351-L440).

This plan introduces a dedicated `edit` mode and a per-model capability flag so the
UI can offer an explicit toggle on every model that supports it, regardless of
whether the model runs locally or via an external API.

## Goals

- Semantically honest mode names — separate denoise-based `img2img` from
  instruction-based `edit`.
- Per-model opt-in toggle in the parameters panel; default behavior stays `txt2img`.
- Backward compatibility for stored metadata and saved workflows.

## Non-Goals

- No new canvas tab. Edit happens inside the existing txt2img canvas flow, toggled
  on via the params panel.
- No automatic migration of saved Workflows — only Recall metadata mapping.
- No changes to the denoise nodes themselves — the edit pathways already exist for
  FLUX.2 Klein (`kontext_conditioning`) and Qwen Image Edit (`init_latents`). Only
  the capability surface and graph builders change.

## Schema Changes

### External API schema — `invokeai/backend/model_manager/configs/external_api.py`

- Extend `ExternalGenerationMode`:
  ```python
  ExternalGenerationMode = Literal["txt2img", "img2img", "inpaint", "edit"]
  ```
- Add capability field on `ExternalModelCapabilities`:
  ```python
  supports_edit_mode: bool = Field(default=False)
  ```
- Extend `ExternalPanelControlName` union:
  ```python
  ExternalPanelControlName = Literal["reference_images", "dimensions", "seed", "edit_mode"]
  ```

### Local model capability surface

Local model configs in `invokeai/backend/model_manager/configs/main.py` need a way
to declare edit support. Two viable approaches:

**Option A — Derived (preferred):** Compute `supports_edit_mode` in the frontend
from existing config fields. No backend schema change.
- FLUX.2 Klein: any `MainConfig` with `base == BaseModelType.Flux2` and a
  Klein-family variant.
- Qwen Image Edit: any `MainConfig` with `base == BaseModelType.QwenImage` whose
  `name` or `source` identifies it as an Edit variant (existing pattern in
  [`buildQwenImageGraph.ts`](invokeai/frontend/web/src/features/nodes/util/graph/generation/buildQwenImageGraph.ts)).

**Option B — Explicit:** Add `supports_edit_mode: bool = Field(default=False)`
to the relevant main-model configs and set it in starter models.

Recommendation: start with Option A (zero schema risk, no migration). Move to
Option B only if user-installed (non-starter) edit-capable models need to declare
the flag manually.

## Model Capability Mapping

### External — `invokeai/backend/model_manager/starter_models.py`

| Model | Edit Mode | Notes |
| --- | --- | --- |
| Gemini 2.5 Flash Image | yes | provider already handles `init_image` |
| Gemini 3 Pro Image Preview | yes | |
| Gemini 3.1 Flash Image Preview | yes | |
| GPT Image 1 | yes | replaces current `img2img` mode |
| GPT Image 1.5 | yes | replaces current `img2img` mode |
| GPT Image 1 Mini | yes | replaces current `img2img` mode |
| DALL-E 3 | no | no edits endpoint |
| Seedream 4.0 / 4.5 / 5.0 / 5.0 Lite | yes | replaces current `img2img` mode |
| Qwen Image Edit Max | yes | reference-image driven edit |
| Qwen Image 2.0 / 2.0 Pro / Max | no | txt2img only |
| Wan 2.6 T2I | no | txt2img only |

For every external model with `supports_edit_mode=True`:
- Add `"edit"` to `capabilities.modes`.
- Remove `"img2img"` from `capabilities.modes` (was always semantically misleading
  for external providers).
- Add `{"name": "edit_mode"}` to `panel_schema.generation`.

### Local — derived from `MainConfig` fields

| Model family | Edit Mode | Detection rule (frontend) |
| --- | --- | --- |
| FLUX.2 Klein 4B / 4B Base / 9B / 9B Base | yes | `base === 'flux2' && variant.startsWith('klein')` |
| Qwen Image Edit 2511 (diffusers, GGUF Q2_K/Q4_K_M/Q6_K/Q8_0) | yes | `base === 'qwen_image' && /edit/i.test(name)` (existing helper) |
| FLUX.1 (Dev/Schnell), FLUX Kontext | no | (Kontext is its own ref-image flow already; not re-modeled) |
| SD1 / SDXL / SD3 / Z-Image / etc. | no | classical img2img only |

Note: FLUX.1 Kontext already has a dedicated reference-image UI path; folding it
into the unified Edit Mode toggle is left out of scope here to avoid touching its
established workflow. Revisit in a follow-up.

## Provider Invocations

`invokeai/app/invocations/external_image_generation.py`

- Remove `ui_hidden=True` from `init_image` on `GeminiImageGenerationInvocation`.
- Adjust mode defaults; expose `mode` field gated by capability rather than always
  hidden.
- Backward-compat alias: at request-build time, map incoming `mode="img2img"` to
  `mode="edit"` for providers where `supports_edit_mode=True` and `img2img` is no
  longer in `modes`. Log a deprecation notice once.

## Provider Implementations

- `providers/gemini.py` — already wired for `init_image`; just unblock UI.
- `providers/openai.py` — replace the current `mode != "txt2img"` heuristic with
  explicit `mode == "edit"`. `/v1/images/edits` routing stays the same.
- `providers/seedream.py` — already accepts init image in img2img path; rename
  internal handling to `edit`.
- `providers/alibabacloud.py` — Qwen Image Edit Max already reference-image-driven;
  no payload change, just mode-name update.

Each provider should append an edit-specific system instruction (Gemini already has
one) so the model treats the prompt as an edit directive.

## Frontend

### State & Schema

- Regenerate `services/api/schema.ts` after backend schema changes.
- Add new panel control type in
  `features/parameters/util/externalPanelSchema.ts` for `edit_mode` (external only;
  local models surface the toggle via the unified capability helper).
- Add `editMode: boolean` to `paramsSlice` (global; reset on model change). Single
  field shared across external + local — keeps recall and UX consistent.

### Capability Helper

New helper `features/parameters/util/supportsEditMode.ts`:
```ts
export const supportsEditMode = (config: AnyModelConfig | null): boolean => {
  if (!config) return false;
  if (isExternalApiModelConfig(config)) return config.capabilities.supports_edit_mode === true;
  if (isFlux2KleinModelConfig(config)) return true;     // all Klein variants
  if (isQwenImageEditModelConfig(config)) return true;  // Qwen Image Edit 2511 family
  return false;
};
```
Centralizing the rule lets the toggle, graph builders, recall, and tooltips share
one source of truth.

### UI Component

- New `EditModeToggle.tsx` in `features/parameters/components/Advanced/` (note:
  renamed from `ExternalEditModeToggle` — now generic).
- Render only when `supportsEditMode(selectedModelConfig) === true`.
- Toggle label: "Edit Mode" with tooltip: "Use the canvas image as an edit
  reference. The prompt is treated as an edit instruction. No denoise strength."

### Graph Builders

Each graph builder reads `params.editMode` and wires the architecture-specific
edit pathway.

**External — `buildExternalGraph.ts`:**
- Mode resolution:
  ```ts
  const requestedMode =
    generationMode === 'txt2img' && params.editMode &&
    model.capabilities.supports_edit_mode
      ? 'edit'
      : generationMode;
  ```
- Capability gate (current line 44) must allow `"edit"`.
- When mode is `"edit"`, attach `init_image` from canvas (same flow as current
  img2img branch at lines 126-152).

**FLUX.2 — `buildFlux2Graph.ts`:**
- When `params.editMode === true` AND model is Klein-family AND canvas has a raster
  layer: composite the canvas, build a `flux_kontext_conditioning` node from it,
  connect to the `kontext_conditioning` input on the Flux2 denoise node
  ([line 134](invokeai/app/invocations/flux2_denoise.py#L134)).
- Treat this as `txt2img` for canvas-output purposes (no init_latents); only the
  reference-image extension is added.

**Qwen Image — `buildQwenImageGraph.ts`:**
- When `params.editMode === true` AND model is the Qwen Image Edit family: wire the
  composited canvas through the existing Qwen Image Edit init-latents path
  ([qwen_image_denoise.py:351](invokeai/app/invocations/qwen_image_denoise.py#L351)).
  This reuses the same VAE-encode + init-latents nodes the current Qwen Edit
  workflow uses.
- Disable / hide the denoise-strength slider in the UI when Edit Mode is on — Qwen
  Edit uses a fixed s_0 schedule, not a denoise scalar.

### Tests

- `buildExternalGraph.test.ts`:
  - Edit toggle off → mode `txt2img`, no init_image.
  - Edit toggle on + supported model → mode `edit`, init_image attached.
  - Edit toggle on + unsupported model → throws `UnsupportedGenerationModeError`.
  - Legacy metadata with `mode: "img2img"` on GPT Image → recalls as `edit`.
- `buildFlux2Graph.test.ts`:
  - Edit toggle on + Klein 4B + canvas raster → graph contains `flux_kontext_conditioning` wired into denoise.
  - Edit toggle on + non-Klein FLUX → toggle has no effect (treated as plain txt2img).
- `buildQwenImageGraph.test.ts`:
  - Edit toggle on + Qwen Image Edit 2511 + canvas raster → graph wires init_latents.
  - Edit toggle on + Qwen Image (non-Edit) → toggle has no effect.

### Locales

`invokeai/frontend/web/public/locales/en.json` — add:
- `parameters.editMode.label`
- `parameters.editMode.tooltip`
- `parameters.editMode.unsupported`

## Backward Compatibility

### Metadata Recall

`features/nodes/util/parseMetadata.ts` (and equivalent backend logic):
- When recalling metadata with `mode: "img2img"` for an external model whose
  current capabilities list `edit` but not `img2img`:
  - Set `editMode = true`.
  - Set mode to `edit`.
  - Restore `init_image` reference as before.
- For local models: legacy metadata for FLUX.2 Klein with a `kontext_conditioning`
  field set, or Qwen Image Edit with init_latents, should set `editMode = true`
  on recall so the UI accurately reflects the original state.

### Workflow Definitions

- Backend invocation should still accept `mode="img2img"` in the request schema
  (don't tighten the Literal to remove it). Internally map to `edit` when the
  provider only supports `edit`.
- Emit a one-time logger warning per workflow with the legacy mode value.

### API Schema Versioning

No bump required — the `mode` field is a string literal union; clients that send
`img2img` continue to work due to the compat alias.

## Documentation

External — `docs/src/content/docs/features/External Models/`:
- `index.mdx` — add a section "Edit Mode" explaining the toggle, when it appears,
  and how it differs from local img2img.
- Per-provider pages — replace `img2img` in the modes column with `edit`, add a
  note that Edit Mode must be toggled on in the params panel.
- Clarify in `gemini.mdx` that the long-standing "reference images only" limitation
  is lifted — canvas images now work via Edit Mode.

Local — add or update model-architecture docs:
- FLUX.2 Klein: document Edit Mode toggle, that it uses the built-in
  `kontext_conditioning` reference-image path, and that denoise strength has no
  effect when Edit Mode is on.
- Qwen Image Edit 2511: document Edit Mode toggle, that it uses the existing init
  latents path, and works with all GGUF quantization variants.

## Rollout Phases

### Phase 1 — External backend (safe to ship alone)

- External schema extension.
- Starter-model capability flags.
- Provider routing for `mode="edit"`.
- Backward-compat alias for `img2img` → `edit`.
- Backend tests.

Old frontend keeps working — it sends `img2img`, backend maps it.

### Phase 2 — Unified frontend toggle

- Schema regen.
- `supportsEditMode` capability helper covering external + Flux2 Klein + Qwen
  Image Edit.
- `editMode` state in `paramsSlice`.
- `EditModeToggle` component.
- All three graph builders wired (`buildExternalGraph`, `buildFlux2Graph`,
  `buildQwenImageGraph`).
- Metadata-recall mapping for all three.
- Frontend tests.

### Phase 3 — Docs

- Update External Models docs and modes tables.
- Add Edit Mode section to the External index page.
- Add Edit Mode notes to FLUX.2 Klein and Qwen Image Edit model docs.

## Risks & Open Questions

1. **Edit Mode + Reference Images together** — both attach images to the same
   provider request (Gemini sends both as `inlineData` parts; FLUX.2 Klein can take
   multiple kontext refs). Decide whether the UI should disable one when the other
   is active, or allow combined use.
2. **Canvas state on Edit Mode toggle** — toggling on with an empty canvas should
   either fall back to txt2img silently or surface a warning.
3. **Per-model vs. global toggle state** — current proposal is global. If users
   want per-model persistence (e.g., always-edit for Gemini, never for Seedream),
   `paramsSlice` would need a `Record<modelKey, boolean>` instead. Probably premature.
4. **OpenAI `img2img` consumers** — any external integrations or saved presets that
   set `mode="img2img"` explicitly need the compat alias to stay in place
   indefinitely, or a deprecation window must be communicated.
5. **Qwen Image Edit denoise-strength UI** — the existing slider must hide when
   Edit Mode is on; verify no other code path depends on it being present.
6. **FLUX.1 Kontext relation** — Kontext already has its own reference-image UI;
   determine whether to fold it into the unified Edit Mode toggle in a follow-up
   or keep separate to avoid breaking existing user workflows.
7. **User-installed (non-starter) edit-capable models** — derived detection works
   for known starter models but a third-party Qwen-Edit fine-tune may not match
   the name heuristic. If this becomes an issue, switch to Option B (explicit
   `supports_edit_mode` field on main config) in a later phase.

## Files Touched (Estimate)

Backend:
- `invokeai/backend/model_manager/configs/external_api.py`
- `invokeai/backend/model_manager/starter_models.py`
- `invokeai/app/invocations/external_image_generation.py`
- `invokeai/app/services/external_generation/providers/*.py`

Frontend — shared:
- `invokeai/frontend/web/src/services/api/schema.ts` (generated)
- `invokeai/frontend/web/src/features/parameters/util/externalPanelSchema.ts`
- `invokeai/frontend/web/src/features/parameters/util/supportsEditMode.ts` (new)
- `invokeai/frontend/web/src/features/parameters/components/Advanced/EditModeToggle.tsx` (new)
- `invokeai/frontend/web/src/features/controlLayers/store/paramsSlice.ts`
- `invokeai/frontend/web/src/features/nodes/util/parseMetadata.ts` (recall mapping)
- `invokeai/frontend/web/public/locales/en.json`

Frontend — graph builders:
- `invokeai/frontend/web/src/features/nodes/util/graph/generation/buildExternalGraph.ts` + `.test.ts`
- `invokeai/frontend/web/src/features/nodes/util/graph/generation/buildFlux2Graph.ts` + `.test.ts`
- `invokeai/frontend/web/src/features/nodes/util/graph/generation/buildQwenImageGraph.ts` + `.test.ts`

Docs:
- `docs/src/content/docs/features/External Models/index.mdx`
- `docs/src/content/docs/features/External Models/gemini.mdx`
- `docs/src/content/docs/features/External Models/openai.mdx`
- `docs/src/content/docs/features/External Models/seedream.mdx`
- `docs/src/content/docs/features/External Models/alibabacloud.mdx`
- FLUX.2 Klein and Qwen Image Edit model documentation (location TBD — check
  whether they have dedicated pages or need new ones)
