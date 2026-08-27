# fal.ai External Provider Implementation Plan

**Goal:** Add fal.ai as a native InvokeAI external image provider usable from the Canvas Image Editor.

**Architecture:** Reuse InvokeAI's existing `ExternalProvider`, external model records, starter model synchronization, and Canvas external graph. Add one REST queue adapter using fal.ai's upload and queue APIs. Ship curated image models with accurate capabilities: Flux Schnell/Dev for txt2img, Flux Kontext Pro for img2img, and Flux Fill for inpaint. Keep generic/video support outside this PR.

**Tech Stack:** Python 3.11+, requests, Pydantic settings, InvokeAI external generation service, React/TypeScript frontend, Vitest/Pytest.

## Global Constraints

- Do not submit billable fal.ai inference jobs in automated tests.
- Store provider credentials through InvokeAI's external provider secret handling; never log or persist raw credentials in normal config.
- Use only existing runtime dependencies; `requests` already belongs to InvokeAI.
- Preserve Canvas model capability filtering and external model installation behavior.
- Use fal.ai queue REST endpoints and fal CDN upload REST endpoints, not a new Python client dependency.
- Keep current standalone custom-node integration separate; native provider covers Canvas image generation/editing only.

### Task 1: Provider contract and configuration

**Files:** `invokeai/app/services/config/config_default.py`, `invokeai/app/api/dependencies.py`, `invokeai/app/api/routers/app_info.py`, `invokeai/app/services/external_generation/providers/__init__.py`, tests for config/API/provider registration.

- [x] Add `external_fal_api_key` and `external_fal_base_url` to provider config fields and API mapping.
- [x] Register `FalProvider` in service construction and export it.
- [x] Add tests proving fal appears in provider config/status APIs and secret redaction remains intact.

### Task 2: fal.ai REST adapter

**Files:** `invokeai/app/services/external_generation/providers/fal.py`, `tests/app/services/external_generation/test_fal_provider.py`.

- [x] Write failing tests for configuration, queue submission, status polling, result parsing, image upload, Flux payload mapping, Kontext payload mapping, Fill mask inversion, HTTP errors, rate limits, and download size limits.
- [x] Implement upload initiation (`rest.fal.ai/storage/upload/initiate`), PUT upload, queue submit/status/result calls, bounded polling, HTTPS image download, and model-specific payload builders.
- [x] Parse fal image URL outputs and provider seed/request metadata without exposing credentials.

### Task 3: Native invocation and curated models

**Files:** `invokeai/app/invocations/external_image_generation.py`, `invokeai/backend/model_manager/starter_models.py`, `tests/app/invocations/test_external_image_generation.py`, `tests/backend/model_manager/test_starter_models.py` or focused tests.

- [x] Add `FalImageGenerationInvocation` with provider filter `fal`.
- [x] Add starter models for `fal-ai/flux/schnell`, `fal-ai/flux/dev`, `fal-ai/flux-pro/kontext`, and `fal-ai/flux-lora-fill` with accurate modes, image requirements, aspect ratios, seed/batch capabilities, and default settings.
- [x] Ensure external starter sync installs these models after fal credentials are configured.

### Task 4: Canvas and frontend contract

**Files:** `invokeai/frontend/web/src/features/nodes/util/graph/generation/buildExternalGraph.ts`, `invokeai/frontend/web/src/features/modelManagerV2/subpanels/AddModelPanel/ExternalProviders/ExternalProvidersForm.tsx`, locale files, generated OpenAPI/type files, frontend tests.

- [x] Map provider `fal` to `fal_image_generation` in Canvas graph construction.
- [x] Add fal provider ordering/icon fallback and localized provider wording.
- [x] Regenerate OpenAPI/type artifacts and test that the generated graph uses fal node for txt2img/img2img/inpaint.

### Task 5: Documentation, validation, and PR

**Files:** `docs/src/content/docs/features/External Models/index.mdx`, new fal provider docs, changelog/PR description as appropriate.

- [ ] Document setup, supported models, Canvas usage, API cost warning, and current scope.
- [ ] Run focused Python tests, frontend tests/typecheck, ruff, OpenAPI/typegen checks, and a server smoke test without inference.
- [ ] Create fork, push branch, open PR with test evidence and explicit note that no paid inference was submitted.
