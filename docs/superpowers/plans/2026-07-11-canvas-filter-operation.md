# Canvas Filter Operation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move raster and control filtering into one guarded engine-owned canvas operation with bottom-canvas controls and atomic apply/save behavior.

**Architecture:** A discriminated filter operation session owns the target guard, initial persisted settings, local draft, request status, error, and preview. The engine exposes launch/update/process/reset/apply/save/cancel methods and one external-store snapshot; React only subscribes and renders `FilterOptions`. Existing graph runners remain pure, while the engine owns guarded preview publication and atomic document/cache/history commits.

**Tech Stack:** TypeScript 6, React 19, Chakra UI 3, Vitest, engine external stores.

## Global Constraints

- Work on `feat/canvas` and never edit, stage, or commit `CANVAS_PLAN.md`.
- Use test-driven development and no direct React `useEffect`.
- Filter preview lifetime belongs to the engine session, not a popover/component mount.
- Raster and control operations share exact project/layer/source guards and are mutually exclusive with SAM.
- Apply and Save As must remain retryable after durability or commit failures.
- The final commit message is `feat(canvas): move layer filtering into canvas controls`.

---

### Task 1: Filter Session State Machine

**Files:**
- Create: `invokeai/frontend/webv2/src/workbench/widgets/layers/filterOperationSession.ts`
- Test: `invokeai/frontend/webv2/src/workbench/widgets/layers/filterOperationSession.test.ts`
- Modify: `invokeai/frontend/webv2/src/workbench/canvas-engine/canvasOperationController.ts`

**Interfaces:**
- Consumes: `CanvasOperationController`, `LayerExportGuard`, existing filter runners.
- Produces: `FilterOperationSession`, `FilterOperationSessionState`, guarded latest-request-wins process/reset/cancel/commit methods.

- [ ] Write failing tests for start snapshots, raster/control processing, rapid requests, reset, cancel, stale guards, retryable failures, and SAM replacement.
- [ ] Run the focused Vitest files and confirm the new tests fail for missing APIs.
- [ ] Implement the smallest React-free session state machine using the operation controller.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Engine-Owned Atomic Commit

**Files:**
- Modify: `invokeai/frontend/webv2/src/workbench/canvas-engine/engine.ts`
- Modify: `invokeai/frontend/webv2/src/workbench/canvas-engine/engine.test.ts`
- Modify: `invokeai/frontend/webv2/src/workbench/types.ts`
- Modify: `invokeai/frontend/webv2/src/workbench/workbenchState.ts`

**Interfaces:**
- Consumes: filter preview image, target guard, content rect/origin, draft filter settings.
- Produces: engine filter-session API and one failure-atomic history entry for replace or raster/control insertion.

- [ ] Add failing engine tests for guarded raster/control preview, nonzero-origin replacement undo/redo, settings persistence, both Save As targets, stale/busy failures, and durability/commit retry behavior.
- [ ] Run the engine test selection and confirm red failures.
- [ ] Extend the guarded commit path to raster/control and commit source, transform correction, settings, cache, and history atomically.
- [ ] Wire engine lifecycle invalidation, interaction locking, and session external-store publication.
- [ ] Run focused engine/session tests and confirm green.

### Task 3: Launch and Bottom Controls

**Files:**
- Create: `invokeai/frontend/webv2/src/workbench/widgets/canvas/tool-options/FilterOptions.tsx`
- Create: `invokeai/frontend/webv2/src/workbench/widgets/canvas/tool-options/FilterOptions.test.ts`
- Modify: `invokeai/frontend/webv2/src/workbench/widgets/canvas/tool-options/CanvasOperationBar.tsx`
- Modify: `invokeai/frontend/webv2/src/workbench/widgets/canvas/engineStoreHooks.ts`
- Modify: `invokeai/frontend/webv2/src/workbench/widgets/layers/LayerContextMenu.tsx`
- Modify: `invokeai/frontend/webv2/src/workbench/widgets/layers/layerContextActions.ts`
- Modify: `invokeai/frontend/webv2/src/workbench/widgets/layers/RasterLayerFilterSection.tsx`
- Modify: `invokeai/frontend/webv2/src/workbench/widgets/layers/ControlLayerSettings.tsx`

**Interfaces:**
- Consumes: engine filter-session snapshot and methods.
- Produces: explicit `CanvasOperationBar` filter branch; Process, Reset, Apply, Save As raster/control, and Cancel controls.

- [ ] Add failing pure eligibility/view-model and context-action tests for busy/disabled states and direct operation launch.
- [ ] Run the focused widget tests and confirm red failures.
- [ ] Implement `FilterOptions` with `LayerFilterControls`; make properties buttons and context action call `engine.startFilterOperation(layerId)`.
- [ ] Remove popover-owned preview/commit lifecycle and ensure component unmount does not affect the session.
- [ ] Run focused widget/session tests and confirm green.

### Task 4: Consolidation and Verification

**Files:**
- Delete after migration: `invokeai/frontend/webv2/src/workbench/widgets/layers/layerFilterController.ts`
- Delete after migration: `invokeai/frontend/webv2/src/workbench/widgets/layers/layerFilterController.test.ts`
- Delete after migration if fully superseded: `invokeai/frontend/webv2/src/workbench/widgets/layers/controlFilterPreview.ts`
- Delete after migration if fully superseded: `invokeai/frontend/webv2/src/workbench/widgets/layers/controlFilterPreview.test.ts`
- Preserve: `invokeai/frontend/webv2/src/workbench/widgets/layers/layerFilterRunner.ts`

**Interfaces:**
- Consumes: completed engine/session/UI migration.
- Produces: one filter path with no unguarded control preview lifecycle.

- [ ] Remove only superseded orchestration while retaining reusable graph/runner code.
- [ ] Run focused tests, full `pnpm test`, `pnpm lint`, and `pnpm build` from `invokeai/frontend/webv2`.
- [ ] Inspect `git diff`, self-review guards/history/error retry behavior, and verify `CANVAS_PLAN.md` remains untracked.
- [ ] Stage only intended files and commit with the required message.
