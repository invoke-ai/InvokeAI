# Task 5 Report: Canonical mixed-media selection and cross-project pruning

## Status

Complete.

Task 5 now persists canonical mixed-media selection state without renaming the
existing persisted properties:

- `selectedImage` writes a canonical `GalleryItem`.
- `selectedImageName` and `selectedImageNames` write qualified
  `GalleryItemKey` values.
- `compareImage` writes only a canonical `GalleryImageItem`.
- `recentImages` remains an image-only `GeneratedImageContract[]`.

The state projection, reducer actions, shared query/cache path, image-only
consumer boundaries, and cross-project reconciliation logic all use the
qualified mixed-media contract.

## TDD evidence

### RED

The first focused run was made after adding the selection, projection,
consumer, and reducer assertions, before implementing the mixed state:

```sh
pnpm test \
  src/features/gallery/core/selection.test.ts \
  src/features/gallery/ui/galleryStateView.test.ts \
  src/workbench/workbenchState.test.ts \
  src/app/gallerySelectionConsumers.test.ts
```

Result:

```text
Test Files 4 failed (4)
Tests 17 failed | 173 passed (190)
```

The failures were the intended missing-contract failures: no canonical item
reader, old image-named projection fields, missing mixed reducer action types,
unqualified selection writes, comparison surviving video selection, and direct
Generation/Workflow consumers not exposing the image-only reader boundary.

After the reducer action candidates could execute, the narrowed reducer RED run
showed the three intended behavioral failures:

```sh
pnpm test src/workbench/workbenchState.test.ts \
  -t "writes canonical mixed|prunes qualified|reconciles authoritative"
```

```text
Tests 3 failed | 155 skipped (158)
```

### GREEN

The final focused regression run after implementation and self-review was:

```sh
pnpm test \
  src/workbench/workbenchState.test.ts \
  src/features/gallery/core/selection.test.ts \
  src/features/gallery/ui/galleryStateView.test.ts \
  src/features/gallery/data/queryCache.test.ts \
  src/app/gallerySelectionConsumers.test.ts
```

```text
Test Files 5 passed (5)
Tests 195 passed (195)
```

An earlier broader focused run also covered mixed query/data overlays and the
image-only compatibility views:

```text
Test Files 9 passed (9)
Tests 211 passed (211)
```

## Implementation

### Selection readers

- Added `getSelectedGalleryItemFromValues`.
- Canonical image and video items round-trip unchanged.
- Legacy generated-image objects and bare/qualified image names adapt to
  canonical image items.
- `getSelectedGalleryImageFromValues` delegates to the item reader, narrows by
  `kind`, and returns `null` for a selected video.
- Generation UI and Workflow field input selectors now use that image-only
  reader directly.

### State projection and data

- Renamed the canonical projection fields to `items`, `selectedItemKey`,
  `selectedItemKeys`, and `compareImageKey`.
- `GalleryCurrentItem` now identifies an item by qualified key.
- Persisted bare names are canonicalized as image keys; image/video items with
  the same name remain distinct.
- Mixed selection order is preserved.
- Gallery data and Preview now consume `galleryItemsInfiniteOptions` and
  `GalleryItemsPage`.
- The current image-only Grid and Preview paths use explicit narrowing adapters
  that filter with `isGalleryImageItem` before constructing `GalleryImage`
  contracts.

### Reducer and commands

- Renamed internal actions to:
  - `selectGalleryItem`
  - `toggleGalleryItemInSelection`
  - `setGalleryMultiSelection`
  - `patchGalleryItems`
  - `removeGalleryItems`
- Added canonical item commands alongside narrow typed image adapters for the
  Task 6/8 callers.
- Selecting or making a video primary clears comparison.
- Queue result routing keeps recents image-only while writing a canonical
  selected image and qualified selection keys.
- Patch operations migrate legacy selected/compare objects to canonical items
  while retaining the legacy recent-image shape.

### Pruning and board outcomes

- Qualified-key patch/removal runs across every open project.
- A video key never removes a same-name recent image or upscale image input.
- Image keys prune image-only recents and upscale inputs across all projects.
- The canonical board reconciliation action accepts authoritative
  deleted/failed image/video arrays.
- Confirmed deletes are removed.
- Every other locally known item on the deleted board—including explicitly
  failed items—is moved to `none`.
- Selected/project board references and pending/running queue destinations are
  repaired across projects.

### Legacy query/cache cleanup

Removed all Task 4 compatibility debt assigned to Task 5:

- `gallery/legacy-images/list` and `galleryKeys.legacy*`
- `galleryImagesInfiniteOptions`
- `flattenGalleryImagesData`
- legacy image query discovery/key guards and type aliases
- `patchGalleryImageCaches` and `GalleryImageCachePatch`
- `invalidateGalleryImages`
- `mergeGalleryImageWindow`
- the corresponding `features/gallery/queries.ts` facade exports

Image actions now translate their image-only inputs into qualified
`GalleryItemMutationResult` values and patch the shared mixed item cache.

## Verification

From `invokeai/frontend/webv2`:

```sh
pnpm test
```

```text
Test Files 374 passed (374)
Tests 4938 passed (4938)
```

```sh
pnpm test:browser
```

```text
Test Files 58 passed (58)
Tests 240 passed (240)
```

The browser suite emitted its existing React `act(...)`, missing-i18n-instance,
and intentional error-boundary console output; the command exited successfully.

```sh
pnpm test:fixtures
```

```text
Tests 4 passed (4)
```

```sh
pnpm format:check
pnpm lint:oxc
pnpm lint:tsc
pnpm architecture:check
```

```text
format: all matched files correctly formatted
oxlint: zero warnings/errors
tsc --noEmit: passed
architecture: 3 files passed, 34 tests passed
```

From the repository root:

```sh
uv run pytest -q tests/app/services/gallery/test_gallery_default.py
```

```text
17 passed, 4 environment/deprecation warnings
```

## Compatibility obligations

The remaining adapters are deliberately narrow and type-safe:

1. `getGalleryImageStateView` is marked `TODO(Task 6)` and filters canonical
   items before serving the current image-only Grid.
2. Image selection command adapters are marked `TODO(Task 6/8)` and convert
   `GeneratedImageContract & Partial<GalleryImage>` to `GalleryImageItem`
   before dispatch.
3. Preview's mixed-query image projection is marked `TODO(Task 8)` and filters
   before converting.
4. The boolean board-deletion adapter is marked `TODO(Task 7)`; the canonical
   outcome command and reducer already require the four authoritative arrays.
5. `CurrentImageFlowNode` remains intentionally image-only by reading only
   `recentImages`.

No adapter accepts a video as an image or relies on an unsafe mixed-to-image
cast.

## Self-review and concerns

- Confirmed every new reducer action is strongly typed in the internal action
  union; the RED-only candidate-action test cast was removed after GREEN.
- Confirmed all new selection writes are canonical and all selected-name writes
  are qualified.
- Confirmed comparison cannot be set to a video through the reducer or public
  commands and is cleared when a video becomes primary.
- Confirmed same-name image/video pruning is kind-qualified across projects.
- Confirmed board reconciliation treats only deleted arrays as confirmed
  deletion and moves every local non-deleted survivor.
- Confirmed all marked Task 5/7 legacy query/cache symbols are absent from
  production and tests.
- Confirmed no `useEffect` was added.
- No unresolved implementation concern.

## Files

### Production

- `invokeai/frontend/webv2/src/app/GenerationUiAdapter.tsx`
- `invokeai/frontend/webv2/src/features/gallery/contracts.ts`
- `invokeai/frontend/webv2/src/features/gallery/core/selection.ts`
- `invokeai/frontend/webv2/src/features/gallery/data/queries.ts`
- `invokeai/frontend/webv2/src/features/gallery/data/queryCache.ts`
- `invokeai/frontend/webv2/src/features/gallery/queries.ts`
- `invokeai/frontend/webv2/src/features/gallery/ui/GalleryImageGrid.tsx`
- `invokeai/frontend/webv2/src/features/gallery/ui/GalleryWidgetView.tsx`
- `invokeai/frontend/webv2/src/features/gallery/ui/galleryStateView.ts`
- `invokeai/frontend/webv2/src/features/gallery/ui/useGalleryData.ts`
- `invokeai/frontend/webv2/src/features/workflow/ui/fields/WorkflowFieldInput.tsx`
- `invokeai/frontend/webv2/src/workbench/image-actions/useImageActions.ts`
- `invokeai/frontend/webv2/src/workbench/widgets/layers/RunLayerWorkflowDialog.tsx`
- `invokeai/frontend/webv2/src/workbench/widgets/preview/PreviewWidgetView.tsx`
- `invokeai/frontend/webv2/src/workbench/workbenchState.ts`
- `invokeai/frontend/webv2/src/workbench/workbenchStore.ts`

### Tests

- `invokeai/frontend/webv2/src/app/gallerySelectionConsumers.test.ts`
- `invokeai/frontend/webv2/src/features/gallery/core/selection.test.ts`
- `invokeai/frontend/webv2/src/features/gallery/data/queryCache.test.ts`
- `invokeai/frontend/webv2/src/features/gallery/ui/galleryStateView.test.ts`
- `invokeai/frontend/webv2/src/features/gallery/ui/useGalleryData.test.ts`
- `invokeai/frontend/webv2/src/workbench/widgets/preview/PreviewNavigation.browser.test.tsx`
- `invokeai/frontend/webv2/src/workbench/workbenchState.test.ts`
