# Task 6 Report: Mixed-media DnD and virtualized gallery grid

## Status

Complete.

Task 6 replaces the gallery-image drag contract with ordered
`GalleryItemRef[]` payloads, forwards image/video/mixed board drops through a
typed item intent, and converts the existing virtualized grid to canonical
mixed-media items. Static video poster cells, qualified item identity, lazy
mixed range selection, and strict image-only drop refinements are covered by
focused browser and unit tests.

The App compatibility layer does not claim mixed board movement: it receives
the complete ordered ref vector from the board UI, handles an all-image vector
with the current image action, and rejects video/mixed vectors intact pending
Task 7.

## TDD evidence

### RED

The first DnD contract run was made after adding the item payload, qualified-ID,
guard, refinement, and board-resolution assertions:

```sh
pnpm test src/features/gallery/ui/galleryDnd.test.ts
```

```text
Test Files 1 failed (1)
Tests 6 failed | 1 passed (7)
```

The failures were the intended missing-contract failures:
`getGalleryItemDragData`, `getGalleryItemDragId`, the item guard, the all-image
refinement shape, and canonical board filtering did not yet exist.

The first board/consumer run was:

```sh
pnpm test \
  src/features/gallery/ui/GalleryBoardSelect.test.ts \
  src/workbench/widgets/canvas/canvasImageDropLayout.test.ts \
  src/workbench/widgets/preview/previewCompareDnd.test.ts \
  src/workbench/widgetDnd.test.ts
```

```text
Test Files 4 failed (4)
Tests 6 failed | 11 passed (17)
```

The missing item utility exports and board forwarding seam caused the expected
failures. The initial Preview drag browser run also failed at module import
because `PreviewFrame` still used the removed image constructor:

```sh
pnpm test:browser \
  src/workbench/widgets/preview/PreviewFrameDrag.browser.test.tsx \
  src/workbench/widgets/preview/PreviewCompareDropZone.browser.test.tsx
```

The initial Grid browser run likewise failed at module import while
`GalleryImageGrid` still depended on the removed image constructor:

```sh
pnpm test:browser src/features/gallery/ui/GalleryImageGrid.browser.test.tsx
```

### GREEN

The final focused unit regression run was:

```sh
pnpm test \
  src/features/gallery/ui/galleryDnd.test.ts \
  src/features/gallery/ui/GalleryBoardSelect.test.ts \
  src/features/gallery/ui/galleryStateView.test.ts \
  src/features/gallery/ui/useGalleryData.test.ts \
  src/workbench/widgets/canvas/canvasImageDropLayout.test.ts \
  src/workbench/widgets/preview/previewCompareDnd.test.ts \
  src/workbench/widgetDnd.test.ts
```

```text
Test Files 7 passed (7)
Tests 69 passed (69)
```

The final focused browser regression run was:

```sh
pnpm test:browser \
  src/features/gallery/ui/GalleryImageGrid.browser.test.tsx \
  src/features/gallery/ui/useGalleryActions.browser.test.tsx \
  src/workbench/widgets/preview/PreviewFrameDrag.browser.test.tsx \
  src/workbench/widgets/preview/PreviewCompareDropZone.browser.test.tsx
```

```text
Test Files 4 passed (4)
Tests 15 passed (15)
```

## Implementation

### Shared DnD contract

- Added `GalleryItemDragData` with `kind: 'gallery-item'` and an ordered
  `GalleryItemRef[]`.
- Draggable IDs are exact `GalleryItemKey` values.
- Added the non-empty item guard and retained
  `isGalleryImageDragData()` as a non-empty all-image refinement whose narrowed
  items expose image names without a media cast.
- Board resolution accepts image, video, and mixed payloads, preserves ref
  order, rejects virtual/date board targets, and excludes already-present refs
  through canonical loaded items keyed by kind and name.
- `GalleryBoardSelect` forwards the complete resolved ref vector to
  `moveItemsToBoard`; it never strips a video subset before forwarding.
- Canvas, Upscale, reference-image, Regional Guidance, and Preview comparison
  consumers use the all-image refinement and therefore reject video/mixed
  payloads.
- Grid, Preview frame, and Preview filmstrip sources now emit the item
  contract. The current native Preview remains image-only until Task 8.

### Canonical virtualized grid

- Removed `GalleryImageStateView` and `getGalleryImageStateView`.
- Kept `GalleryImageGrid.tsx` and exactly two cell arms:
  canonical item and queue placeholder.
- Selection, context targets, hotkey targets, React keys, and draggable IDs use
  qualified item identity, so same-name images and videos remain independent.
- Preserved square cells, constant row estimates, viewport-width seeding,
  scroll-to-index navigation, overscan `4`, the near-end infinite-load trigger,
  and image-only queue placeholders.
- Shift-click lazily fetches ordered item refs on normal boards and reads the
  existing item-name cache for date boards. It captures account epoch, filter
  identity, and the selection anchor before awaiting, ignores stale results,
  and falls back to the materialized range on cache misses or request failure.
- Alt-click sets comparison only for images. Alt-clicking a video performs
  normal item selection, allowing the canonical reducer to clear comparison.
- Right-click outside the current selection forms a target containing only the
  clicked canonical item.

### Video poster cell

- Renders the static thumbnail with an ordinary `<img decoding="async">`.
- Adds no `loading="lazy"` and no grid `<video>`.
- Shows an always-visible decorative Play icon and formatted duration badge in
  the lower badge position.
- Uses tabular numerals and the required opacity-only transition.
- Uses a localized video label containing both the item name and formatted
  duration.
- Renamed the list label to `Gallery items` and updated the accessibility
  journey selector.
- Image thumbnail, dimensions, and star affordances retain their existing
  behavior.

## Verification

All commands ran from `invokeai/frontend/webv2`.

```sh
pnpm lint:tsc
```

```text
tsc --noEmit: passed
```

```sh
pnpm test
```

```text
Test Files 375 passed (375)
Tests 4951 passed (4951)
```

```sh
pnpm test:browser
```

```text
Test Files 61 passed (61)
Tests 254 passed (254)
```

The browser suite emitted its existing React `act(...)`, missing-i18n-instance,
and intentional error-boundary diagnostics; the command exited successfully.

```sh
pnpm test:fixtures
```

```text
Tests 4 passed (4)
```

```sh
pnpm architecture:check
```

```text
Test Files 3 passed (3)
Tests 34 passed (34)
```

```sh
pnpm lint
```

```text
format: all matched files correctly formatted
oxlint: zero warnings/errors
tsc --noEmit: passed
architecture: 3 files passed, 34 tests passed
```

## Intentionally retained Task 7/8 adapters

1. `GalleryImageActions.moveItemsToBoard` is a narrow Task 7 bridge. The board
   UI forwards the full ordered mixed ref vector; the App adapter handles only
   an all-image vector and rejects video/mixed vectors without moving an image
   subset.
2. The canonical Gallery context target is converted to the current
   `ImageContextMenu` target only when every target item is an image. Video and
   mixed targets are rejected pending the Task 7 common item action menu.
3. Grid delete/star hotkeys still call the image-action port only for a fully
   resolved all-image selection. Video, mixed, and unresolved selections are
   rejected pending Task 7.
4. Workbench `patchImages`/`removeImages` and the image-action methods remain
   strict image adapters pending the Task 7 mixed mutation/action port.
5. Legacy Workbench selection commands (`selectImage`, `setCompareImage`,
   `setMultiSelection`, and `toggleImageSelection`) remain for Preview and the
   image command palette. Grid now uses canonical commands; the remaining
   adapters are Task 8 cleanup.
6. Preview still narrows the mixed gallery query to images through
   `flattenPreviewImages` and constructs an image ref for the selected saved
   frame. Native video preview/navigation and the main preview video
   non-draggable boundary remain Task 8 work.
7. Canvas, Upscale, reference-image, Regional Guidance, workflow collision, and
   comparison targets remain intentionally image-only. Their shared refinement
   accepts only non-empty all-image item payloads.

No retained adapter converts or casts a video to an image.

## Self-review

- Confirmed `GeneratedImageContract` is byte-for-byte unchanged from Task 5.
- Confirmed no image-only contract gained optional media fields.
- Confirmed no production file was added or renamed.
- Confirmed no `useEffect` was added.
- Confirmed no Task 6 production change adds `<video>` or lazy image loading.
- Confirmed `migrationExceptions.ts` and performance baselines are untouched.
- Confirmed old DnD constructors/IDs and the Task 5 image grid projection are
  absent; the only old `gallery-image` payload literal is a negative guard
  regression test.
- Confirmed `git diff --check` is clean.
- No unresolved implementation concern.
