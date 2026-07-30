# Task 7B Report: Mixed-media gallery actions and downloads

## Status

Complete.

Task 7B consumes the reviewed Task 7A `galleryItemOrganization` result and
finishes Task 7's App/action/UI integration. Gallery now receives an honest
common `GalleryItemActions` port, while App-owned `ImageActions` retains the
strict `GalleryImage` operations. Mixed image/video selections can be moved,
starred, deleted, and downloaded without narrowing or unqualified-name
collisions.

Confirmed mutation results are applied once at the operation boundary: only
`result.succeeded` patches caches and Workbench state, one all-gallery
invalidation follows the confirmed patch, and one success/partial/failure
notification is emitted.

## TDD evidence

All frontend commands ran from `invokeai/frontend/webv2`.

### Common action port and confirmed mutations

The first focused RED was:

```sh
pnpm test:browser src/workbench/image-actions/useImageActions.browser.test.tsx
```

```text
Test Files 1 failed (1)
Tests 6 failed | 2 passed (8)
```

The failures covered the then-missing common delete/download/move/star methods,
confirmed-only patching, same-name image/video identity, mixed download order,
continuation, and operation-level invalidation/notification. The first common
action implementation made the file green with 8 tests.

The lazy App bridge then captured RED for the canonical mixed target and real
mixed board move:

```text
Test Files 1 failed (1)
Tests 2 failed (2)
```

After removing the unsafe cast and image-only move bridge, it passed with 2
tests. A later live-successor-context regression failed 1 of 3 tests before
the getter was forwarded. The honest item-named provider/export cleanup was
also test-first: all 3 bridge tests failed against the missing item-aware
symbols, then all 3 passed.

### Context menus and keyboard actions

The context-menu regression run was:

```sh
pnpm test:browser src/workbench/image-actions/ImageContextMenu.browser.test.tsx
```

```text
Test Files 1 failed (1)
Tests 2 failed | 4 passed (6)
```

It showed that single-video and mixed selections had no common menu. The green
run passed all 6 tests with image-only actions hidden for video,
mixed-media, and unresolved selections.

The grid/hotkey regression run was:

```sh
pnpm test:browser src/features/gallery/ui/GalleryImageGrid.browser.test.tsx
```

```text
Test Files 1 failed (1)
Tests 2 failed | 12 passed (14)
```

It showed that mixed hotkeys were rejected and videos had no star affordance.
The green run passed all 14 tests with qualified refs, the required
inside/outside right-click semantics, unloaded selection preservation, and
same-name media isolation.

### Delete successor guards

Adding ordered predecessor/successor, unloaded resolution, failed-ref, and
stale filter/selection cases produced:

```text
Test Files 1 failed (1)
Tests 5 failed | 8 passed (13)
```

All 13 passed after the ordered-names implementation. Two additional
regressions were captured and fixed independently:

- a captured names list without the deleted primary attempted to promote an
  unrelated item;
- an account rotation while `galleryItems.resolve(ref)` was pending still
  applied confirmed deletion side effects.

The stale-account test failed 1 of 18 tests before the post-resolution abort
guard and then passed all 18.

### Downloads and board archive wording

The board preparation-toast regression failed 1 of 3
`useGalleryActions.browser.test.tsx` tests against the old generic archive
message. The new board-menu regression failed its only test against
`Download Board`. Both passed after the exact `board.videoCount` omission
wording was added without a fetch.

Download regressions cover:

- one existing archive for an image-only selection;
- a protected single-video `fullUrl` download using the actual name;
- image archive first, then sequential videos for a mixed selection;
- continuation after rejected or unresolved videos;
- one partial/skipped summary.

### Final focused GREEN

```sh
pnpm test \
  src/features/gallery/ui/GalleryBoardSelect.test.ts \
  src/workbench/image-actions/ImageContextMenu.test.ts \
  src/features/gallery/ui/GalleryWidgetView.test.ts \
  src/features/gallery/ui/useGalleryData.test.ts \
  src/features/gallery/data/queryCache.test.ts \
  src/workbench/workbenchState.test.ts
```

```text
Test Files 6 passed (6)
Tests 183 passed (183)
```

```sh
pnpm test:browser \
  src/app/GalleryImageActionsBridge.browser.test.tsx \
  src/workbench/image-actions/useImageActions.browser.test.tsx \
  src/workbench/image-actions/ImageContextMenu.browser.test.tsx \
  src/features/gallery/ui/GalleryImageGrid.browser.test.tsx \
  src/features/gallery/ui/useGalleryActions.browser.test.tsx \
  src/features/gallery/ui/GalleryBoardMenu.browser.test.tsx
```

```text
Test Files 6 passed (6)
Tests 45 passed (45)
```

## Implementation

### Honest common and image-only ports

- `GalleryItemActions` accepts ordered qualified refs and canonical loaded
  items for delete, download, move, preview/new-tab, and star operations.
- Gallery's provider, hook, widget context, and adapter field are item-named
  and expose only that common surface.
- App `ImageActions` extends the common port and keeps canvas, reference,
  compare, clipboard, prompt-template, and metadata-recall inputs strictly
  `GalleryImage`.
- The lazy App bridge keeps the full `ImageActions` only in an App-local
  context for the image-capable context menu. The retained
  `as unknown as ImageActions` cast and all-image mixed-board bridge are gone.

### Confirmed operation boundary

- Delete, move, star, and unstar call the Task 7A
  `galleryItemOrganization` methods directly.
- Cache and Workbench changes use only `result.succeeded` qualified keys;
  `result.failed` remains materialized and selected.
- Every completed operation attempts exactly one `invalidateGallery` after
  confirmed patches and emits one localized success, partial, or failure
  notification.
- Account ownership is checked before patches, after async successor
  resolution, before invalidation, and before notification.

### Primary selection after deletion

- The current filter and selection plus materialized items are captured before
  transport.
- The ordered names query is captured before deletion and searched for the
  nearest surviving predecessor, then successor.
- Every confirmed deletion and failed deletion is ineligible for promotion.
- An unloaded candidate resolves through `galleryItems.resolve(ref, signal)`.
- Failed names lookup falls back to materialized order.
- Account, filter, and selection currency are rechecked before selecting an
  asynchronously resolved candidate.

### UI actions

- Board drops forward the complete ordered image/video/mixed ref payload.
- Keyboard delete/star uses common refs, including unloaded selected refs.
- Right-click outside a selection targets one item; right-click inside a
  multi-selection keeps its complete ordered qualified selection.
- Single-video and mixed/unresolved menus show only supported common actions.
  Fully resolved image-only targets retain the existing strict image menu.
- Video thumbnails now expose the shared star affordance. Compare and every
  other image-only path remain image-gated.

### Downloads and archives

- Single video uses its protected `fullUrl` and actual file name.
- Image selections reuse the existing single archive transport.
- Mixed selection downloads the image archive first, then video files
  sequentially in video-selection order.
- Rejected or unresolved videos do not prevent later videos from downloading;
  the operation emits one summary.
- Board archive remains image-only. Its menu label and preparation toast state
  the exact omitted video count from the already-loaded board; no count fetch
  was added.

## Full verification

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
Test Files 376 passed (376)
Tests 4973 passed (4973)
```

```sh
pnpm test:browser
```

```text
Test Files 64 passed (64)
Tests 290 passed (290)
```

The browser suite emitted its existing React `act(...)`,
missing-i18n-instance, and intentional error-boundary diagnostics and exited
successfully.

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

## Self-review

- Confirmed no production module or production file rename was added; the one
  new source-tree file is a browser test.
- Confirmed no `useEffect`, platform UI barrel importer, migration exception,
  or performance baseline was added.
- Confirmed `GeneratedImageContract` and the strict image-only action inputs
  remain unchanged.
- Confirmed there is no mixed-to-image cast in the lazy bridge.
- Confirmed every mutation carries qualified identity end to end and only
  confirmed successes can patch or prune state.
- Confirmed one operation cannot issue per-partition invalidations or
  notifications.
- Confirmed `git diff --check` is clean.

## Retained Task 8–10 seams

- Task 8 retains native video rendering/controls in Preview. Task 7B opens the
  selected video item and leaves the current image-only preview renderer
  unchanged.
- Task 9 retains preview-only copy-current-frame and video Details/metadata.
  Video and mixed gallery menus deliberately hide the existing image clipboard
  and metadata-recall actions.
- Task 10 retains video upload, mock-backend fixture, broad accessibility, and
  full gallery translation follow-through. Task 7B neither widens the current
  image upload accept list nor changes mock fixtures or accessibility
  exceptions.

## Concerns

None.
