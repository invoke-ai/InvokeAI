# Task 9 Report: Video copy-current-frame and raw Details

## Status

Complete.

Preview videos now expose a readiness-gated Copy Current Frame action and
video Details from the header action strip, Preview context menu, and footer
disclosure. Frame capture stays inside the existing keyed `PreviewVideo`
boundary in `PreviewFrame.tsx`, which remains the sole owner of a production
`<video>`.

Details work is query-driven only while the disclosure is open. Query identity
includes the account epoch and qualified `GalleryItemKey`; video metadata and
workflow/graph requests start concurrently with the same abort signal. Images
retain parsed metadata rows, source-run identity, and existing recall actions,
while videos render raw Metadata, Workflow, and Graph JSON with no recall
controls.

## TDD evidence

All frontend commands ran from `invokeai/frontend/webv2`.

### Copy-current-frame controller

The first focused RED extended the existing native-video browser contract:

```sh
pnpm test:browser src/workbench/widgets/preview/PreviewFrameVideo.browser.test.tsx
```

```text
Test Files 1 failed (1)
Tests 9 failed | 8 passed (17)
```

Those failures covered:

- Clipboard API support, `HAVE_CURRENT_DATA`, and nonzero intrinsic dimensions;
- exact intrinsic-size canvas drawing and PNG `ClipboardItem` writes;
- unsupported, not-ready, canvas draw/taint, null/thrown encoding, and
  ClipboardItem/write failures;
- account-epoch and selected-item staleness after an asynchronous clipboard
  write.

After the local keyed video controller was implemented, the file passed all 17
tests.

### Query-driven Details and Preview-only actions

Before production edits, the consolidated Task 9 browser RED was:

```text
Test Files 4 failed (4)
Tests 18 failed | 20 passed (38)
```

The new Details file required closed-state inactivity, epoch + qualified-item
query keys, concurrent video requests with one query signal, close/item/epoch
abort and stale-result isolation, raw JSON tabs without recall, and the parsed
image metadata/recall regression. It passed all 7 tests after implementation.

The action-strip and shared-menu tests required localized Preview video
actions while retaining image copy and keeping Gallery video menus free of
frame copy. The focused action/menu set passed all 14 tests.

The notification translator was added from a focused unit RED:

```text
Test Files 1 failed (1)
Tests 7 failed | 5 passed (12)
```

It then passed all 12 tests, proving one success notice and distinct localized
notices for all six failure results. The combined Task 9 browser set passed:

```text
Test Files 4 passed (4)
Tests 38 passed (38)
```

### Full-suite timing regression

The first full browser run exposed a timing-sensitive assertion in the
pre-existing protected-video retry test:

```text
Test Files 1 failed | 66 passed (67)
Tests 1 failed | 327 passed (328)
```

Its intentionally invalid data-video URL could emit a trusted native media
error after the test's deterministic manual error when the full suite was
busy. That second event correctly entered the terminal retry state, racing the
first-retry assertion. The final harness uses a minimal valid media URL for
tests that dispatch deterministic recovery events and reserves an invalid URL
for one focused native-error contract. No trusted media error is suppressed.

### Completion-review follow-up

The completion review found no critical issues and three important edge cases.
All three were captured before follow-up production edits:

```text
Test Files 3 failed (3)
Tests 4 failed | 34 passed (38)
```

The REDs proved that:

- image recall-capability metadata work did not yet receive the Details query
  signal;
- pending image Details had lost the prior disabled recall-button skeleton;
- a captured video A context menu could show actions bound to newly selected
  video B.

The recall-capability port now accepts the caller signal and combines it with
the account-lifecycle signal through `AbortSignal.any`. Image Details keep
their recall row mounted and disabled while pending. Preview video context
extras now carry their selected `GalleryItemKey` and render only when it
matches the menu's captured single-video target.

The same three files then passed all 38 tests. The Details file passed 9/9
without its prior `act(...)` diagnostics after its asynchronous query
assertions were wrapped correctly.

### Review-fix round

The next review found that image recall capabilities were cached under only
the account epoch and item key even though they also depend on current
generation values and model inventory. The focused regression kept the epoch,
item, and cached metadata fixed, changed the current capability action, and
failed because the Recall All button remained enabled:

```text
Test Files 1 failed (1)
Tests 1 failed | 9 skipped (10)
AssertionError: expected false to be true
```

The cache now owns server metadata only. `ImageActions` exposes a synchronous
`deriveImageRecallCapabilities(image, metadata)` operation backed by its
current generation values and model inventory, and Preview calls it on every
render after metadata resolves. The regression proves that the button updates
without another metadata request. A second hook-level test switches the
current model from SD-1 to FLUX and proves clip-skip capability changes without
starting a metadata transport.

The review also found that a test-host capture listener suppressed every
trusted media error in the native-video file. A focused browser-generated
error test failed because protected-media recovery was never called:

```text
Test Files 1 failed (1)
Tests 1 failed | 17 skipped (18)
AssertionError: expected "vi.fn()" to be called once, but got 0 times
```

The blanket listener was removed. Deterministic synthetic recovery tests use a
valid media data URL, while the focused invalid-source test observes an
`isTrusted` browser error and proves it reaches cookie refresh. The existing
deterministic recovery tests retain terminal failure-UI coverage. The three
directly affected browser files then passed:

```text
Test Files 3 passed (3)
Tests 49 passed (49)
```

## Implementation

### Preview-owned frame capture

- `PreviewFrame` exposes a typed imperative controller through the existing
  `PreviewFrame`/keyed `PreviewVideo` boundary.
- Readiness is published from media events and ref lifetime, with no
  synchronization effect.
- Capture verifies ClipboardItem/write support, current-frame data, and
  nonzero dimensions before drawing.
- The canvas uses `videoWidth`/`videoHeight`, draws the selected native player,
  encodes `image/png`, and performs one clipboard write.
- Canvas draw/taint, encoding, clipboard, unsupported, not-ready, and stale
  outcomes stay distinct.
- Account epoch, selected `GalleryItemKey`, and video element identity are
  checked before the write and again before success.
- `PreviewWidgetView` translates every result into exactly one localized
  success/error notification.

### Preview action surfaces

- Full-density Preview video headers expose Copy Current Frame and Video
  Details; copy remains disabled until the controller publishes readiness.
- Preview's shared context-menu port is opt-in and video-only. Gallery callers
  omit the port, so Gallery video/mixed context menus still hide frame copy.
- The opt-in port is qualified by `GalleryItemKey`; stale captured menus never
  operate on a newly selected video.
- Image copy, compare, metadata, and recall behavior remain on the strict
  image arm.
- Video Details explicitly opens the existing footer disclosure.

### Query-driven Details

- The old image metadata synchronization effect was removed.
- The query observer is mounted only while Details is open and is keyed by
  `['preview', 'details', accountEpoch, itemKey]`.
- Unmounting on close/item/epoch change cancels supported transports through
  the TanStack Query signal.
- The image query performs exactly one metadata transport. Recall capabilities
  are derived synchronously from that cached metadata and the current
  generation/model state, so they never become stale behind the item-only key.
- The asynchronous recall-capability port retained for context menus still
  combines caller and account-lifecycle signals.
- Video metadata and workflow/graph requests use `Promise.all` and the same
  signal.
- Video Metadata, Workflow, and Graph use `JsonPreview` from the direct
  `@platform/ui/JsonPreview` subpath and never render recall controls.
- Image Details retain `parsePreviewMetadata`, Source Run, recall capability
  loading, and a stable disabled-while-pending `RecallActionButtons` skeleton.

## Full verification

```sh
pnpm lint
```

```text
format: passed
OXC: passed
tsc --noEmit: passed
architecture: 3 files, 34 tests passed
```

```sh
pnpm test
```

```text
Test Files 376 passed (376)
Tests 4985 passed (4985)
```

```sh
pnpm test:browser
```

```text
Test Files 67 passed (67)
Tests 335 passed (335)
```

The full suite retained non-failing React `act(...)`, missing-i18n-instance,
and intentional error-boundary diagnostics from pre-existing files.

```sh
pnpm test:fixtures
```

```text
Tests 4 passed (4)
```

## Source-policy and contract scans

```sh
rg -n '^[[:space:]]*<video' src --glob '!**/*.test.*' --glob '!**/*.spec.*'
```

Only `src/workbench/widgets/preview/PreviewFrame.tsx` matched.

```sh
git diff -U0 -- '*.ts' '*.tsx' | rg '^\+.*\buseEffect\b'
```

No added `useEffect` matched.

The only added platform UI import is the required direct
`@platform/ui/JsonPreview` subpath. No production module, file rename,
migration exception, architecture/performance baseline, or additional
`@platform/ui` barrel importer was added.

`src/features/gallery/core/types.ts`, which owns
`GeneratedImageContract`, is byte-for-byte identical to reviewed head
`35644ab5be4bde2beb91cb9c52f6b92f7f2a3a65`; both SHA-256 values are:

```text
79398f2e0e0dd4c72409612531ba990cc0b6b5df64ee8e13c47a93f1b1190979
```

`git diff --check` also passed.
