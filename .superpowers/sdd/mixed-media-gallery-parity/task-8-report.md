# Task 8 Report: Native mixed-media Preview and protected video recovery

## Status

Complete.

Preview now consumes canonical `GalleryItem` contracts end to end. Images keep
their existing streaming, loupe, compare, and main-frame drag behavior, while
videos render through the only production `<video>` element in
`PreviewFrame.tsx`. Mixed navigation, filmstrip identity, header/common
actions, footer status, context-menu targets, and deletion context all use
qualified `GalleryItemKey`/`GalleryItemRef` identity.

Protected videos get one account-scoped cookie refresh and automatic
`video.load()` retry. Stale account/item completions are ignored, a second
failure shows a localized poster-backed Retry state, and manual Retry starts a
fresh one-refresh budget without synchronization effects.

## TDD evidence

All frontend commands ran from `invokeai/frontend/webv2`.

### Identity single-flight

The first focused RED was:

```sh
pnpm test src/features/identity/session.test.ts
```

```text
Test Files 1 failed (1)
Tests 3 failed | 15 passed (18)
```

The failures covered same-epoch single-flight, new-epoch isolation/stale
completion, and failure-as-`false`. After exporting
`refreshProtectedMediaCookie()`, the focused file passed all 18 tests.

### Qualified mixed navigation

Converting the pure navigation tests to canonical same-name image/video items
produced:

```text
Test Files 1 failed (1)
Tests 13 failed | 3 passed (16)
```

After the sequence, cursor, and target model moved to `GalleryItemKey`, all 16
tests passed.

The first full Preview integration RED was:

```text
Test Files 1 failed (1)
Tests 9 failed | 1 passed (10)
```

It covered canonical selection, same-name image/video order, native video
rendering, video comparison suppression, and image-only neighbor prefetch.

### Mixed filmstrip, footer, and actions

The new mixed chrome browser file initially failed all four tests. It passed
after:

- filmstrip keys and DnD IDs became source-qualified;
- static image/video posters selected canonical items;
- footer status added localized duration/fps and tabular dimensions;
- common star/download remained available for video while compare/copy stayed
  image-only.

The initial native frame contract also failed three tests across the frame and
drag files before the source union and keyed video arm were introduced.

### Protected video retry

The tightened retry RED was:

```sh
pnpm test:browser src/workbench/widgets/preview/PreviewFrameVideo.browser.test.tsx
```

```text
Test Files 1 failed (1)
Tests 6 failed | 2 passed (8)
```

Those failures required the first-error refresh/load, account and selected-item
stale guards, poster-backed terminal state, manual fresh budget, and keyed
per-item reset. The focused file then passed all 8 tests.

The combined Task 8 browser set passed:

```text
Test Files 5 passed (5)
Tests 37 passed (37)
```

The focused unit set passed:

```text
Test Files 3 passed (3)
Tests 38 passed (38)
```

Final review caught that Preview still supplied both Task 7's canonical
deletion-successor context and the legacy image callback. The regression test
failed 1 of 11 tests, then passed all 11 after removing the competing callback.

## Implementation

### Mixed Preview surface

- `PreviewWidgetView` derives, merges, navigates, and selects canonical mixed
  items in backend order.
- Qualified keys preserve same-name image/video independence through cursor,
  page-boundary, filmstrip, header, context-menu, and action contexts.
- Selecting a video clears comparison in the canonical reducer; Preview also
  gates compare rendering, compare drops, swap, loupe, and zoom commands to
  images.
- Neighbor prefetch creates `Image` requests only for image neighbors and
  never assigns a full video URL.
- The common Task 7 deletion context supplies ordered mixed refs once; the
  obsolete image-only successor callback is no longer registered.

### Native video arm

- `PreviewFrame` owns a local `PreviewMediaSource` union and delegates video to
  a child keyed by `itemKey`.
- The keyed child contains the sole production `<video>` with `controls`,
  `playsInline`, `preload="metadata"`, `poster`, and protected full `src`.
- The video arm registers no draggable, DnD listeners, compare drop zone,
  loupe/pan/wheel handlers, or `touchAction: none`.
- Video failure state retains the poster, localized failure text, and a manual
  Retry button.

### Protected-cookie recovery

- `refreshProtectedMediaCookie()` is account-epoch scoped and shares one
  in-flight transport promise per epoch.
- A new epoch never reuses the prior promise, and transport failure/stale
  completion resolves `false` rather than escaping into a media event.
- `PreviewVideo` captures the account epoch, element, and selected key before
  refresh; all three must still be current before calling `load()`.
- Refs and keyed component lifetime implement retry synchronization without a
  new `useEffect`.

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
Tests 4977 passed (4977)
```

```sh
pnpm test:browser
```

```text
Test Files 66 passed (66)
Tests 307 passed (307)
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

## Source-policy scans

```sh
rg -n '^[[:space:]]*<video' src --glob '!**/*.test.*' --glob '!**/*.spec.*'
```

Only `src/workbench/widgets/preview/PreviewFrame.tsx` matched.

```sh
git diff -U0 -- '*.ts' '*.tsx' | rg '^\+.*\buseEffect\b'
```

No added `useEffect` matched.

`git diff --check` also passed.

## Retained Task 9 seams

- Video Details/metadata, workflow/graph rendering, and copy-current-frame are
  intentionally absent.
- `actionImage` remains `null` for video, preserving the strict image-only
  metadata/recall boundary for Task 9.
- `PreviewVideo` remains local to `PreviewFrame.tsx`; Task 9 can extend its
  video-only UI without adding another media element or weakening image
  contracts.
