# Task 10 Report: Boards, mixed upload, and i18n completion

## Status

Complete.

Board rows now count images and videos together while preserving the separate
asset count, expose a localized image/video breakdown, and mark static video
covers with a decorative Play glyph. Board deletion confirmation and outcome
messages include both media kinds, and the existing image-only archive wording
uses the board's exact video count through localized plural forms.

Gallery upload now classifies PNG, JPEG/JPG, WebP, and MP4 files by MIME first
with the established extension fallback. Images retain concurrent upload
behavior and the user category; videos upload one at a time as durable general
media. Date boards reject before transport, synthetic board IDs never reach
either upload route, partial failures do not stop later files, and one summary
reports the complete settled outcome. The newest successful upload visible in
the active board/view becomes the canonical selection.

The compact gallery status and remaining mixed-media gallery/Preview labels
are catalog-backed. Gallery/video `defaultValue` fallbacks were removed now
that their English keys exist.

## TDD evidence

All frontend commands ran from `invokeai/frontend/webv2`.

### Upload classifier and transports

The first backend RED was:

```sh
pnpm test src/features/gallery/data/backend.test.ts
```

```text
Test Files 1 failed (1)
Tests 19 failed | 40 passed (59)
```

It proved the named classifier and video transport were absent and that the
image upload still leaked the synthetic `all` board ID. After implementation,
the focused file passed all 59 tests.

### Board projection and presentation

Adding `videoCount` to the board-count contract first failed one of 24 state
projection tests. The focused file passed after `getBoardCounts()` returned
all three counts.

The new board browser contract initially failed both cases because the
mixed-total/tooltip surface and exported existing cover seam were absent. It
then passed 2/2 with:

- video-only `5 | 2` rendering;
- tabular numerals;
- localized image/video tooltip content;
- a decorative Play glyph over the static thumbnail;
- no board-list `<video>`.

The board-menu RED failed 2/2 against the image-only confirmation and
hardcoded archive label. It passed 2/2 after image/video/asset confirmation
counts, media-aware deletion wording, and localized omission text landed.

### Settled mixed upload and delete outcomes

The first consolidated action RED was:

```text
Test Files 1 failed (1)
Tests 6 failed | 2 passed (8)
```

Those failures covered authoritative board-delete notifications, date-board
pre-transport rejection, concurrent images plus sequential videos, continuation
after per-file failure, one partial mixed summary, unsupported files, and
newest visible selection. The file passed 8/8 after implementation.

A follow-up account-lifetime regression then failed because the second video
was scheduled after the first request aborted:

```text
Test Files 1 failed (1)
Tests 1 failed | 8 passed (9)
```

The sequential worker now checks the captured account signal before every
video and escalates account abort instead of treating it as a file-local
failure. The file then passed 9/9.

### Status and catalog

The compact status browser test first failed because `GalleryStatusChip` did
not exist; it passed after the component rendered
`widgets.gallery.statusChip`.

The English catalog test first failed on the missing status key. It now
resolves and interpolates status, image/video/asset counts, board omissions
and outcomes, mixed upload/split, playback/retry, frame copy, and video Details
keys. It also exercises plural-safe composed media counts.

## Implementation

### Boards

- `getBoardCounts()` returns image, video, and asset counts.
- Board badges render `(imageCount + videoCount) | assetCount` with tabular
  numerals.
- The badge trigger and tooltip share the localized image/video breakdown.
- A video cover remains a static poster and gains only a decorative Play
  overlay.
- Delete confirmation shows images, videos, and assets; completed deletion
  messages use the backend's authoritative confirmed/failed arrays.

### Upload

- `classifyGalleryUpload()` recognizes only PNG, JPEG/JPG, WebP, and MP4.
  Supported MIME wins; otherwise the established filename-extension fallback
  handles empty, generic, or misreported MIME.
- Both image and video transports omit `none`, `all`, and date-board IDs.
- Image upload stays `image_category=user`, preserves current concurrency, and
  remains strictly `GalleryImage`.
- Video upload is `video_category=general`, `is_intermediate=false`, maps to
  `GalleryVideoItem`, and runs sequentially.
- Images and the sequential video lane run independently. Per-file failures
  settle, unsupported files contribute to the failed total, and one localized
  notification summarizes the batch.
- Account expiry stops the video lane before another expensive request.
- Selection is canonical and restricted to successful media visible in the
  captured active board/view.

### i18n

- Replaced the hardcoded compact `Gallery: {total}` chip.
- Added pluralized image/video/asset counts, board archive/deletion outcomes,
  mixed upload results/split, and partial item-mutation/download messages.
- Generalized upload/search/drop/window copy from images to gallery items or
  media where appropriate.
- Removed gallery/video `defaultValue` fallbacks from mixed item actions and
  Preview video labels.

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
pnpm exec vitest run --config vitest.config.mts --reporter=dot
```

```text
Test Files 376 passed (376)
Tests 5005 passed (5005)
```

```sh
pnpm test:browser
```

The first full run hit the repository's known timing-sensitive
`src/platform/ui/Tabs.browser.test.tsx` timeout. The focused Tabs file passed
1/1, and the required complete rerun passed:

```text
Test Files 69 passed (69)
Tests 345 passed (345)
```

The browser suite retained existing non-failing React `act(...)`,
missing-i18n-instance, and intentional error-boundary diagnostics.

```sh
pnpm test:fixtures
```

```text
Tests 4 passed (4)
```

## Source-policy checks

- `GeneratedImageContract` and `core/types.ts` are unchanged.
- `GalleryImage` and the image upload result remain strict image-only
  contracts.
- No production module or existing file rename was added.
- No `useEffect`, migration exception, mock fixture, accessibility exception,
  or performance baseline was added.
- The only production `<video>` remains in `PreviewFrame.tsx`.
- No new file imports the `@platform/ui` barrel; the existing board-select
  importer only adds `Tooltip`.
- Scoped gallery/Preview/image-action translation calls contain no
  `defaultValue` fallback.
- `git diff --check` passed.

## Independent review

The independent read-only review approved the uncommitted Task 10 diff with no
Critical, Important, or Minor findings. It explicitly checked the board
outcomes, classifier fallback rules, image/video concurrency, date/synthetic
board handling, settled summary, visible-newest selection, i18n completion,
tests, and global architecture invariants.

## Task 11 dependencies

Task 11 still owns additive mock-backend videos and media assets/routes,
Range/HEAD fixture behavior, mixed mock ordering/filtering, and the dedicated
video accessibility journey with its narrowly scoped caption-rule exception.
No Task 11 or Task 12 baseline work was included here.
