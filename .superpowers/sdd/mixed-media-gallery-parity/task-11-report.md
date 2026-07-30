# Task 11 Report: Additive mock videos and accessibility journey

## Status

Complete, with one pre-existing release-suite blocker recorded for Task 12.

The representative mock backend now provides deterministic mixed image/video
data for the gallery, date boards, board summaries and deletion, uploads,
mutations, Details, posters, and protected full media. The fixture-count
contract remains unchanged, and the default Images performance route still
orders an image first.

The dedicated representative-video accessibility journey passes. It is the
only call site that disables Axe's `video-caption` rule, with the required
generated-media explanation.

## Mock fixture and route coverage

- Added `state.videos` without replacing any of the 1,000 representative
  images or adding a video key to profile counts.
- Added owned general, board-cover, same-name collision, control-category,
  intermediate, foreign-owner, and prior-day video rows.
- Kept the newest default Images result as an image while retaining a real
  static video board cover.
- Added the production-consumed mixed gallery list and names routes, including
  backend defaults and starred/created/kind/name ordering in both directions.
- Added board, ownership, category, intermediate, search, inclusive date range,
  offset, limit, total, and date-board ref behavior.
- Added video DTO, metadata, workflow/graph, upload, star/unstar/delete, and
  single-video board add/remove routes.
- Added dynamic mixed board counts, covers, detach outcomes, and cascade-delete
  outcomes.
- Added poster and full-media responses. Full media supports bodyless `HEAD`,
  ordinary `200`, satisfiable `206`, and malformed/unsatisfiable `416` with
  exact `Accept-Ranges`, `Content-Length`, and `Content-Range` headers.

`getStateCounts`, `MOCK_BACKEND_PROFILE_COUNTS`, and
`performance/architecture-fixtures.json` are unchanged.

## Media assets

The repository had no suitably tiny H.264 fixture in the existing webv2 mock
asset area, so both assets were generated with FFmpeg n8.1.2:

```sh
ffmpeg -hide_banner -loglevel error \
  -f lavfi -i "testsrc2=size=64x64:rate=10:duration=1" \
  -an -c:v libx264 -preset veryslow -crf 28 -pix_fmt yuv420p \
  -profile:v baseline -level 3.0 -movflags +faststart \
  -fflags +bitexact -flags:v +bitexact -map_metadata -1 \
  -y scripts/mock-assets/fixture-video.mp4

ffmpeg -hide_banner -loglevel error \
  -i scripts/mock-assets/fixture-video.mp4 -frames:v 1 \
  -c:v libwebp -lossless 1 -map_metadata -1 \
  -y scripts/mock-assets/fixture-video.webp
```

Asset evidence:

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| `fixture-video.mp4` | 4,766 | `372ad163e36c7b0c104ff50e706554268149b437fdf26086e0af2aea6939f7f6` |
| `fixture-video.webp` | 5,954 | `a1d099ffa75a177a7cdd64335212257654f4a0eaf150c0ef7557064137034d4d` |

`ffprobe` reports a 1.0-second, 64×64, 10 fps H.264 Constrained Baseline
stream with `yuv420p` pixels. `file` recognizes the poster as lossless WebP.

## Accessibility journey

The new `workbench-video-preview-representative` journey:

- opens the Gallery layout and uses the renamed `Gallery items` list;
- keyboard-focuses the representative video cell;
- verifies its Play glyph is decorative;
- activates the cell with Enter without starting DnD;
- verifies the opened mixed Preview's native controls, `playsInline`, poster,
  non-draggable video behavior, and localized duration;
- runs Axe after the surface settles.

Only this exact Axe call passes:

```js
{ rules: { 'video-caption': { enabled: false } } }
```

Every existing surface and the keyboard journey retain the critical caption
rule.

## Demonstrated Task 1–10 corrections

Three production mismatches appeared only when the representative video journey
exercised the real production bundle. All three were reported to and authorized by
the controller before correction.

### Stable gallery virtualizer callbacks

`react-hook-tanstack-virtual` shallow-compares its external-store options.
Fresh inline `estimateSize` and `getScrollElement` functions produced a new
snapshot on every render, causing React's cached-snapshot warning and then
error 301 (`Too many re-renders`) once real gallery data arrived. The existing
browser mock hid the identity requirement.

The two callbacks are now memoized with `useCallback`. A focused browser test
first failed on callback identity, then passed, and the production
representative gallery rendered normally.

### Compact Preview contrast

Once the video opened, Axe exposed 3.56:1 contrast for the 10px position,
dimensions/duration, and Details text. A focused browser test first failed
against the `fg.muted` semantic foreground. Those three existing compact text
surfaces now use `fg.muted`; image and video footer cases both pass, and the
focused video Axe journey is clean.

### Keyboard item activation versus DnD

The workbench's real `KeyboardSensor` received the Enter/Space keydown bubbled
from the gallery item's native button. It activated dragging and prevented the
button click that opens Preview. Focused browser tests reproduced both keys
with the production sensor: both initially made zero selection calls.

The native item button now stops propagation only for Enter/Space keydown.
Native button click behavior remains intact, pointer DnD remains green, and
keyboard DnD remains available when its own parent handle is focused. Both
focused key cases pass, and the production accessibility journey now opens the
video with Enter.

No new production module, `<video>`, `useEffect`, migration exception,
performance baseline, or `@platform/ui` importer was added.

## TDD evidence

The first fixture run failed six new cases while all four legacy cases passed:
the video state was absent and the mixed/date/video/board routes returned
404s or image-only results. After implementation, all ten fixture tests pass.

Additional focused RED/GREEN cycles covered:

- default Images ordering remaining image-first;
- the backend's omitted-parameter `starred_first=true` behavior;
- backend-accurate board-cover starred ordering and the
  `starred_count=0` contract when `starred_first=false`;
- stable virtualizer callback identity (1 failed / 14 passed, then 15/15);
- Enter/Space gallery activation with a real keyboard sensor (2 failed / 15
  passed, then 17/17);
- readable compact Preview foreground (1 failed / 6 passed, then 7/7);
- the representative video journey and its sole caption exception.

## Verification

All frontend commands ran from `invokeai/frontend/webv2`.

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
Tests 5005 passed (5005)
```

```sh
pnpm test:browser
```

The first full run hit the repository's known timing-sensitive Tabs hover
timeout. The focused Tabs file passed 1/1, and the required complete rerun
passed:

```text
Test Files 69 passed (69)
Tests 349 passed (349)
```

```sh
pnpm test:fixtures
```

```text
Tests 10 passed (10)
```

```sh
INVOKEAI_ACCESSIBILITY_JOURNEY=workbench-video-preview-representative \
  node scripts/run-accessibility-journeys.mjs
```

```text
workbench-video-preview-representative: passed
```

Invariant scans also confirmed:

- the only production `<video>` remains in `PreviewFrame.tsx`;
- no production source file was added in Task 11;
- `GeneratedImageContract`, migration exceptions, architecture fixtures, and
  performance baselines are unchanged;
- exactly one `video-caption` exception exists;
- `git diff --check` passes;
- the checked-in MP4/WebP formats, dimensions, lengths, and hashes match the
  values above.

## Full accessibility suite and Task 12 dependency

`pnpm test:accessibility` builds successfully, then stops at the existing
`workbench-canvas-representative` surface before later surfaces run. Axe
reports:

- `nested-interactive` on the 64 sortable canvas layer rows because their
  role-button containers contain focusable descendants;
- `target-size` on 12×12 layer visibility buttons.

These are unrelated to mixed media and predate Task 11. The video-specific
journey passes independently with no violation. Task 12 must either fix or
explicitly disposition the Canvas violations before `check:release` can be
fully green; Task 11 does not weaken either rule.

## Independent review

The independent read-only review found no Critical issues. It reported two
Important findings and one Minor finding:

- mock board-cover selection omitted starred-first ordering;
- the video journey focused the item but activated it with a mouse click;
- item-name `starred_count` did not become zero when starred-first ordering was
  disabled.

All three were addressed test-first. Board covers now use the backend's exact
starred/created/kind/name precedence and the intended fixture cover is itself
starred. The journey now activates with Enter after focused Enter/Space
browser coverage fixed the DnD conflict. The Minor `starred_count` mismatch is
fixed and asserted for `starred_first=false`.

The scoped re-review approved the code corrections with no Critical or
Important findings. Its two report-only Minor findings—the stale count of
production mismatches and stale browser-test total—are corrected in this
report.
