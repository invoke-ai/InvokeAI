# Mixed-media gallery parity: final review fix report

Base reviewed: `0b7acabad952597073a2ea1b3298e8848351aba3`

## Findings resolved

### Upload completion uses the latest gallery context

- `GalleryWidgetView` now exposes a render-assigned, stable live-read port for
  the current board and Images/Assets view. No `useEffect` was added.
- An upload still captures its transport target and target-board name when it
  starts.
- After all image/video requests settle and the account scope is revalidated,
  selection visibility is evaluated against the latest board and view.
- Switching either board or view while an upload is pending no longer selects
  media that is invisible in the new context.
- Existing concurrent image uploads, sequential video uploads, settled
  outcomes, newest-item comparator, one invalidation, and one summary
  notification are preserved.

Strict TDD evidence:

- RED: the deferred view-switch browser test received one `selectItem` call for
  `photo.png` when it expected none.
- GREEN: the deferred board-switch and view-switch cases both pass, including
  assertions that transport still targets the launch board and completion
  emits one invalidation and one summary.

### Persisted selection-key compatibility is centralized

- Added `getPersistedSelectedGalleryItemKeys()` to Gallery core and exported it
  through the existing Gallery contract.
- The reader preserves the established precedence:
  `selectedImageNames` → `selectedImageName` → canonical/legacy
  `selectedImage`.
- Bare image names and qualified mixed-media keys are canonicalized through the
  existing parser.
- Both `galleryStateView` and `workbenchState` now use the shared reader.

Strict TDD evidence:

- RED: all three new cases received `undefined` before the shared reader
  existed.
- GREEN: qualified/bare arrays, legacy singular selection, and canonical
  selected-video fallback pass.

### Unused temporary adapters are removed

- Removed only `patchImages`, `removeImages`, `setMultiSelection`, and
  `toggleImageSelection`.
- Removed matching `GalleryCommandsPort` members and test scaffolding.
- Retained the still-used image-only `selectImage` and `setCompareImage`
  compatibility seams.

Strict TDD evidence:

- Removing `setMultiSelection` from the adapter fixture first produced
  `TS2741`: the property was still required by `GalleryCommandsPort`; removing
  the port/store adapter restored the typecheck.
- The same RED/GREEN cycle was repeated for `toggleImageSelection`.
- Repository production search confirms none of the four adapter definitions
  remain.

## Verification

- Focused core/state tests: 3 files, 193 tests passed.
- Focused Gallery browser tests: 2 files, 28 tests passed.
- `pnpm lint`: passed, including 34 architecture tests.
- `pnpm test:all`: passed:
  - 376 unit files / 5,008 tests
  - 70 browser files / 360 tests
  - 10 mock-fixture tests
- `pnpm test:performance:architecture`: passed without updating baselines.
- `pnpm check:release`: passed end-to-end on the final rerun, including all
  nine accessibility journeys.

The first `check:release` attempt reached its final accessibility command and
reported a canvas-only `Raster Layers (64)` contrast violation. No source was
changed. An immediate unchanged `pnpm test:accessibility:browser` reproduction
passed all nine journeys, and the complete unchanged `pnpm check:release`
rerun also passed. This was treated as a transient journey-state failure, not
masked or baseline-recorded.

## Independent review

An independent read-only review of
`0b7acabad952597073a2ea1b3298e8848351aba3..working tree` returned:

- Standards: 0 Critical, 0 Important, 0 Minor findings.
- Specification: 0 Critical, 0 Important, 0 Minor findings.

The reviewer specifically confirmed the launch/completion upload-context
split, the exact persisted-reader precedence, removal of only the four unused
adapters, retention of `selectImage`/`setCompareImage`, Gallery-core purity,
and dependency direction.

## Invariants

- `GeneratedImageContract` is unchanged.
- No production module or existing file was added/renamed for this fix.
- No `useEffect` was added.
- No new `@platform/ui` barrel importer was added.
- No performance baseline or importer cap was updated.
