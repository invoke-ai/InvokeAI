# Task 12 Report: Final performance baselines and release verification

## Status

Complete. The mixed-media implementation has recorded architecture and browser
performance baselines, all release gates pass, and the baseline diff received
an independent read-only approval with no findings.

The first pre-update measurement correctly stopped on zero-headroom growth.
Systematic tracing isolated the cause to one new standalone first-party chunk.
A targeted editor-only chunk consolidation removed that request and reduced the
guarded entry sizes before either baseline was recorded.

## Pre-update measurement and stop condition

The clean starting head was:

```text
f9fb07fef639a2542a58fd76fbdb8766bdf71383
```

`pnpm test:performance:contracts` passed all 9 tests and
`pnpm exec vite build` built 5,116 modules. The required non-update checker then
failed with:

| Route | Metric | Baseline | Measured | Delta |
| --- | --- | ---: | ---: | ---: |
| launchpad | owned raw bytes | 57,925 | 57,966 | +41 |
| editor | owned raw bytes | 110,133 | 110,272 | +139 |
| editor | requests | 74 | 75 | +1 |
| editor | script requests | 58 | 59 | +1 |

The editor source-owner set added only
`source:src/features/gallery/core/items.ts`. The importer audit still measured
`@platform/ui` at exactly 156.

No baseline update command was run at this point.

## Root cause and consolidation

Rollup emitted `core/items.ts` as the sole owner of a new 1,316-byte
`items-*.js` common chunk. It had runtime consumers across four editor-initial
chunks and five independent lazy chunks. The new chunk path also appeared in
Vite's dependency/preload tables in both route entry files, explaining the
small owned-entry growth separately from the leaf payload.

Nearby pure gallery helpers remain coalesced because their runtime consumers
already converge in `galleryStateView` or `workbenchState`.

After controller approval, `vite.config.mts` received one targeted rule that
places:

- `features/gallery/core/items.ts`
- `features/gallery/ui/galleryStateView.ts`

in an editor-only `gallery-state` chunk. It does not use the launchpad-eager
`shared` chunk and does not couple gallery core to DnD/vendor code.

The single-variable rebuild emitted one 11,026-byte `gallery-state` chunk and
removed the standalone item chunk. The non-update checker then failed only for
the expected unrecorded editor source owner:

| Route | Metric | Baseline | Measured | Delta |
| --- | --- | ---: | ---: | ---: |
| launchpad | owned raw bytes | 57,925 | 57,857 | -68 |
| launchpad | requests / scripts | 31 / 15 | 31 / 15 | 0 / 0 |
| editor | owned raw bytes | 110,133 | 108,775 | -1,358 |
| editor | requests / scripts | 74 / 58 | 72 / 56 | -2 / -2 |

Launchpad source ownership remained unchanged. Editor source ownership added
only `core/items.ts`.

## Exactly-once baseline recording

After the zero-headroom audit passed, the following sequence ran once, in the
specified order:

```sh
pnpm test:performance:contracts
pnpm exec vite build
node scripts/check-architecture-performance.mjs --update-baseline
pnpm test:performance:browser:update-baseline
```

The contracts passed 9/9, both builds transformed 5,116 modules, and both
baseline writers exited successfully. Neither update command was repeated.

The architecture writer's JSON serialization expanded one single-element
structural array, which the repository formatter rejected. With controller
approval, only `performance/architecture-baseline.json` was passed through
`oxfmt`. Canonical JSON hashes before and after were identical:

```text
c66003e91cfee4b6ff2f57484ac71fdea97e78a9c793bbe638eeb70bf39ac13c
```

No baseline value was hand-edited.

## Baseline diff audit

The intended production/baseline files are:

- `invokeai/frontend/webv2/vite.config.mts`
- `invokeai/frontend/webv2/performance/architecture-baseline.json`
- `invokeai/frontend/webv2/performance/browser-baseline.json`

Architecture changes:

- launchpad source owners: no additions or removals;
- editor source owners: added only
  `source:src/features/gallery/core/items.ts`;
- launchpad requests/scripts remained 31/15;
- editor requests/scripts decreased from 74/58 to 72/56;
- launchpad/editor owned bytes decreased to 57,857/108,775;
- editor initial raw bytes increased by 11,893, gzip by 2,670, and Brotli by
  2,648 under the existing generated 1% policy;
- launchpad initial raw bytes increased by 815 while gzip and Brotli decreased;
- `developmentInvalidation` is semantically unchanged;
- the `@platform/ui` cap and measured importer count remain exactly 156;
- structural policy is semantically unchanged.

Browser changes:

- route identities, schema, sampling, and timing policy remain unchanged;
- launchpad request/script counts remain 19/15;
- editor static request/script counts decrease by two;
- activated request/script counts do not increase;
- static script-owner sets add only `core/items.ts` to the editor minimal and
  canvas sets; launchpad is unchanged and no owner is removed;
- recorded resource growth is payload under the existing generated 1% policy,
  not a zero-headroom request or owned-entry increase.

`performance/architecture-fixtures.json`, profile-count shapes, and
`migrationExceptions.ts` are unchanged. The latter remains `[]`.

## Independent baseline review

The read-only reviewer approved the config and generated baselines with no
Critical, Important, or Minor findings. It independently confirmed:

- only the expected config/baseline files changed before this report;
- architecture values exactly match `build-report.json`;
- browser values and limits exactly derive from `browser-report.json`;
- launchpad ownership is unchanged;
- editor adds only `core/items.ts`;
- owned bytes and request counts do not grow;
- the 156-importer cap, structural policy, fixture schema, and migration
  exceptions are unchanged;
- no suspicious budget weakening or hand edit is present.

## Final frontend verification

All commands ran from `invokeai/frontend/webv2`.

```text
pnpm lint
  format: passed
  OXC: passed
  tsc: passed
  architecture: 3 files, 34 tests passed

pnpm test:all
  unit: 376 files, 5,005 tests passed
  browser: 70 files, 358 tests passed
  fixtures: 10 tests passed

pnpm test:performance:architecture
  contracts: 9 tests passed
  deterministic build/source-owner checks: passed
  all browser route/resource/timing policies: passed

pnpm test:accessibility
  production build: passed
  all 9 representative/keyboard/video journeys: passed

pnpm check:release
  complete aggregate lint, unit, browser, fixture, performance, and
  accessibility sequence: passed
```

No timing-sensitive Tabs failure occurred, so the focused-rerun protocol was
not needed. Browser tests still emit their existing non-failing React `act`,
intentional error-boundary, and missing-test-i18n-instance console diagnostics.
Vite still reports its existing large-chunk and Babel plugin-timing warnings.

## Backend verification

From repository root:

```text
uv run pytest -q \
  tests/app/services/gallery/test_gallery_default.py \
  tests/app/routers/test_gallery.py

23 passed, 4 warnings
```

The warnings are environmental/upstream: the installed PyTorch build does not
target the host RTX 5090 CUDA capability, two protobuf container types use a
Python API deprecated for 3.14, and passlib imports `crypt`, which is deprecated
for Python 3.13. They do not affect the focused test result.

## Invariant scans

- The only production `<video>` element is in `PreviewFrame.tsx`.
- No production `useEffect` line was added since branch start.
- `GeneratedImageContract` is byte-for-byte unchanged.
- No existing production file was renamed.
- `core/items.ts` is the only new production source file.
- `migrationExceptions.ts` remains empty.
- Exactly one production accessibility call disables `video-caption`; the
  other occurrence is the fixture test enforcing that count.
- The representative performance fixture keeps image index 0 as the default
  selection.
- `/gallery/items/names` remains fetched on Shift-click/action demand, not
  initial gallery render.
- Details query components mount only while Details is open.
- `git diff --check` passes.
