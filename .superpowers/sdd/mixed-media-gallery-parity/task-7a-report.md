# Task 7A Report: Mixed-media organization transport and domain results

## Status

Complete.

Task 7A adds the mixed-media `galleryItemOrganization` transport/domain API
through the existing gallery entry. Delete, star/unstar, and board movement
partition ordered qualified refs by media kind, trust only backend-confirmed
success names, and return confirmed successes, unconfirmed failures, and
deterministic affected board IDs.

This subtask does not patch caches or Workbench state, invalidate queries,
notify, add action providers, change UI, or implement downloads. Those
operation-level effects remain Task 7B responsibilities after it consumes the
confirmed result.

## TDD evidence

### Inherited RED

The original Task 7A implementer captured the focused RED before the production
edits and handed off this evidence with the uncommitted implementation:

```sh
pnpm test src/features/gallery/data/organization.test.ts
```

```text
Test Files 1 failed (1)
Tests 11 failed (11)
```

The failures covered the then-missing mixed organization entry, video bulk
transports, per-video board movement, confirmed-only result mapping, and abort
and concurrency behavior.

### GREEN

After independently auditing the inherited diff, the resumed focused run was:

```sh
pnpm test \
  src/features/gallery/core/items.test.ts \
  src/features/gallery/data/backend.test.ts \
  src/features/gallery/data/organization.test.ts \
  src/features/gallery/data/queryCache.test.ts \
  src/features/gallery/data/itemQueryCache.test.ts
```

```text
Test Files 5 passed (5)
Tests 71 passed (71)
```

Aggregate lint then identified only an OXC structural warning in the bounded
worker loop (`no-unmodified-loop-condition`). The loop was rewritten without
changing its early-abort or scheduling behavior. The same focused command
passed again with 5 files and 71 tests before the full gates were repeated.

## Implementation

### Strict organization transports

- Image move, remove, star/unstar, and delete transports now map the required
  authoritative success arrays and `affected_boards` array from strict runtime
  DTOs.
- Existing image-only gallery organization methods remain available and keep
  their image-name-only return types.
- Video delete, star, and unstar use one backend bulk request with the exact
  `video_names` body.
- Video board movement uses the existing single-video
  `POST /api/v1/videos/board` and `DELETE /api/v1/videos/board` contracts.
- Every request forwards the caller's abort signal and checks it before and
  after transport resolution so an aborted response cannot become a confirmed
  success.

### Confirmed mixed-media results

- Input refs are deduplicated by `GalleryItemKey`, retaining first-request
  order.
- Image and video names retain their per-kind request order and execute as
  independent `Promise.allSettled` partitions.
- Backend success names are intersected with the requested names; malformed or
  unexpected names confirm nothing.
- `succeeded` and `failed` are reconstructed in original qualified-ref order,
  so an image and video with the same name remain independent.
- A rejected bulk partition confirms none of its requested refs while the
  independently fulfilled media partition is retained.
- Affected source/destination board IDs are included only for outcomes with a
  requested confirmed success, deduplicated, and sorted deterministically.
  The `none` board is retained.
- The shared mutation result can carry affected board IDs without weakening
  its required `succeeded` and `failed` arrays. The public Task 7A API returns
  affected board IDs as a required array.

### Bounded video movement

- At most four single-video move requests are active at once.
- A rejected or malformed per-video request remains unconfirmed without
  converting it into success.
- Abort stops workers before scheduling another request. Aborted and unstarted
  refs remain in `failed`.
- Per-request responses are reduced in input order; transport completion order
  cannot reorder the domain result.

## Verification

All frontend commands ran from `invokeai/frontend/webv2`.

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
Tests 4962 passed (4962)
```

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

The final full unit, fixture, and aggregate lint commands were run after the
worker-loop lint correction.

Browser tests were intentionally not run for this transport/domain-only
subtask; Task 7A requires proportionate non-UI gates and expressly excludes UI
and action behavior.

## Self-review

- Confirmed `GeneratedImageContract` and `core/types.ts` are unchanged.
- Confirmed image-only APIs remain strictly image-name/image typed.
- Confirmed the diff adds no production module or production file rename.
- Confirmed no UI, action provider, context menu, hotkey, download, archive, or
  board UI file changed.
- Confirmed no cache/Workbench patch, query invalidation, or notification occurs
  in the backend or domain API.
- Confirmed no `useEffect`, `@platform/ui` import, migration exception, or
  performance baseline change was added.
- Confirmed same-name image/video identity is preserved end to end.
- Confirmed malformed/unexpected response names and rejected partitions cannot
  become confirmed successes.
- Confirmed video board movement has one request per deduplicated video, at
  most four in flight, and no per-video invalidation or duplicate operation
  side effect.
- Confirmed `git diff --check` is clean.

Task 7B owns consuming the confirmed arrays to patch/prune state, perform one
operation-level invalidation, and issue one user notification. The subsequent
independent review and Task 7A fix round are recorded below.

## Independent review fix round

An independent review of `35bf5e98ae` found two transport-boundary gaps:

1. Bulk video delete/star/unstar accepted a success array without validating
   the backend-required `failed_videos` array.
2. The bounded video-board worker treated authentication, identity-expiry, and
   abort failures like isolated item failures, allowing later index claims and
   potentially retaining video successes from a stale identity lifetime.

### Fix-round RED

The focused organization run after adding the malformed DTO and fatal
concurrency regressions was:

```sh
pnpm test src/features/gallery/data/organization.test.ts
```

```text
Test Files 1 failed (1)
Tests 9 failed | 13 passed (22)
```

Six failures showed that missing or wrong-type `failed_videos` still confirmed
the requested video across star, unstar, and delete. Two failures showed HTTP
401 and `HttpRequestIdentityExpiredError` were swallowed: all remaining video
requests were scheduled and their successes retained. The ninth showed that
an omitted caller signal did not bind movement to the captured account
lifetime, so account rotation scheduled eight requests instead of stopping at
the four already in flight.

### Fix-round implementation

- Added a strict bulk-video result mapper that requires arrays containing only
  non-empty strings for the operation success field, `affected_boards`, and
  `failed_videos`.
- Captured the account scope at the start of video board movement and composed
  its lifetime signal with an optional caller signal.
- Added one shared fatal sentinel checked synchronously before every worker
  index claim.
- Classified caller/account abort, `AccountScopeExpiredError`,
  `HttpRequestIdentityExpiredError`, and HTTP 401 as partition-fatal.
- A fatal outcome rejects the video partition after the current workers settle,
  so the outer `Promise.allSettled` orchestration confirms no video refs while
  retaining an independently completed image partition.
- Kept ordinary HTTP 403/500 and malformed per-item move responses isolated to
  their requested video; later video requests still run and may confirm.

### Fix-round GREEN and full verification

All commands ran from `invokeai/frontend/webv2` on the final formatted tree.

```sh
pnpm test src/features/gallery/data/organization.test.ts
```

```text
Test Files 1 passed (1)
Tests 22 passed (22)
```

```sh
pnpm test \
  src/features/gallery/core/items.test.ts \
  src/features/gallery/data/backend.test.ts \
  src/features/gallery/data/organization.test.ts \
  src/features/gallery/data/queryCache.test.ts \
  src/features/gallery/data/itemQueryCache.test.ts
```

```text
Test Files 5 passed (5)
Tests 82 passed (82)
```

```sh
pnpm test
```

```text
Test Files 376 passed (376)
Tests 4973 passed (4973)
```

```sh
pnpm test:fixtures
```

```text
Tests 4 passed (4)
```

```sh
pnpm lint:tsc
```

```text
tsc --noEmit: passed
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

The first aggregate lint attempt stopped only on formatting in the expanded
organization test. The repository formatter was applied to that test, then
the focused suite, full unit suite, fixtures, and aggregate lint were all run
again with the final counts above.

### Fix-round self-review

- Confirmed malformed bulk video results cannot contribute successes or
  affected board IDs.
- Confirmed fatal 401/identity/account/abort outcomes stop new claims and reject
  the complete video partition.
- Confirmed an already completed image partition remains independently
  confirmable through the existing settled orchestration.
- Confirmed ordinary 403/500 move failures remain item-local and do not stop
  later video work.
- Confirmed every video board request is account-lifetime-scoped even when the
  caller omits a signal.
- Confirmed Task 7A still performs no cache patch, invalidation, notification,
  UI/action, or download work.
- Confirmed no production module, `useEffect`, `@platform/ui` importer,
  migration exception, or baseline change was added.
