# Task 4 report — mixed gallery queries, hydration, overlays, and cache patches

## Delivered

- Renamed the final gallery read model to item semantics:
  - `galleryItemsInfiniteOptions`
  - `flattenGalleryItemsData`
  - `GalleryItemsFilter`
  - `GalleryItemsListQueryKey`
  - `GalleryItemsWindow`
  - `gallery/items/list` cache keys
- Updated the positional key guard to accept only the item-list layout and deduplicated flattened pages by `GalleryItemKey`, preserving same-name image/video pairs.
- Added lazy `galleryItemNamesOptions` for normal boards, date boards, and created ranges. Constructing the options performs no request; consumers must explicitly call `fetchQuery`.
- Replaced webv2 date-board `/image_names` use with `/item_names`.
- Kept the refcounted `dateBoardNamesConsumers` request sharing for date-board infinite and anchor consumers.
- Added mixed date hydration:
  - images are fetched in one `images_by_names` request;
  - videos are fetched through individual DTO requests with at most six active requests;
  - hydrated output preserves reference order;
  - missing 404 video refs are omitted;
  - abort, authentication, network, and every other non-404 error propagate.
- Converted recent queue images to `GalleryImageItem` before mixed overlay merge.
- Added exact server-order comparison: starred first, created time, kind, then name, with order direction applied to every non-starred term.
- Merged and deduplicated overlays by qualified item key while retaining the existing 60-recent-item and rendered-window caps.
- Added confirmed-result cache patches. `patchGalleryItemCaches` derives qualified keys only from `GalleryItemMutationResult.succeeded`; failed refs are never patched.
- Generalized invalidation to item pages and item-name lists, retained one-time board invalidation, and preserved same-tick coalescing and account scoping.
- Removed the temporary backend `listGalleryImages = listPaletteImages` alias.
- Kept palette search explicitly image-only. Its date-board path now derives image refs from `/item_names`, so no webv2 gallery path calls `/image_names`.

## TDD evidence

### RED

Focused command from `invokeai/frontend/webv2`:

```sh
pnpm test src/features/gallery/data/task4Queries.test.ts src/features/gallery/data/task4QueryCache.test.ts src/features/gallery/data/task4Backend.test.ts src/features/gallery/ui/useGalleryData.test.ts
```

Initial result:

```text
Test Files 4 failed (4)
Tests 7 failed | 9 passed (16)
```

The query, cache, and overlay failures were causal missing-API failures:

```text
canonicalizeGalleryItemsFilter is not a function
flattenGalleryItemsData is not a function
galleryItemNamesOptions is not a function
galleryItemsInfiniteOptions is not a function
mergeGalleryItemWindow is not a function
```

After correcting a test-only hoisted mock, the backend-only RED command produced:

```text
Test Files 1 failed (1)
Tests 7 failed (7)
```

All seven failures were the expected missing seams:

```text
listGalleryDateBoardItemNames is not a function
listGalleryItemNames is not a function
hydrateGalleryDateBoardItemPage is not a function
```

The focused files were renamed after GREEN to their durable behavior names:

- `dateBoardItemHydration.test.ts`
- `mixedQueries.test.ts`
- `itemQueryCache.test.ts`

### GREEN

Final focused command:

```sh
pnpm test src/features/gallery/data/dateBoardItemHydration.test.ts src/features/gallery/data/mixedQueries.test.ts src/features/gallery/data/itemQueryCache.test.ts src/features/gallery/ui/useGalleryData.test.ts
```

Result:

```text
Test Files 4 passed (4)
Tests 23 passed (23)
```

The focused coverage includes the item key segment and positional guard, same-name image/video dedupe, lazy item-name options with created ranges, mixed page shape, date item-name URL and counts, reference order, one image batch, six-wide video concurrency, 404 omission, abort/auth/network propagation, server-exact overlay ordering, success-only qualified cache patches, and coalesced item/name/board invalidation.

## Compatibility obligations

Task 5/7 must remove the following narrowly scoped compatibility surfaces after Gallery state/projection, image actions, and Preview consume `GalleryItem`:

1. `gallery/legacy-images/list` key domain and `galleryKeys.legacy*`.
2. `galleryImagesInfiniteOptions`, which calls `listPaletteImages` directly and returns only `GalleryImagesPage` under the isolated legacy key. It does not alter or share the mixed item cache.
3. `flattenGalleryImagesData`.
4. Legacy image query discovery/key-guard helpers and the `GalleryImages*`/`CanonicalGalleryImagesFilter` aliases.
5. `patchGalleryImageCaches` and `GalleryImageCachePatch`; image actions must switch to `patchGalleryItemCaches` with confirmed `GalleryItemMutationResult`.
6. `invalidateGalleryImages`; legacy image consumers must switch to `invalidateGalleryItems`/`invalidateGallery`.
7. `mergeGalleryImageWindow`; Gallery projection must switch to `mergeGalleryItemWindow`.
8. The compatibility exports for those symbols in `features/gallery/queries.ts`.

Every production compatibility seam carries an inline `TODO(Task 5/7)` or `TODO(Task 5)` marker. The backend `listGalleryImages` alias was not retained.

## Files

### Production

- `invokeai/frontend/webv2/src/features/gallery/data/backend.ts`
- `invokeai/frontend/webv2/src/features/gallery/data/queries.ts`
- `invokeai/frontend/webv2/src/features/gallery/data/queryCache.ts`
- `invokeai/frontend/webv2/src/features/gallery/queries.ts`
- `invokeai/frontend/webv2/src/features/gallery/ui/useGalleryData.ts`

### Tests

- `invokeai/frontend/webv2/src/features/gallery/data/backend.test.ts`
- `invokeai/frontend/webv2/src/features/gallery/data/queries.test.ts`
- `invokeai/frontend/webv2/src/features/gallery/data/queryCache.test.ts`
- `invokeai/frontend/webv2/src/features/gallery/data/dateBoardItemHydration.test.ts`
- `invokeai/frontend/webv2/src/features/gallery/data/mixedQueries.test.ts`
- `invokeai/frontend/webv2/src/features/gallery/data/itemQueryCache.test.ts`
- `invokeai/frontend/webv2/src/features/gallery/ui/useGalleryData.test.ts`

## Verification

From `invokeai/frontend/webv2`:

```sh
pnpm test
```

```text
Test Files 372 passed (372)
Tests 4928 passed (4928)
```

```sh
pnpm run lint
```

```text
format: all matched files correctly formatted
oxlint: zero warnings/errors
tsc --noEmit: passed
architecture tests: 3 files passed, 34 tests passed
```

```sh
pnpm test src/features/gallery/data/backend.test.ts src/features/gallery/data/queries.test.ts src/features/gallery/data/queryCache.test.ts src/features/gallery/ui/useGalleryData.test.ts src/features/gallery/data/dateBoardItemHydration.test.ts src/features/gallery/data/mixedQueries.test.ts src/features/gallery/data/itemQueryCache.test.ts
```

```text
Test Files 7 passed (7)
Tests 72 passed (72)
```

## Self-review and concerns

- Confirmed the mixed query always returns `GalleryItemsPage`; legacy image pages live under a separate key and call the palette list directly.
- Confirmed no production webv2 gallery/workbench source contains `listGalleryImages` or `/image_names`.
- Confirmed item-name options are not subscribed or fetched eagerly.
- Confirmed cancellation and identity checks surround both names and page hydration.
- Confirmed no `useEffect` was added.
- No unresolved implementation concern. The listed Task 5/7 removals are deliberate branch-compatibility debt, not mixed-contract ambiguity.
