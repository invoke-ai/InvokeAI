import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppStore } from 'app/store/storeHooks';
import { useCallback, useEffect, useState } from 'react';
import type { ListRange } from 'react-virtuoso';
import { queueApi, useGetQueueItemDTOsByItemIdsMutation } from 'services/api/endpoints/queue';
import { useThrottledCallback } from 'use-debounce';

interface UseRangeBasedQueueItemFetchingArgs {
  itemIds: number[];
  enabled: boolean;
}

interface UseRangeBasedQueueItemFetchingReturn {
  onRangeChanged: (range: ListRange) => void;
}

const getUncachedItemIds = (itemIds: number[], cachedItemIds: number[], ranges: ListRange[]): number[] => {
  const uncachedItemIdsSet = new Set<number>();
  const cachedItemIdsSet = new Set(cachedItemIds);

  for (const range of ranges) {
    for (let i = range.startIndex; i <= range.endIndex; i++) {
      const n = itemIds[i]!;
      if (n && !cachedItemIdsSet.has(n)) {
        uncachedItemIdsSet.add(n);
      }
    }
  }

  return Array.from(uncachedItemIdsSet);
};

/**
 * Hook for bulk fetching queue items based on the visible range from virtuoso.
 * Individual quite item components should use `useGetQueueItemQuery(item_id)` to get their specific DTO.
 * This hook ensures DTOs are bulk fetched and cached efficiently.
 */
export const useRangeBasedQueueItemFetching = ({
  itemIds,
  enabled,
}: UseRangeBasedQueueItemFetchingArgs): UseRangeBasedQueueItemFetchingReturn => {
  const store = useAppStore();
  const [getQueueItemDTOsByItemIds] = useGetQueueItemDTOsByItemIdsMutation();
  const [lastRange, setLastRange] = useState<ListRange | null>(null);
  const [pendingRanges, setPendingRanges] = useState<ListRange[]>(EMPTY_ARRAY);

  const fetchQueueItems = useCallback(
    (ranges: ListRange[], itemIds: number[]) => {
      if (!enabled) {
        return;
      }
      const cachedItemIds = queueApi.util.selectCachedArgsForQuery(store.getState(), 'getQueueItem');
      const uncachedItemIds = getUncachedItemIds(itemIds, cachedItemIds, ranges);
      if (uncachedItemIds.length > 0) {
        getQueueItemDTOsByItemIds({ item_ids: uncachedItemIds })
          .unwrap()
          .catch(() => {
            // This bulk fetch is the ONLY fetcher for these rows: `QueueItemAtPosition` consumes
            // the cache with `skip: isUninitialized`, so a row whose DTO never arrived does not
            // fetch for itself. Put the ranges back so the effect re-runs and tries again —
            // otherwise a transient failure leaves placeholders until the user happens to scroll.
            setPendingRanges((prev) => (prev.length > 0 ? prev : ranges));
          });
      }
      // Clear unconditionally. Returning early without clearing (the previous behaviour when
      // everything was already cached) let ranges accumulate for the lifetime of the list,
      // growing the scan on every subsequent pass.
      //
      // Clear with a stable reference. `pendingRanges` is a dependency of the effect that calls
      // this function, so a fresh `[]` — a new identity every time — re-runs the effect, which
      // re-arms the throttle, which calls this again. The old early return happened to prevent
      // that while everything was cached, so the loop only ran while items were genuinely
      // uncached; clearing on both paths means the stable reference is now what stops it.
      setPendingRanges(EMPTY_ARRAY);
    },
    [enabled, getQueueItemDTOsByItemIds, store]
  );

  const throttledFetchQueueItems = useThrottledCallback(fetchQueueItems, 500);

  const onRangeChanged = useCallback((range: ListRange) => {
    setLastRange(range);
    setPendingRanges((prev) => [...prev, range]);
  }, []);

  useEffect(() => {
    const combinedRanges = lastRange ? [...pendingRanges, lastRange] : pendingRanges;
    throttledFetchQueueItems(combinedRanges, itemIds);
  }, [itemIds, lastRange, pendingRanges, throttledFetchQueueItems]);

  return {
    onRangeChanged,
  };
};
