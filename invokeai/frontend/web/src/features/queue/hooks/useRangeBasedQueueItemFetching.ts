import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppStore } from 'app/store/storeHooks';
import { coalesceRanges, useBoundedRangeRetry } from 'common/hooks/useBoundedRangeRetry';
import { useCallback, useEffect, useRef, useState } from 'react';
import type { ListRange } from 'react-virtuoso';
import { queueApi, useGetQueueItemSummariesByItemIdsMutation } from 'services/api/endpoints/queue';
import { useThrottledCallback } from 'use-debounce';

interface UseRangeBasedQueueItemFetchingArgs {
  itemIds: number[];
  enabled: boolean;
}

interface UseRangeBasedQueueItemFetchingReturn {
  onRangeChanged: (range: ListRange) => void;
}

/**
 * Mirrors MAX_QUEUE_ITEM_IDS_PER_REQUEST on the backend, which rejects larger batches outright.
 * A fast fling can union enough visible ranges to exceed it, so split instead of risking a 422.
 */
const MAX_ITEM_IDS_PER_REQUEST = 1000;

export const getItemIdBatches = (itemIds: number[]): number[][] => {
  const batches: number[][] = [];
  for (let i = 0; i < itemIds.length; i += MAX_ITEM_IDS_PER_REQUEST) {
    batches.push(itemIds.slice(i, i + MAX_ITEM_IDS_PER_REQUEST));
  }
  return batches;
};

export const getUncachedItemIds = (
  itemIds: number[],
  cachedItemIds: number[],
  ranges: ListRange[],
  pendingItemIds: Set<number> = new Set()
): number[] => {
  const uncachedItemIdsSet = new Set<number>();
  const cachedItemIdsSet = new Set(cachedItemIds);

  for (const range of ranges) {
    for (let i = range.startIndex; i <= range.endIndex; i++) {
      const n = itemIds[i]!;
      if (n && !cachedItemIdsSet.has(n) && !pendingItemIds.has(n)) {
        uncachedItemIdsSet.add(n);
      }
    }
  }

  return Array.from(uncachedItemIdsSet);
};

/**
 * Hook for bulk fetching queue item summaries based on the visible range from virtuoso.
 * Individual queue item components read the cached summary via `getQueueItemSummary`; only the
 * expanded detail view fetches the full item, which is what keeps the session graph and workflow
 * off the wire while scrolling.
 * This hook ensures summaries are bulk fetched and cached efficiently.
 */
export const useRangeBasedQueueItemFetching = ({
  itemIds,
  enabled,
}: UseRangeBasedQueueItemFetchingArgs): UseRangeBasedQueueItemFetchingReturn => {
  const store = useAppStore();
  const [getQueueItemSummariesByItemIds] = useGetQueueItemSummariesByItemIdsMutation();
  const [lastRange, setLastRange] = useState<ListRange | null>(null);
  const [pendingRanges, setPendingRanges] = useState<ListRange[]>(EMPTY_ARRAY);
  const pendingItemIdsRef = useRef<Set<number>>(new Set());

  const restoreFailedRanges = useCallback((failedRanges: ListRange[]) => {
    // Merge with whatever is pending — replacing either side would drop ranges the user reported
    // while the failed fetch was in flight, or ranges that failed while the user was scrolling.
    setPendingRanges((prev) => (prev.length > 0 ? coalesceRanges([...prev, ...failedRanges]) : failedRanges));
  }, []);
  const { onFetchFailure, resetRetryBudget } = useBoundedRangeRetry(restoreFailedRanges);

  const fetchQueueItems = useCallback(
    (ranges: ListRange[], itemIds: number[]) => {
      if (!enabled) {
        return;
      }
      const cachedItemIds = queueApi.util.selectCachedArgsForQuery(store.getState(), 'getQueueItemSummary');
      const uncachedItemIds = getUncachedItemIds(itemIds, cachedItemIds, ranges, pendingItemIdsRef.current);
      for (const item_ids of getItemIdBatches(uncachedItemIds)) {
        item_ids.forEach((item_id) => pendingItemIdsRef.current.add(item_id));
        void getQueueItemSummariesByItemIds({ item_ids })
          .unwrap()
          .then(
            () => {
              item_ids.forEach((item_id) => pendingItemIdsRef.current.delete(item_id));
              resetRetryBudget();
            },
            () => {
              item_ids.forEach((item_id) => pendingItemIdsRef.current.delete(item_id));
              // This bulk fetch is the ONLY fetcher for these rows: `QueueItemAtPosition` consumes
              // the cache with `skip: isUninitialized`, so a row whose summary never arrived does
              // not fetch for itself. Hand the ranges to the bounded retry so they are restored
              // after a backoff — otherwise a transient failure leaves placeholders until the user
              // happens to scroll.
              onFetchFailure(ranges);
            }
          );
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
    [enabled, getQueueItemSummariesByItemIds, onFetchFailure, resetRetryBudget, store]
  );

  const throttledFetchQueueItems = useThrottledCallback(fetchQueueItems, 500);

  const onRangeChanged = useCallback(
    (range: ListRange) => {
      // A new range report is fresh user input — restart the retry budget so a list that gave up
      // after sustained failure resumes retrying as the user scrolls.
      resetRetryBudget();
      setLastRange(range);
      setPendingRanges((prev) => [...prev, range]);
    },
    [resetRetryBudget]
  );

  useEffect(() => {
    const combinedRanges = lastRange ? [...pendingRanges, lastRange] : pendingRanges;
    throttledFetchQueueItems(combinedRanges, itemIds);
  }, [itemIds, lastRange, pendingRanges, throttledFetchQueueItems]);

  return {
    onRangeChanged,
  };
};
