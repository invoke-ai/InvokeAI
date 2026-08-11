import { useAppStore } from 'app/store/storeHooks';
import { useCallback, useEffect, useState } from 'react';
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
  const [pendingRanges, setPendingRanges] = useState<ListRange[]>([]);

  const fetchQueueItems = useCallback(
    (ranges: ListRange[], itemIds: number[]) => {
      if (!enabled) {
        return;
      }
      const cachedItemIds = queueApi.util.selectCachedArgsForQuery(store.getState(), 'getQueueItemSummary');
      const uncachedItemIds = getUncachedItemIds(itemIds, cachedItemIds, ranges);
      if (uncachedItemIds.length === 0) {
        return;
      }
      for (const item_ids of getItemIdBatches(uncachedItemIds)) {
        getQueueItemSummariesByItemIds({ item_ids });
      }
      setPendingRanges([]);
    },
    [enabled, getQueueItemSummariesByItemIds, store]
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
