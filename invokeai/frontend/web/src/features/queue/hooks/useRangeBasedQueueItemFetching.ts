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
  const [pendingRanges, setPendingRanges] = useState<ListRange[]>([]);

  const fetchQueueItems = useCallback(
    (ranges: ListRange[], itemIds: number[]) => {
      if (!enabled) {
        return;
      }
      const cachedItemIds = queueApi.util.selectCachedArgsForQuery(store.getState(), 'getQueueItem');
      const uncachedItemIds = getUncachedItemIds(itemIds, cachedItemIds, ranges);
      if (uncachedItemIds.length === 0) {
        return;
      }
      getQueueItemDTOsByItemIds({ item_ids: uncachedItemIds });
      setPendingRanges([]);
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
