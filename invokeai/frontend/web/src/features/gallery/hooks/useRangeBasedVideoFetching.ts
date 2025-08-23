import { useAppStore } from 'app/store/storeHooks';
import { useCallback, useEffect, useState } from 'react';
import type { ListRange } from 'react-virtuoso';
import { useGetVideoDTOsByNamesMutation, videosApi } from 'services/api/endpoints/videos';
import { useThrottledCallback } from 'use-debounce';

interface UseRangeBasedVideoFetchingArgs {
  videoIds: string[];
  enabled: boolean;
}

interface UseRangeBasedVideoFetchingReturn {
  onRangeChanged: (range: ListRange) => void;
}

const getUncachedIds = (videoIds: string[], cachedVideoIds: string[], ranges: ListRange[]): string[] => {
  const uncachedIdsSet = new Set<string>();
  const cachedVideoIdsSet = new Set(cachedVideoIds);

  for (const range of ranges) {
    for (let i = range.startIndex; i <= range.endIndex; i++) {
      const id = videoIds[i]!;
      if (id && !cachedVideoIdsSet.has(id)) {
        uncachedIdsSet.add(id);
      }
    }
  }

  return Array.from(uncachedIdsSet);
};

/**
 * Hook for bulk fetching image DTOs based on the visible range from virtuoso.
 * Individual image components should use `useGetImageDTOQuery(imageName)` to get their specific DTO.
 * This hook ensures DTOs are bulk fetched and cached efficiently.
 */
export const useRangeBasedVideoFetching = ({
  videoIds,
  enabled,
}: UseRangeBasedVideoFetchingArgs): UseRangeBasedVideoFetchingReturn => {
  const store = useAppStore();
  const [getVideoDTOsByNames] = useGetVideoDTOsByNamesMutation();
  const [lastRange, setLastRange] = useState<ListRange | null>(null);
  const [pendingRanges, setPendingRanges] = useState<ListRange[]>([]);

  const fetchVideos = useCallback(
    (ranges: ListRange[], videoIds: string[]) => {
      if (!enabled) {
        return;
      }
      const cachedVideoIds = videosApi.util.selectCachedArgsForQuery(store.getState(), 'getVideoDTO');
      const uncachedIds = getUncachedIds(videoIds, cachedVideoIds, ranges);
      // console.log('uncachedIds', uncachedIds);
      if (uncachedIds.length === 0) {
        return;
      }
      getVideoDTOsByNames({ video_ids: uncachedIds });
      setPendingRanges([]);
    },
    [enabled, getVideoDTOsByNames, store]
  );

  const throttledFetchVideos = useThrottledCallback(fetchVideos, 500);

  const onRangeChanged = useCallback((range: ListRange) => {
    setLastRange(range);
    setPendingRanges((prev) => [...prev, range]);
  }, []);

  useEffect(() => {
    const combinedRanges = lastRange ? [...pendingRanges, lastRange] : pendingRanges;
    throttledFetchVideos(combinedRanges, videoIds);
  }, [videoIds, lastRange, pendingRanges, throttledFetchVideos]);

  return {
    onRangeChanged,
  };
};
