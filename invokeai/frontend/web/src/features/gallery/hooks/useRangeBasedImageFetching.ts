import { useAppStore } from 'app/store/storeHooks';
import { isVideoName } from 'features/gallery/store/types';
import { useCallback, useEffect, useState } from 'react';
import type { ListRange } from 'react-virtuoso';
import { imagesApi, useGetImageDTOsByNamesMutation } from 'services/api/endpoints/images';
import { videosApi } from 'services/api/endpoints/videos';
import { useThrottledCallback } from 'use-debounce';

interface UseRangeBasedImageFetchingArgs {
  imageNames: string[];
  enabled: boolean;
}

interface UseRangeBasedImageFetchingReturn {
  onRangeChanged: (range: ListRange) => void;
}

export const getVideoPrefetchOptions = () => ({ subscribe: false, forceRefetch: true }) as const;

export const hasCachedVideoDTO = (queryState: { data?: unknown; isError?: boolean }): boolean =>
  queryState.data !== undefined;

const getUncachedNames = (imageNames: string[], cachedImageNames: string[], ranges: ListRange[]): string[] => {
  const uncachedNamesSet = new Set<string>();
  const cachedImageNamesSet = new Set(cachedImageNames);

  for (const range of ranges) {
    for (let i = range.startIndex; i <= range.endIndex; i++) {
      const n = imageNames[i]!;
      if (n && !cachedImageNamesSet.has(n)) {
        uncachedNamesSet.add(n);
      }
    }
  }

  return Array.from(uncachedNamesSet);
};

/**
 * Hook for bulk fetching gallery item DTOs based on the visible range from virtuoso.
 *
 * Names are polymorphic — image names go through the bulk `getImageDTOsByNames` mutation while
 * video names dispatch individual `getVideoDTO` queries (the videos API doesn't have a batch
 * endpoint yet; per-item is fine while video counts are low). Individual components still call
 * `useGetImageDTOQuery` / `useGetVideoDTOQuery` to subscribe — this hook only triggers fetches.
 */
export const useRangeBasedImageFetching = ({
  imageNames,
  enabled,
}: UseRangeBasedImageFetchingArgs): UseRangeBasedImageFetchingReturn => {
  const store = useAppStore();
  const [getImageDTOsByNames] = useGetImageDTOsByNamesMutation();
  const [lastRange, setLastRange] = useState<ListRange | null>(null);
  const [pendingRanges, setPendingRanges] = useState<ListRange[]>([]);

  const fetchItems = useCallback(
    (ranges: ListRange[], allNames: string[]) => {
      if (!enabled) {
        return;
      }
      const state = store.getState();

      // Images — bulk fetch via the existing batch endpoint.
      const cachedImageNames = imagesApi.util.selectCachedArgsForQuery(state, 'getImageDTO');
      const uncachedImageNames = getUncachedNames(allNames, cachedImageNames, ranges).filter((n) => !isVideoName(n));
      if (uncachedImageNames.length > 0) {
        getImageDTOsByNames({ image_names: uncachedImageNames });
      }

      // Videos — fetch one at a time (no batch endpoint yet). Each `initiate()` is a no-op for
      // already-cached entries, so this is safe to call repeatedly while scrolling.
      const cachedVideoNames = videosApi.util
        .selectCachedArgsForQuery(state, 'getVideoDTO')
        .filter((videoName) => hasCachedVideoDTO(videosApi.endpoints.getVideoDTO.select(videoName)(state)));
      const uncachedVideoNames = getUncachedNames(allNames, cachedVideoNames, ranges).filter((n) => isVideoName(n));
      for (const videoName of uncachedVideoNames) {
        store.dispatch(videosApi.endpoints.getVideoDTO.initiate(videoName, getVideoPrefetchOptions()));
      }

      setPendingRanges([]);
    },
    [enabled, getImageDTOsByNames, store]
  );

  const throttledFetchItems = useThrottledCallback(fetchItems, 500);

  const onRangeChanged = useCallback((range: ListRange) => {
    setLastRange(range);
    setPendingRanges((prev) => [...prev, range]);
  }, []);

  useEffect(() => {
    const combinedRanges = lastRange ? [...pendingRanges, lastRange] : pendingRanges;
    throttledFetchItems(combinedRanges, imageNames);
  }, [imageNames, lastRange, pendingRanges, throttledFetchItems]);

  return {
    onRangeChanged,
  };
};
