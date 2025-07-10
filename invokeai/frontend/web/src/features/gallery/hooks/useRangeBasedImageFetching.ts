import { useAppStore } from 'app/store/storeHooks';
import { useCallback, useEffect, useState } from 'react';
import type { ListRange } from 'react-virtuoso';
import { imagesApi, useGetImageDTOsByNamesMutation } from 'services/api/endpoints/images';
import { useThrottledCallback } from 'use-debounce';

interface UseRangeBasedImageFetchingArgs {
  imageNames: string[];
  enabled: boolean;
}

interface UseRangeBasedImageFetchingReturn {
  onRangeChanged: (range: ListRange) => void;
}

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
 * Hook for bulk fetching image DTOs based on the visible range from virtuoso.
 * Individual image components should use `useGetImageDTOQuery(imageName)` to get their specific DTO.
 * This hook ensures DTOs are bulk fetched and cached efficiently.
 */
export const useRangeBasedImageFetching = ({
  imageNames,
  enabled,
}: UseRangeBasedImageFetchingArgs): UseRangeBasedImageFetchingReturn => {
  const store = useAppStore();
  const [getImageDTOsByNames] = useGetImageDTOsByNamesMutation();
  const [lastRange, setLastRange] = useState<ListRange | null>(null);
  const [pendingRanges, setPendingRanges] = useState<ListRange[]>([]);

  const fetchImages = useCallback(
    (ranges: ListRange[], imageNames: string[]) => {
      if (!enabled) {
        return;
      }
      const cachedImageNames = imagesApi.util.selectCachedArgsForQuery(store.getState(), 'getImageDTO');
      const uncachedNames = getUncachedNames(imageNames, cachedImageNames, ranges);
      if (uncachedNames.length === 0) {
        return;
      }
      getImageDTOsByNames({ image_names: uncachedNames });
      setPendingRanges([]);
    },
    [enabled, getImageDTOsByNames, store]
  );

  const throttledFetchImages = useThrottledCallback(fetchImages, 500);

  const onRangeChanged = useCallback((range: ListRange) => {
    setLastRange(range);
    setPendingRanges((prev) => [...prev, range]);
  }, []);

  useEffect(() => {
    const combinedRanges = lastRange ? [...pendingRanges, lastRange] : pendingRanges;
    throttledFetchImages(combinedRanges, imageNames);
  }, [imageNames, lastRange, pendingRanges, throttledFetchImages]);

  return {
    onRangeChanged,
  };
};
