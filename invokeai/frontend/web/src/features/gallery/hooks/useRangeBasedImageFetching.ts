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

const getUncachedNames = (imageNames: string[], cachedImageNames: string[], range: ListRange): string[] => {
  if (range.startIndex === range.endIndex) {
    // If the start and end indices are the same, no range to fetch
    return [];
  }

  if (imageNames.length === 0) {
    return [];
  }

  const start = Math.max(0, range.startIndex);
  const end = Math.min(imageNames.length - 1, range.endIndex);

  if (cachedImageNames.length === 0) {
    return imageNames.slice(start, end + 1);
  }

  const uncachedNames: string[] = [];

  for (let i = start; i <= end; i++) {
    const imageName = imageNames[i]!;
    if (!cachedImageNames.includes(imageName)) {
      uncachedNames.push(imageName);
    }
  }

  return uncachedNames;
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
  const [visibleRange, setVisibleRange] = useState<ListRange>({ startIndex: 0, endIndex: 0 });
  const [getImageDTOsByNames] = useGetImageDTOsByNamesMutation();

  const fetchImages = useCallback(
    (visibleRange: ListRange) => {
      const cachedImageNames = imagesApi.util.selectCachedArgsForQuery(store.getState(), 'getImageDTO');
      const uncachedNames = getUncachedNames(imageNames, cachedImageNames, visibleRange);
      if (uncachedNames.length === 0) {
        return;
      }
      getImageDTOsByNames({ image_names: uncachedNames });
    },
    [getImageDTOsByNames, imageNames, store]
  );

  const throttledFetchImages = useThrottledCallback(fetchImages, 100);

  useEffect(() => {
    if (!enabled) {
      return;
    }
    throttledFetchImages(visibleRange);
  }, [enabled, throttledFetchImages, imageNames, visibleRange]);

  const onRangeChanged = useCallback((range: ListRange) => {
    setVisibleRange(range);
  }, []);

  return {
    onRangeChanged,
  };
};
