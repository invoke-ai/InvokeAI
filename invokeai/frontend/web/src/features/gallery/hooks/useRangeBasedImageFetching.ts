import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppStore } from 'app/store/storeHooks';
import { coalesceRanges, useBoundedRangeRetry } from 'common/hooks/useBoundedRangeRetry';
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
  const [pendingRanges, setPendingRanges] = useState<ListRange[]>(EMPTY_ARRAY);

  const restoreFailedRanges = useCallback((failedRanges: ListRange[]) => {
    // Merge with whatever is pending — replacing either side would drop ranges the user reported
    // while the failed fetch was in flight, or ranges that failed while the user was scrolling.
    setPendingRanges((prev) => (prev.length > 0 ? coalesceRanges([...prev, ...failedRanges]) : failedRanges));
  }, []);
  const { onFetchFailure, resetRetryBudget } = useBoundedRangeRetry(restoreFailedRanges);

  const fetchItems = useCallback(
    (ranges: ListRange[], allNames: string[]) => {
      if (!enabled) {
        // Clear here too, for the same reason as the clear at the end of this callback: returning
        // early while disabled let ranges pile up until `enabled` flipped, so the first enabled
        // pass scanned every range reported during the disabled window instead of the viewport.
        setPendingRanges(EMPTY_ARRAY);
        return;
      }
      const state = store.getState();

      // Images — bulk fetch via the existing batch endpoint.
      const cachedImageNames = imagesApi.util.selectCachedArgsForQuery(state, 'getImageDTO');
      const uncachedImageNames = getUncachedNames(allNames, cachedImageNames, ranges).filter((n) => !isVideoName(n));
      if (uncachedImageNames.length > 0) {
        getImageDTOsByNames({ image_names: uncachedImageNames })
          .unwrap()
          .then(resetRetryBudget)
          .catch(() => {
            // This bulk fetch is the ONLY fetcher for these rows: `ImageAtPosition` consumes the
            // cache with `skip: isUninitialized`, so a row whose DTO never arrived does not fetch
            // for itself, and images (unlike videos) have no retry affordance. Hand the ranges to
            // the bounded retry so they are restored after a backoff — otherwise a transient
            // failure leaves grey placeholders until the user happens to scroll.
            onFetchFailure(ranges);
          });
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

      // Clear with a stable reference. `pendingRanges` is a dependency of the effect that
      // calls this function, so a fresh `[]` — a new identity every time — re-runs the
      // effect, which re-arms the throttle, which calls this again: a self-sustaining
      // render loop, running as fast as the throttle allows, for as long as the grid is
      // mounted and with no user input. Setting state to the value it already holds makes
      // React bail out instead.
      setPendingRanges(EMPTY_ARRAY);
    },
    [enabled, getImageDTOsByNames, onFetchFailure, resetRetryBudget, store]
  );

  const throttledFetchItems = useThrottledCallback(fetchItems, 500);

  const onRangeChanged = useCallback(
    (range: ListRange) => {
      // A new range report is fresh user input — restart the retry budget so a grid that gave up
      // after sustained failure resumes retrying as the user scrolls.
      resetRetryBudget();
      setLastRange(range);
      setPendingRanges((prev) => [...prev, range]);
    },
    [resetRetryBudget]
  );

  useEffect(() => {
    const combinedRanges = lastRange ? [...pendingRanges, lastRange] : pendingRanges;
    throttledFetchItems(combinedRanges, imageNames);
  }, [imageNames, lastRange, pendingRanges, throttledFetchItems]);

  return {
    onRangeChanged,
  };
};
