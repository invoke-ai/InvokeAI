import type { GalleryItemsFilter } from '@features/gallery/data/queries';

import { galleryStarredStripOptions } from '@features/gallery/data/queries';
import { useQuery } from '@tanstack/react-query';
import { useCallback, useMemo } from 'react';

import type { GalleryStarredStrip } from './GalleryWidgetContext';

import { getGalleryReadStatus } from './galleryStateView';

const retryNothing = () => Promise.resolve();

export const EMPTY_GALLERY_STARRED_STRIP: GalleryStarredStrip = {
  items: [],
  state: { error: null, isRetrying: false, retry: retryNothing, status: 'empty' },
  total: 0,
};

/**
 * The bounded starred strip for the listing `filter` describes. Disabled
 * strips report empty rather than their last cached page (or failure), so the
 * section leaves the grid the moment it stops applying.
 */
export const useGalleryStarredStrip = ({
  enabled,
  filter,
}: {
  enabled: boolean;
  filter: GalleryItemsFilter;
}): GalleryStarredStrip => {
  const { data, error, errorUpdateCount, isError, isFetching, refetch } = useQuery({
    ...galleryStarredStripOptions(filter),
    enabled,
  });
  const status = getGalleryReadStatus(
    { errorUpdateCount, hasData: data !== undefined, isError },
    data?.items.length ?? 0
  );
  const isRetrying = (status === 'error' || status === 'stale-error') && isFetching;
  const retry = useCallback(async () => {
    await refetch();
  }, [refetch]);

  return useMemo(
    () =>
      enabled
        ? {
            items: data?.items ?? EMPTY_GALLERY_STARRED_STRIP.items,
            state: { error, isRetrying, retry, status },
            total: data?.total ?? 0,
          }
        : EMPTY_GALLERY_STARRED_STRIP,
    [data, enabled, error, isRetrying, retry, status]
  );
};
