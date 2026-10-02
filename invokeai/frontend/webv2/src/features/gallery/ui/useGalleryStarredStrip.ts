import type { GalleryItemsFilter } from '@features/gallery/data/queries';

import { galleryStarredStripOptions } from '@features/gallery/data/queries';
import { useQuery } from '@tanstack/react-query';
import { useMemo } from 'react';

import type { GalleryStarredStrip } from './GalleryWidgetContext';

export const EMPTY_GALLERY_STARRED_STRIP: GalleryStarredStrip = { items: [], total: 0 };

/**
 * The bounded starred strip for the listing `filter` describes. Disabled
 * strips report empty rather than their last cached page, so the section
 * leaves the grid the moment it stops applying.
 */
export const useGalleryStarredStrip = ({
  enabled,
  filter,
}: {
  enabled: boolean;
  filter: GalleryItemsFilter;
}): GalleryStarredStrip => {
  const { data } = useQuery({ ...galleryStarredStripOptions(filter), enabled });

  return useMemo(
    () => (enabled && data ? { items: data.items, total: data.total } : EMPTY_GALLERY_STARRED_STRIP),
    [data, enabled]
  );
};
