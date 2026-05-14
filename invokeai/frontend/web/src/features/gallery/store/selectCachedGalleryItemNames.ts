import type { AppGetState } from 'app/store/store';
import { galleryApi } from 'services/api/endpoints/gallery';

/**
 * Returns the names (in display order) of the currently-cached gallery item list.
 *
 * The grid renders via the polymorphic ``getGalleryItemNames`` endpoint, which returns a
 * mixed image+video list. Range-selection click handlers (shift-click for ranges, ctrl-click
 * for discontiguous selection) need that ordered list to compute the items between two
 * clicks. Reading from the older image-only ``getImageNames`` cache would silently fail on
 * any board with videos and, since that cache is no longer the grid's source of truth,
 * fail on image-only boards too once the cache evicts.
 *
 * For most sessions there is exactly one active list at a time. We scan the RTK Query
 * subscriptions invalidated by ``GalleryItemNameList`` and return the first hit.
 */
export const selectCachedGalleryItemNames = (state: ReturnType<AppGetState>): string[] => {
  const entries = galleryApi.util.selectInvalidatedBy(state, ['GalleryItemNameList']);
  for (const entry of entries) {
    if (entry.endpointName !== 'getGalleryItemNames') {
      continue;
    }
    const data = galleryApi.endpoints.getGalleryItemNames.select(entry.originalArgs)(state).data;
    if (data) {
      return data.items.map((ref) => ref.name);
    }
  }
  return [];
};
