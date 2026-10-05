import type { GalleryItemRef } from '@features/gallery/contracts';

import { galleryItems } from '@features/gallery';
import { toGalleryItemKey } from '@features/gallery/contracts';
import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';

/**
 * Hover thumbnail and optional video duration derive from immutable item names and can be cached independently of
 * mutable DTOs.
 */
export interface HoverThumbnail {
  url: string;
  /** Seconds, for videos only; images have no duration to show. */
  durationSeconds: number | null;
}

// A video's thumbnail is the frame the index embedded it by, so the hover
// preview shows exactly what the map placed.
const urls = new Map<string, HoverThumbnail | null>();
const inflight = new Map<string, Promise<HoverThumbnail | null>>();

// Thumbnail URLs are account-owned gallery data: drop them on login/logout so
// one account's map hovers can never serve another account's thumbnails.
registerAccountOwnedResource({
  clear: () => {
    urls.clear();
    inflight.clear();
  },
  name: 'image-map-thumbnails',
});

export const getThumbnailUrl = (item: GalleryItemRef): Promise<HoverThumbnail | null> => {
  const key = toGalleryItemKey(item);
  const cached = urls.get(key);

  if (cached !== undefined) {
    return Promise.resolve(cached);
  }

  const pending = inflight.get(key);

  if (pending) {
    return pending;
  }

  const owner = captureAccountScope();
  const request = galleryItems
    .resolve(item)
    .then((resolved): HoverThumbnail => {
      const thumbnail: HoverThumbnail = {
        durationSeconds: resolved.kind === 'video' ? resolved.durationSeconds : null,
        url: resolved.thumbnailUrl,
      };

      // A resolution that raced an account switch must not seed the next
      // account's cache.
      if (isAccountScopeCurrent(owner)) {
        urls.set(key, thumbnail);
      }

      return thumbnail;
    })
    .catch((error: unknown): null => {
      // Remember definitive missing/invisible-item responses so each hover does not repeat the same 404.
      if (error instanceof ApiError && (error.status === 403 || error.status === 404) && isAccountScopeCurrent(owner)) {
        urls.set(key, null);
      }

      return null;
    })
    .finally(() => {
      // Release only this request's claim; an account switch already cleared
      // the in-flight map.
      if (inflight.get(key) === request) {
        inflight.delete(key);
      }
    });
  inflight.set(key, request);

  return request;
};
