import { galleryImages } from '@features/gallery';
import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';

/**
 * Image-name → thumbnail-URL cache for map hover previews. Thumbnail URLs are
 * derived from the immutable image name, so unlike full DTOs they never go
 * stale and can be cached for the session.
 */
const urls = new Map<string, string | null>();
const inflight = new Map<string, Promise<string | null>>();

// Thumbnail URLs are account-owned gallery data: drop them on login/logout so
// one account's map hovers can never serve another account's thumbnails.
registerAccountOwnedResource({
  clear: () => {
    urls.clear();
    inflight.clear();
  },
  name: 'image-map-thumbnails',
});

export const getThumbnailUrl = (imageName: string): Promise<string | null> => {
  const cached = urls.get(imageName);

  if (cached !== undefined) {
    return Promise.resolve(cached);
  }

  const pending = inflight.get(imageName);

  if (pending) {
    return pending;
  }

  const owner = captureAccountScope();
  const request = galleryImages
    .resolveMany([imageName])
    .then((images) => {
      const url = images.at(0)?.thumbnailUrl ?? null;

      // A resolution that raced an account switch must not seed the next
      // account's cache.
      if (isAccountScopeCurrent(owner)) {
        urls.set(imageName, url);
      }

      return url;
    })
    .catch(() => null)
    .finally(() => {
      // Release only this request's claim; an account switch already cleared
      // the in-flight map.
      if (inflight.get(imageName) === request) {
        inflight.delete(imageName);
      }
    });
  inflight.set(imageName, request);

  return request;
};
