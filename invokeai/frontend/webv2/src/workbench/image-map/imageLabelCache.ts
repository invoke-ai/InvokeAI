import type { GalleryItemRef } from '@features/gallery/contracts';

import { toGalleryItemKey } from '@features/gallery/contracts';
import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';

import type { ImageMapImageLabels } from './api';

import { fetchImageMapImageLabels } from './api';

/**
 * Caches per-item vocabulary labels, including empty results, until vocabulary rebuild. Embeddings are stable;
 * editable vocabulary determines invalidation.
 */
const labels = new Map<string, ImageMapImageLabels | null>();
const inflight = new Map<string, Promise<ImageMapImageLabels | null>>();

/**
 * Server-wide cooldowns prevent one failing request per hovered point or thumbnail while allowing recovery. A 409
 * usually means the index worker is still building the vocabulary, which on a warm disk cache lands in about a
 * second, so it backs off briefly; outages back off longer.
 */
const BUILDING_COOLDOWN_MS = 5_000;
const OUTAGE_COOLDOWN_MS = 60_000;

let unavailableUntil = 0;
/** Bumped by every clear so requests issued before it cannot answer or seed the cache after it. */
let generation = 0;

// Clear labels and cooldown on login/logout to prevent cross-account results or backend failure state.
registerAccountOwnedResource({
  clear: () => {
    generation += 1;
    labels.clear();
    inflight.clear();
    unavailableUntil = 0;
  },
  name: 'image-map-image-labels',
});

/**
 * Vocabulary rebuild clears hover labels and cooldown so item tags agree with refreshed cluster labels and
 * pending-build retries resume.
 */
export const clearImageLabels = (): void => {
  generation += 1;
  labels.clear();
  inflight.clear();
  unavailableUntil = 0;
};

export const getImageLabels = (item: GalleryItemRef): Promise<ImageMapImageLabels | null> => {
  const key = toGalleryItemKey(item);
  // Checked before the cooldown: labels already fetched for this item stay
  // available even while a server-wide 409 is being backed off.
  const cached = labels.get(key);

  if (cached !== undefined) {
    return Promise.resolve(cached);
  }

  const pending = inflight.get(key);

  if (pending) {
    return pending;
  }

  if (Date.now() < unavailableUntil) {
    return Promise.resolve(null);
  }

  const owner = captureAccountScope();
  const issuedIn = generation;
  // A resolution that raced an account switch or vocabulary rebuild answers for state that is gone: it must
  // neither seed the cache nor reach the caller.
  const isCurrent = () => issuedIn === generation && isAccountScopeCurrent(owner);
  const request = fetchImageMapImageLabels(item)
    .then((result): ImageMapImageLabels | null => {
      if (!isCurrent()) {
        return null;
      }

      labels.set(key, result);
      return result;
    })
    .catch((error: unknown): null => {
      if (!isCurrent()) {
        return null;
      }

      if (error instanceof ApiError && error.status === 409) {
        // Build in progress (or index/vocabulary unavailable): server-wide, not per item.
        unavailableUntil = Date.now() + BUILDING_COOLDOWN_MS;
      } else if (error instanceof ApiError && error.status >= 500) {
        unavailableUntil = Date.now() + OUTAGE_COOLDOWN_MS;
      } else if (error instanceof ApiError && (error.status === 403 || error.status === 404)) {
        // Cache definitive item misses: not indexed or not visible to this account.
        labels.set(key, null);
      }

      // Other transient failures remain uncached so later hover retries.
      return null;
    })
    .finally(() => {
      // Release only this request's claim; a clear already emptied the
      // in-flight map.
      if (inflight.get(key) === request) {
        inflight.delete(key);
      }
    });
  inflight.set(key, request);

  return request;
};
