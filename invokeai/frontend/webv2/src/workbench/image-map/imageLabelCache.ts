import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';

import type { ImageMapImageLabels } from './api';

import { fetchImageMapImageLabels } from './api';

/**
 * Image-name → vocabulary-labels cache for map hover cards. Labels derive from
 * the image's stored embedding, which never changes, and the labeling
 * vocabulary, which does: an admin can edit the supplementary term list. So
 * results (including "no labels for this image") are cached until a
 * vocabulary rebuild lands, at which point `clearImageLabels` drops them.
 */
const labels = new Map<string, ImageMapImageLabels | null>();
const inflight = new Map<string, Promise<ImageMapImageLabels | null>>();

/**
 * How long a server-wide failure suppresses further requests. Such a failure
 * is not about this image, so hovering across the map would otherwise fire one
 * doomed request per point — but it is not necessarily permanent either: the
 * vocabulary is built lazily by the index worker, and every call until that
 * build lands answers "still being prepared". A cooldown backs off without
 * giving up.
 */
const UNAVAILABLE_COOLDOWN_MS = 60_000;

let unavailableUntil = 0;

// Labels are account-owned gallery data: drop them on login/logout so one
// account's hovers can never serve another account's labels. The cooldown is
// account state too — a different login may have a different backend.
registerAccountOwnedResource({
  clear: () => {
    labels.clear();
    inflight.clear();
    unavailableUntil = 0;
  },
  name: 'image-map-image-labels',
});

/**
 * Drop every cached label. Called when a supplementary-vocabulary rebuild
 * lands: the same phrases the cluster labels are re-fetched for also decide
 * these, and a card pairing freshly-fetched cluster tags with tags scored
 * against the previous vocabulary would be quietly inconsistent. Also clears
 * the cooldown, since a completed rebuild is exactly what a 409 was waiting
 * for.
 */
export const clearImageLabels = (): void => {
  labels.clear();
  unavailableUntil = 0;
};

export const getImageLabels = (imageName: string): Promise<ImageMapImageLabels | null> => {
  // Checked before the cooldown: labels already fetched for this image stay
  // available even while a server-wide 409 is being backed off.
  const cached = labels.get(imageName);

  if (cached !== undefined) {
    return Promise.resolve(cached);
  }

  const pending = inflight.get(imageName);

  if (pending) {
    return pending;
  }

  if (Date.now() < unavailableUntil) {
    return Promise.resolve(null);
  }

  const owner = captureAccountScope();
  const request = fetchImageMapImageLabels(imageName)
    .then((result): ImageMapImageLabels | null => {
      // A resolution that raced an account switch must not seed the next
      // account's cache.
      if (isAccountScopeCurrent(owner)) {
        labels.set(imageName, result);
      }

      return result;
    })
    .catch((error: unknown): null => {
      if (!isAccountScopeCurrent(owner)) {
        return null;
      }

      if (error instanceof ApiError && (error.status === 409 || error.status >= 500)) {
        // Server-wide and possibly temporary: a 409 means the vocabulary may
        // still be building, a 5xx that the backend is unwell. Back off rather
        // than cache anything per image — but back off, because this is driven
        // by pointer movement, and a deterministic failure would otherwise
        // refire several times a second for the rest of the session.
        unavailableUntil = Date.now() + UNAVAILABLE_COOLDOWN_MS;
      } else if (error instanceof ApiError && (error.status === 403 || error.status === 404)) {
        // The only definitive per-image answers: not indexed, or not visible
        // to this account. Nothing about this session can change either.
        labels.set(imageName, null);
      }

      // Everything else (5xx, 429, a dropped connection) is transient and is
      // deliberately not cached: the next hover of this image retries.
      return null;
    })
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
