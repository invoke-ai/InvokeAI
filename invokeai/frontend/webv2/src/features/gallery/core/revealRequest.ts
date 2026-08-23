import type { GalleryItemKey } from './items';

/**
 * An explicit "scroll this item into view" signal from surfaces outside the
 * grid (the image map's reveal). Deliberately NOT derived from the selection:
 * the selection also changes when a finished generation auto-selects its
 * image, and scrolling on that yanked the grid out from under a browsing
 * user. A reveal is a deliberate gesture, so it gets its own channel — and a
 * token, so repeating the same gesture (re-clicking the same map point after
 * scrolling away) reveals again even though the selection is unchanged.
 *
 * Module-scoped rather than persisted widget state: a reveal is an ephemeral
 * intent for the currently mounted grid, and persisting it would replay a
 * stale scroll on the next session.
 */

export interface GalleryRevealRequest {
  itemKey: GalleryItemKey;
  token: number;
}

let currentRequest: GalleryRevealRequest | null = null;
let nextToken = 0;

const listeners = new Set<() => void>();

export const requestGalleryItemReveal = (itemKey: GalleryItemKey): void => {
  nextToken += 1;
  currentRequest = { itemKey, token: nextToken };
  for (const listener of listeners) {
    listener();
  }
};

export const getGalleryRevealRequest = (): GalleryRevealRequest | null => currentRequest;

export const subscribeGalleryRevealRequests = (listener: () => void): (() => void) => {
  listeners.add(listener);

  return () => {
    listeners.delete(listener);
  };
};
