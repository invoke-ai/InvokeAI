import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { getApiErrorMessage } from '@platform/transport/http';

import type { ImageMapPoints } from './api';

import { fetchImageMapPoints } from './api';

/**
 * Read model for the semantic image map, shared by the widget body and any
 * future header/footer chrome. One store, one in-flight fetch; refreshes are
 * driven by user action now and by socket events in a later runtime.
 */

export interface ImageIndexCounts {
  total: number;
  embedded: number;
  pending: number;
  /** Given up on after repeated failures; excluded from `pending`. */
  failed: number;
}

export interface ImageMapSnapshot {
  data: ImageMapPoints | null;
  loadState: 'idle' | 'loading' | 'loaded' | 'error';
  error: string | null;
  /** Embedding-index progress; only ever pushed to admins by the backend. */
  indexCounts: ImageIndexCounts | null;
  /**
   * The plot canvas itself failed (WebGL unavailable). Distinct from `error`,
   * which means a fetch failed: with `error` the cached points are still worth
   * showing, whereas here there is nothing that can draw them, so the view must
   * stop mounting the plot and say so.
   */
  renderError: string | null;
}

const EMPTY_IMAGE_MAP_SNAPSHOT: ImageMapSnapshot = {
  data: null,
  error: null,
  indexCounts: null,
  loadState: 'idle',
  renderError: null,
};

export const imageMapStore = createExternalStore<ImageMapSnapshot>(EMPTY_IMAGE_MAP_SNAPSHOT);

let inflight: Promise<void> | null = null;
let rerunRequested = false;

// The projection is per-user server state: a login/logout must drop it before
// the next account's widgets can observe it.
registerAccountOwnedResource({
  clear: () => {
    inflight = null;
    rerunRequested = false;
    imageMapStore.setSnapshot(EMPTY_IMAGE_MAP_SNAPSHOT);
  },
  name: 'image-map',
});

export const refreshImageMapPoints = (): Promise<void> => {
  if (inflight) {
    // A refresh requested mid-flight (e.g. projection_ready arriving while
    // the fetch that triggered the recompute is still running) must not be
    // swallowed by the dedup: run once more when the current fetch settles.
    rerunRequested = true;

    return inflight;
  }

  const owner = captureAccountScope();

  if (imageMapStore.getSnapshot().loadState === 'idle') {
    imageMapStore.patchSnapshot({ loadState: 'loading' });
  }

  const refresh = fetchImageMapPoints()
    .then((data) => {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      // renderError is cleared too: a retry is the user's way out of a
      // transient WebGL failure, so a fresh point set must get a fresh attempt
      // at drawing rather than staying stuck on the previous canvas failure.
      imageMapStore.patchSnapshot({ data, error: null, loadState: 'loaded', renderError: null });
    })
    .catch((error: unknown) => {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      imageMapStore.patchSnapshot({
        error: getApiErrorMessage(error, 'Failed to load the image map'),
        loadState: 'error',
      });
    })
    .finally(() => {
      // Release only this refresh's claim: an account switch already reset
      // `inflight` and may have let a fresh refresh start.
      if (inflight === refresh) {
        inflight = null;
      }

      if (rerunRequested) {
        rerunRequested = false;
        void refreshImageMapPoints();
      }
    });

  inflight = refresh;

  return refresh;
};

export const ensureImageMapLoaded = (): void => {
  if (imageMapStore.getSnapshot().loadState === 'idle') {
    void refreshImageMapPoints();
  }
};
