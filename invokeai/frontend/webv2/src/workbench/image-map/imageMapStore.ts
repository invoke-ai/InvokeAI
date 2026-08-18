import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { getApiErrorMessage } from '@platform/transport/http';

import type { ImageMapPoints } from './api';
import type { ImageIndexCounts, IndexRateSample } from './indexProgress';

import { fetchImageMapClusterLabels, fetchImageMapPoints, fetchImageMapStatus } from './api';
import { appendRateSample, estimateRate } from './indexProgress';

/**
 * Read model for the semantic image map, shared by the widget body and any
 * future header/footer chrome. One store, one in-flight fetch; refreshes are
 * driven by user action now and by socket events in a later runtime.
 */

export interface ImageMapSnapshot {
  data: ImageMapPoints | null;
  loadState: 'idle' | 'loading' | 'loaded' | 'error';
  error: string | null;
  /** Embedding-index progress; only ever pushed to admins by the backend. */
  indexCounts: ImageIndexCounts | null;
  /**
   * Images embedded per second, measured from consecutive status events. Null
   * until two of them have arrived far enough apart to measure, which is what
   * the progress UI shows as "estimating" rather than a made-up time.
   */
  indexRate: number | null;
  /** Cluster id -> automatic label; null when unavailable (e.g. no text encoder). */
  clusterLabels: Record<string, string> | null;
  /**
   * The plot canvas itself failed (WebGL unavailable). Distinct from `error`,
   * which means a fetch failed: with `error` the cached points are still worth
   * showing, whereas here there is nothing that can draw them, so the view must
   * stop mounting the plot and say so.
   */
  renderError: string | null;
}

const EMPTY_IMAGE_MAP_SNAPSHOT: ImageMapSnapshot = {
  clusterLabels: null,
  data: null,
  error: null,
  indexCounts: null,
  indexRate: null,
  loadState: 'idle',
  renderError: null,
};

export const imageMapStore = createExternalStore<ImageMapSnapshot>(EMPTY_IMAGE_MAP_SNAPSHOT);

let inflight: Promise<void> | null = null;
let rerunRequested = false;
let rateSamples: IndexRateSample[] = [];

// The projection is per-user server state: a login/logout must drop it before
// the next account's widgets can observe it.
registerAccountOwnedResource({
  clear: () => {
    inflight = null;
    rerunRequested = false;
    // Orphan any in-flight labels request so its completion (or failure)
    // cannot touch the next account's labels.
    labelsSequence += 1;
    rateSamples = [];
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
      refreshClusterLabels(data);
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

let labelsSequence = 0;

const areLabelMapsEqual = (left: Record<string, string> | null, right: Record<string, string>): boolean => {
  if (left === null) {
    return false;
  }

  const keys = Object.keys(right);

  return keys.length === Object.keys(left).length && keys.every((key) => left[key] === right[key]);
};

/**
 * Labels are decoration: fetched best-effort after the points land. Cluster
 * ids are only meaningful against the projection the points came from, so a
 * label response for a different projection — or one overtaken by a newer
 * request — is discarded rather than mislabeling clusters.
 */
/**
 * Whether the map is currently showing labels. Pushed down from the widget's
 * toggle so the request can be skipped outright: labelling costs the server a
 * DBSCAN pass over the visible set, an embedding gather and a 1675-way matmul
 * per cluster, and it is precisely the people with galleries large enough to
 * feel that who turn the labels off.
 */
let clusterLabelsEnabled = true;

export const setClusterLabelsEnabled = (enabled: boolean): void => {
  if (enabled === clusterLabelsEnabled) {
    return;
  }

  clusterLabelsEnabled = enabled;

  if (!enabled) {
    // Bump the sequence so a request already in flight cannot land after this.
    labelsSequence += 1;
    imageMapStore.patchSnapshot({ clusterLabels: null });

    return;
  }

  const { data } = imageMapStore.getSnapshot();

  if (data) {
    refreshClusterLabels(data);
  }
};

const refreshClusterLabels = (data: ImageMapPoints): void => {
  if (!clusterLabelsEnabled) {
    return;
  }

  if (data.state !== 'ready') {
    // Nothing to label; a disabled index would 409 on every refresh. Bump the
    // sequence so an in-flight labels response cannot repopulate the labels
    // this clears.
    labelsSequence += 1;
    imageMapStore.patchSnapshot({ clusterLabels: null });

    return;
  }

  labelsSequence += 1;
  const sequence = labelsSequence;
  // Pass the points' effective eps so both requests cluster with the same
  // value — the adaptive default is derived from the visible set, which can
  // drift between the two requests. Same eps alone does not pin cluster ids
  // under drift; the visibleHash comparison below discards those responses.
  void fetchImageMapClusterLabels(data.clusterEps !== null ? { eps: data.clusterEps } : undefined)
    .then((response) => {
      const current = imageMapStore.getSnapshot();

      if (
        sequence !== labelsSequence ||
        response.updatedAt !== current.data?.updatedAt ||
        response.visibleHash !== current.data?.visibleHash
      ) {
        return;
      }

      if (!areLabelMapsEqual(current.clusterLabels, response.labels)) {
        imageMapStore.patchSnapshot({ clusterLabels: response.labels });
      }
    })
    .catch(() => {
      // Same staleness rule as success: only the newest request may clear the
      // labels. A slow stale request failing after a newer one already set
      // fresh labels must not wipe them.
      if (sequence === labelsSequence) {
        imageMapStore.patchSnapshot({ clusterLabels: null });
      }
    });
};

/**
 * Fold one status report into the snapshot, keeping the throughput estimate
 * that drives the ETA up to date. Both the socket event and the seeding fetch
 * below land here, so the rate is measured over whatever mix of the two
 * arrives.
 */
export const recordImageIndexStatus = (counts: ImageIndexCounts, at: number = Date.now()): void => {
  if (counts.pending === 0) {
    // The run is over. Dropping the history means the next one measures its
    // own rate from scratch instead of inheriting the tail of this one, which
    // for a backfill followed by a single new image is wildly optimistic.
    rateSamples = [];
    imageMapStore.patchSnapshot({ indexCounts: counts, indexRate: null });

    return;
  }

  rateSamples = appendRateSample(rateSamples, { at, embedded: counts.embedded });
  imageMapStore.patchSnapshot({ indexCounts: counts, indexRate: estimateRate(rateSamples) });
};

/**
 * Seed the counts from the status endpoint. Status events only fire as batches
 * complete, so without this a panel opened while the worker is parked (it
 * waits out every generation) shows no progress at all until the queue moves
 * again. Best-effort: a failure just leaves the socket to fill them in.
 */
const seedImageIndexStatus = (): void => {
  const owner = captureAccountScope();

  void fetchImageMapStatus()
    .then((status) => {
      // Non-admins get no counts at all — the totals aggregate every user's
      // images — so `index` is null for them and there is nothing to seed.
      if (status.index === null || !isAccountScopeCurrent(owner)) {
        return;
      }

      // Never over an event: those are strictly newer, and a slow seed
      // landing after one would rewind the counts the user is watching.
      if (imageMapStore.getSnapshot().indexCounts === null) {
        recordImageIndexStatus(status.index);
      }
    })
    .catch(() => {
      // Progress detail is optional chrome; the map itself reports its own
      // load failures.
    });
};

export const ensureImageMapLoaded = (): void => {
  if (imageMapStore.getSnapshot().loadState === 'idle') {
    void refreshImageMapPoints();
    seedImageIndexStatus();
  }
};
