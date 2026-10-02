import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { ApiError, getApiErrorMessage } from '@platform/transport/http';

import type { ImageMapClusterLabelInfo, ImageMapPoints } from './api';
import type { ImageIndexCounts } from './indexProgress';

import { fetchImageMapClusterLabels, fetchImageMapPoints, fetchImageMapStatus } from './api';
import { hasProgressed } from './indexProgress';

/** Shared image-map read model with one in-flight fetch and refreshes from user actions and socket events. */

export interface ImageMapSnapshot {
  data: ImageMapPoints | null;
  loadState: 'idle' | 'loading' | 'loaded' | 'error';
  error: string | null;
  /** Embedding-index progress; only ever pushed to admins by the backend. */
  indexCounts: ImageIndexCounts | null;
  /** Time of actual indexing progress, not count writes or gallery-total changes. Null without counts. */
  indexUpdatedAt: number | null;
  /** Cluster id -> automatic label info; null when unavailable (e.g. no text encoder). */
  clusterLabels: Record<string, ImageMapClusterLabelInfo> | null;
  /**
   * Fingerprint of the visible set used for cluster labels. Labels may lag refreshed points whose cluster ids
   * changed; precise consumers such as hover cards require equality with `data.visibleHash`.
   */
  clusterLabelsHash: string | null;
  /**
   * The eps `clusterLabels` were clustered at. The visible-set hash alone
   * stopped being enough once the strength became adjustable: changing it
   * renumbers every cluster while the projection and the visible set — and
   * so `visibleHash` — stay exactly the same, which would let the previous
   * strength's tags be shown against the new clustering for a round trip.
   */
  clusterLabelsEps: number | null;
  /**
   * Plot/WebGL failure differs from fetch failure: cached points remain useful after fetch errors, but render
   * errors require unmounting the plot and reporting inability to draw.
   */
  renderError: string | null;
}

const EMPTY_IMAGE_MAP_SNAPSHOT: ImageMapSnapshot = {
  clusterLabels: null,
  clusterLabelsEps: null,
  clusterLabelsHash: null,
  data: null,
  error: null,
  indexCounts: null,
  indexUpdatedAt: null,
  loadState: 'idle',
  renderError: null,
};

export const imageMapStore = createExternalStore<ImageMapSnapshot>(EMPTY_IMAGE_MAP_SNAPSHOT);

let inflight: Promise<void> | null = null;
let rerunRequested = false;
let statusInflight: Promise<void> | null = null;
// Bumped by every socket-delivered status report, so a status fetch that
// started earlier can tell whether one landed while it was in flight.
let indexEventSequence = 0;

// The projection is per-user server state: a login/logout must drop it before
// the next account's widgets can observe it.
registerAccountOwnedResource({
  clear: () => {
    inflight = null;
    rerunRequested = false;
    // Per-project widget state, so it must not survive into the next
    // account: the widget pushes the new one down when it mounts, but a
    // socket-driven refresh could beat it there.
    clusterEps = null;
    // Retire label requests so old-account completion/failure cannot update the new account.
    labelsSequence += 1;
    statusInflight = null;
    indexEventSequence += 1;
    imageMapStore.setSnapshot(EMPTY_IMAGE_MAP_SNAPSHOT);
  },
  name: 'image-map',
});

/**
 * The eps the clustering-strength control has been set to, or null for the
 * server's adaptive value. Pushed down from the widget rather than passed by
 * callers: every refresh — socket-driven, manual, or the first load — has to
 * cluster at the strength the user chose, and only this module sees them all.
 */
let clusterEps: number | null = null;

export const setClusterEps = (eps: number | null): void => {
  if (eps === clusterEps) {
    return;
  }

  clusterEps = eps;

  // The labels in the store describe the previous strength's cluster ids, and
  // the refresh below will not move `visibleHash`, so nothing else retires
  // them.
  imageMapStore.patchSnapshot({ clusterLabels: null, clusterLabelsEps: null, clusterLabelsHash: null });

  // Nothing has been fetched yet, so the first fetch will carry the new value
  // on its own; refreshing here would race it for no gain.
  if (imageMapStore.getSnapshot().loadState === 'idle') {
    return;
  }

  void refreshImageMapPoints();
};

export const refreshImageMapPoints = (): Promise<void> => {
  if (inflight) {
    // Coalesce mid-flight refresh requests into one rerun after settling instead of swallowing them through
    // dedupe.
    rerunRequested = true;

    return inflight;
  }

  const owner = captureAccountScope();
  imageMapStore.patchSnapshot({ loadState: 'loading' });

  const refresh = fetchImageMapPoints(clusterEps !== null ? { eps: clusterEps } : undefined)
    .then((data) => {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      // Clear renderError on retry so fresh points receive a new WebGL draw attempt.
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

const areLabelMapsEqual = (
  left: Record<string, ImageMapClusterLabelInfo> | null,
  right: Record<string, ImageMapClusterLabelInfo>
): boolean => {
  if (left === null) {
    return false;
  }

  const keys = Object.keys(right);

  return (
    keys.length === Object.keys(left).length &&
    keys.every((key) => {
      const before = left[key];
      const after = right[key];

      return (
        before !== undefined &&
        before.label === after.label &&
        before.alternates.length === after.alternates.length &&
        before.alternates.every((alternate, index) => alternate === after.alternates[index])
      );
    })
  );
};

/** Best-effort labels must match point projection and current request sequence before publication. */
/** Skip labels entirely when hidden, avoiding server clustering, embedding gathers and vocabulary scoring. */
let clusterLabelsEnabled = true;

export const setClusterLabelsEnabled = (enabled: boolean): void => {
  if (enabled === clusterLabelsEnabled) {
    return;
  }

  clusterLabelsEnabled = enabled;

  if (!enabled) {
    // Bump the sequence so a request already in flight cannot land after this.
    labelsSequence += 1;
    imageMapStore.patchSnapshot({ clusterLabels: null, clusterLabelsEps: null, clusterLabelsHash: null });

    return;
  }

  const { data } = imageMapStore.getSnapshot();

  if (data) {
    refreshClusterLabels(data);
  }
};

/**
 * Bounded label backoff covers pending vocabulary builds, including cold starts. After exhaustion, the next points
 * refresh or label toggle retries.
 */
const LABELS_RETRY_DELAYS_MS = [1_000, 2_000, 4_000, 8_000, 15_000, 30_000, 60_000, 60_000, 60_000];

/**
 * Retry 409 vocabulary/text-tower unavailability, 5xx and fetch TypeErrors. Auth/contract failures 401, 403 and
 * 422 settle immediately.
 */
const isRetryableLabelsFailure = (error: unknown): boolean => {
  if (error instanceof ApiError) {
    return error.status === 409 || error.status >= 500;
  }

  return error instanceof TypeError;
};

/** Every deferred retry checks sequence; newer requests, toggles and account switches retire prior callbacks. */
const attemptClusterLabels = (sequence: number, data: ImageMapPoints, attempt: number): void => {
  // Reuse the points' effective eps, but still require visibleHash equality because cluster ids can drift with the
  // visible set.
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
        imageMapStore.patchSnapshot({
          clusterLabels: response.labels,
          clusterLabelsEps: data.clusterEps,
          clusterLabelsHash: response.visibleHash,
        });
      }
    })
    .catch((error: unknown) => {
      // Same staleness rule as success: only the newest request may clear the
      // labels. A slow stale request failing after a newer one already set
      // fresh labels must not wipe them.
      if (sequence !== labelsSequence) {
        return;
      }

      imageMapStore.patchSnapshot({ clusterLabels: null, clusterLabelsEps: null, clusterLabelsHash: null });

      if (attempt >= LABELS_RETRY_DELAYS_MS.length || !isRetryableLabelsFailure(error)) {
        return;
      }

      // Retry pending vocabulary responses because loaded-widget activation may fetch nothing and the next socket
      // event may be far away.
      const delay = LABELS_RETRY_DELAYS_MS[attempt];
      setTimeout(() => {
        if (sequence === labelsSequence) {
          attemptClusterLabels(sequence, data, attempt + 1);
        }
      }, delay);
    });
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
    imageMapStore.patchSnapshot({ clusterLabels: null, clusterLabelsEps: null, clusterLabelsHash: null });

    return;
  }

  labelsSequence += 1;
  const sequence = labelsSequence;
  attemptClusterLabels(sequence, data, 0);
};

/** Refresh labels without moving points after vocabulary rebuild; no-op when labels are hidden or points absent. */
export const refetchClusterLabels = (): void => {
  const { data } = imageMapStore.getSnapshot();

  if (data) {
    refreshClusterLabels(data);
  }
};

/**
 * Socket reports alone bump the fetch-fencing sequence, preventing older status fetches from rewinding event
 * counts.
 */
export const recordImageIndexStatus = (
  counts: ImageIndexCounts,
  at: number = Date.now(),
  { measure = true }: { measure?: boolean } = {}
): void => {
  if (measure) {
    indexEventSequence += 1;
  }

  const { indexCounts: previous, indexUpdatedAt: previousAt } = imageMapStore.getSnapshot();
  // Restart the stall clock only for actual completed work, not repeated counts or generation-driven total
  // changes.
  const updatedAt = hasProgressed(previous, counts) ? at : (previousAt ?? at);

  imageMapStore.patchSnapshot({ indexCounts: counts, indexUpdatedAt: updatedAt });
};

/**
 * Best-effort status fetch recovers paused-worker counts and missed completion events on load/reconnect/retry.
 * Failure leaves socket reports authoritative.
 */
export const refreshImageIndexStatus = (): void => {
  // Deduplicate status requests so an older response cannot land last and rewind counts.
  if (statusInflight) {
    return;
  }

  const owner = captureAccountScope();
  // Events that land while this is in flight are strictly newer than what it
  // will return, and must not be rewound by it.
  const sequence = indexEventSequence;

  const request: Promise<void> = fetchImageMapStatus()
    .then((status) => {
      // Non-admins get no counts at all — the totals aggregate every user's
      // images — so `index` is null for them and there is nothing to record.
      if (status.index === null || !isAccountScopeCurrent(owner) || sequence !== indexEventSequence) {
        return;
      }

      recordImageIndexStatus(status.index, Date.now(), { measure: false });
    })
    .catch(() => {
      // Progress failure is optional detail; point loading reports its own errors.
    })
    .finally(() => {
      // Release only this request's claim: an account switch already cleared
      // it and may have let a fresh one start.
      if (statusInflight === request) {
        statusInflight = null;
      }
    });

  statusInflight = request;
};

export const ensureImageMapLoaded = (): void => {
  if (imageMapStore.getSnapshot().loadState === 'idle') {
    void refreshImageMapPoints();
  }

  // Always refresh cheap status counts on reopen; a paused worker may emit no event. Store guards prevent stale
  // rewinds.
  refreshImageIndexStatus();
};
