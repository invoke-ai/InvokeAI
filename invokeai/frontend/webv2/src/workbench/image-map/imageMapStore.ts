import type { GalleryItemKind } from '@features/gallery/contracts';

import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { ApiError, getApiErrorMessage } from '@platform/transport/http';

import type { ImageMapClusterLabelInfo, ImageMapPoints } from './api';
import type { ImageIndexCounts } from './indexProgress';

import { fetchImageMapClusterLabels, fetchImageMapPoints, fetchImageMapStatus, requestImageMapRefresh } from './api';
import { hasProgressed } from './indexProgress';

export interface ImageMapSnapshot {
  data: ImageMapPoints | null;
  loadState: 'idle' | 'loading' | 'loaded' | 'error';
  error: string | null;
  /** Only ever pushed to admins by the backend. */
  indexCounts: ImageIndexCounts | null;
  /** Time of actual indexing progress, not count writes or gallery-total changes. */
  indexUpdatedAt: number | null;
  /** Null when unavailable (e.g. no text encoder). */
  clusterLabels: Record<string, ImageMapClusterLabelInfo> | null;
  /** Labels may lag refreshed points; precise consumers (hover cards) require equality with `data.visibleHash`. */
  clusterLabelsHash: string | null;
  /** Changing the strength renumbers clusters without moving `visibleHash`, so the hash alone cannot retire labels. */
  clusterLabelsEps: number | null;
  /** Unlike fetch errors, render errors unmount the plot; cached points stay useful after fetch errors. */
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
// Bumped by every socket status report, so a status fetch can tell whether one landed while it was in flight.
let indexEventSequence = 0;
// Only a response that began against the current read model may replace its availability.
let mapRevision = 0;

registerAccountOwnedResource({
  clear: () => {
    inflight = null;
    rerunRequested = false;
    // Per-project: a socket-driven refresh could beat the next account's widget pushing its own value.
    clusterEps = null;
    mediaKind = undefined;
    pointsMediaKind = undefined;
    labelsSequence += 1;
    statusInflight = null;
    indexEventSequence += 1;
    mapRevision += 1;
    imageMapStore.setSnapshot(EMPTY_IMAGE_MAP_SNAPSHOT);
  },
  name: 'image-map',
});

/** Null for the server's adaptive value. Pushed down from the widget because only this module sees every refresh. */
let clusterEps: number | null = null;

export const setClusterEps = (eps: number | null): void => {
  if (eps === clusterEps) {
    return;
  }

  clusterEps = eps;

  // The refresh below will not move `visibleHash`, so nothing else retires the old strength's labels.
  imageMapStore.patchSnapshot({ clusterLabels: null, clusterLabelsEps: null, clusterLabelsHash: null });

  // Nothing fetched yet: the first fetch carries the new value itself.
  if (imageMapStore.getSnapshot().loadState === 'idle') {
    return;
  }

  void refreshImageMapPoints();
};

/** Undefined for both kinds; pushed down by the data runtime so every refresh path fetches the same set. */
let mediaKind: GalleryItemKind | undefined;
/** Labels must be requested with the same kind as the published points. */
let pointsMediaKind: GalleryItemKind | undefined;

/** A change nobody is watching waits for the next `ensureImageMapLoaded` instead of fetching. */
let viewerCount = 0;

export const attachImageMapViewer = (): (() => void) => {
  viewerCount += 1;
  let attached = true;

  return () => {
    if (attached) {
      attached = false;
      viewerCount -= 1;
    }
  };
};

export const setImageMapMediaKind = (kind: GalleryItemKind | undefined): void => {
  if (kind === mediaKind) {
    return;
  }

  mediaKind = kind;
  labelsSequence += 1;

  // With no points response in flight the revision stays, so an unrelated status diagnosis still lands.
  if (inflight) {
    mapRevision += 1;
  }

  const { data, loadState } = imageMapStore.getSnapshot();

  if (data === null && loadState === 'idle') {
    return;
  }

  // The previous kind's points must not stay clickable: revealing an item the filter hides would reset the filter.
  imageMapStore.patchSnapshot({
    clusterLabels: null,
    clusterLabelsEps: null,
    clusterLabelsHash: null,
    data: null,
    error: null,
    loadState: 'idle',
    renderError: null,
  });

  if (viewerCount > 0) {
    void refreshImageMapPoints();
  }
};

export const refreshImageMapPoints = (): Promise<void> => {
  if (inflight) {
    // Coalesce mid-flight requests into one rerun instead of swallowing them through dedupe.
    rerunRequested = true;

    return inflight;
  }

  const owner = captureAccountScope();
  const revision = mapRevision;
  imageMapStore.patchSnapshot({ loadState: 'loading' });

  const requestMediaKind = mediaKind;
  const refresh = fetchImageMapPoints({
    ...(clusterEps !== null ? { eps: clusterEps } : {}),
    mediaKind: requestMediaKind,
  })
    .then((data) => {
      if (!isAccountScopeCurrent(owner) || revision !== mapRevision) {
        return;
      }

      mapRevision += 1;
      pointsMediaKind = requestMediaKind;
      imageMapStore.patchSnapshot({ data, error: null, loadState: 'loaded', renderError: null });
      refreshClusterLabels(data);
    })
    .catch((error: unknown) => {
      if (!isAccountScopeCurrent(owner) || revision !== mapRevision) {
        return;
      }

      imageMapStore.patchSnapshot({
        error: getApiErrorMessage(error, 'Failed to load the image map'),
        loadState: 'error',
      });
    })
    .finally(() => {
      // An account switch may already have reset `inflight` and started a fresh refresh.
      if (inflight === refresh) {
        inflight = null;
      }

      if (isAccountScopeCurrent(owner) && rerunRequested) {
        rerunRequested = false;
        void refreshImageMapPoints();
      }
    });

  inflight = refresh;

  return refresh;
};

/** Only user requests may bypass the refit cooldown; automatic callers here would bring the refit loop back. */
export const requestImageMapRecompute = (): Promise<void> => {
  void requestImageMapRefresh().catch(() => {
    // Best effort; the points fetch below reports connectivity errors.
  });

  return refreshImageMapPoints();
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

let clusterLabelsEnabled = true;

export const setClusterLabelsEnabled = (enabled: boolean): void => {
  if (enabled === clusterLabelsEnabled) {
    return;
  }

  clusterLabelsEnabled = enabled;

  if (!enabled) {
    labelsSequence += 1;
    imageMapStore.patchSnapshot({ clusterLabels: null, clusterLabelsEps: null, clusterLabelsHash: null });

    return;
  }

  const { data } = imageMapStore.getSnapshot();

  if (data) {
    refreshClusterLabels(data);
  }
};

/** Spans vocabulary builds and cold starts; then a points refresh or label toggle retries. */
const LABELS_RETRY_DELAYS_MS = [1_000, 2_000, 4_000, 8_000, 15_000, 30_000, 60_000, 60_000, 60_000];

/** 401, 403 and 422 settle at once. */
const isRetryableLabelsFailure = (error: unknown): boolean => {
  if (error instanceof ApiError) {
    return error.status === 409 || error.status >= 500;
  }

  return error instanceof TypeError;
};

const attemptClusterLabels = (sequence: number, data: ImageMapPoints, attempt: number): void => {
  // Reuse the points' eps, but still require visibleHash equality: cluster ids drift with the visible set.
  void fetchImageMapClusterLabels({
    ...(data.clusterEps !== null ? { eps: data.clusterEps } : {}),
    mediaKind: pointsMediaKind,
  })
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
      // Only the newest request may clear the labels, so a slow stale failure cannot wipe fresh ones.
      if (sequence !== labelsSequence) {
        return;
      }

      imageMapStore.patchSnapshot({ clusterLabels: null, clusterLabelsEps: null, clusterLabelsHash: null });

      if (attempt >= LABELS_RETRY_DELAYS_MS.length || !isRetryableLabelsFailure(error)) {
        return;
      }

      // Retry pending vocabulary: the next socket event may be far off.
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
    // A disabled index would 409 on every refresh.
    labelsSequence += 1;
    imageMapStore.patchSnapshot({ clusterLabels: null, clusterLabelsEps: null, clusterLabelsHash: null });

    return;
  }

  labelsSequence += 1;
  const sequence = labelsSequence;
  attemptClusterLabels(sequence, data, 0);
};

export const refetchClusterLabels = (): void => {
  const { data } = imageMapStore.getSnapshot();

  if (data) {
    refreshClusterLabels(data);
  }
};

/** Only socket reports bump the fencing sequence, so older status fetches cannot rewind event counts. */
export const recordImageIndexStatus = (
  counts: ImageIndexCounts,
  at: number = Date.now(),
  { measure = true }: { measure?: boolean } = {}
): void => {
  if (measure) {
    indexEventSequence += 1;
  }

  const { indexCounts: previous, indexUpdatedAt: previousAt } = imageMapStore.getSnapshot();
  // Restart the stall clock only for completed work.
  const updatedAt = hasProgressed(previous, counts) ? at : (previousAt ?? at);

  imageMapStore.patchSnapshot({ indexCounts: counts, indexUpdatedAt: updatedAt });
};

/** Recovers paused-worker counts and missed completion events; failure leaves socket reports authoritative. */
export const refreshImageIndexStatus = (): void => {
  // Deduplicate so an older response cannot land last and rewind counts.
  if (statusInflight) {
    return;
  }

  const owner = captureAccountScope();
  const sequence = indexEventSequence;
  const revision = mapRevision;

  const request: Promise<void> = fetchImageMapStatus(mediaKind)
    .then((status) => {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      if (revision === mapRevision) {
        const { data } = imageMapStore.getSnapshot();
        const missing = status.state === 'model_missing';
        // A replacement waiting for the retired encoder's work to drain: loaded points belong to a gone encoder.
        const draining = status.state === 'computing' && status.modelId === null;
        // A repeated diagnosis must not bump the revision: that would discard an in-flight Check again.
        const unchanged = missing
          ? data?.state === 'model_missing' && data.modelName === status.modelName
          : draining && data?.state === 'computing' && data.modelId === null;
        const replaced = data !== null && status.modelId !== null && data.modelId !== status.modelId;
        const recovered = data?.state === 'model_missing' && !missing && status.state !== 'disabled';

        if (((missing || draining) && !unchanged) || replaced || recovered) {
          mapRevision += 1;
          labelsSequence += 1;
          imageMapStore.patchSnapshot({
            clusterLabels: null,
            clusterLabelsEps: null,
            clusterLabelsHash: null,
            data: {
              clusterEps: null,
              modelId: status.modelId,
              modelName: status.modelName,
              pointCount: 0,
              points: [],
              stale: false,
              state: missing ? 'model_missing' : 'computing',
              updatedAt: null,
              visibleHash: null,
            },
            error: null,
            indexCounts: null,
            indexUpdatedAt: null,
            loadState: 'loaded',
            renderError: null,
          });
          if (missing || draining) {
            return;
          }
          void refreshImageMapPoints();
        }
      }

      // Counts aggregate every account and are omitted for non-admins; newer socket counts fence only them.
      if (status.index !== null && sequence === indexEventSequence) {
        recordImageIndexStatus(status.index, Date.now(), { measure: false });
      }
    })
    .catch(() => {
      // Optional detail; point loading reports its own errors.
    })
    .finally(() => {
      if (statusInflight === request) {
        statusInflight = null;
      }
    });

  statusInflight = request;
};

export const ensureImageMapLoaded = (): void => {
  const { data, loadState } = imageMapStore.getSnapshot();

  // The Model Manager unmounts this widget, so ask again on every reopen after a missing encoder.
  if (loadState === 'idle' || (data?.state === 'model_missing' && loadState !== 'loading')) {
    void refreshImageMapPoints();
  }

  // A paused worker may emit no event; store guards stop stale rewinds.
  refreshImageIndexStatus();
};

/** Missing encoders emit no index events, so poll while a view is mounted and visible. */
export const attachImageMapStatusPolling = (): (() => void) => {
  const owner = captureAccountScope();
  const poll = () => {
    if (document.visibilityState === 'visible') {
      refreshImageIndexStatus();
    }
  };
  const timer = setInterval(poll, 5_000);
  const dispose = () => {
    clearInterval(timer);
    document.removeEventListener('visibilitychange', poll);
    owner.signal.removeEventListener('abort', dispose);
  };

  document.addEventListener('visibilitychange', poll);

  owner.signal.addEventListener('abort', dispose, { once: true });

  return dispose;
};
