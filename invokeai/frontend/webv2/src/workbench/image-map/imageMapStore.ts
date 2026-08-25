import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { getApiErrorMessage } from '@platform/transport/http';

import type { ImageMapClusterLabelInfo, ImageMapPoints } from './api';
import type { ImageIndexCounts } from './indexProgress';

import { fetchImageMapClusterLabels, fetchImageMapPoints, fetchImageMapStatus } from './api';
import { hasProgressed } from './indexProgress';

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
   * When the index last actually moved, so the UI can say how long it has
   * stood still. Not when the counts were last written: `total` shifts
   * whenever anyone saves or deletes an image, and a generation running while
   * the indexer waits for it must not read as the index progressing. Null
   * whenever there are no counts.
   */
  indexUpdatedAt: number | null;
  /** Cluster id -> automatic label info; null when unavailable (e.g. no text encoder). */
  clusterLabels: Record<string, ImageMapClusterLabelInfo> | null;
  /**
   * The visible-set fingerprint `clusterLabels` were computed over, so a
   * consumer can tell whether they still describe the drawn clustering.
   *
   * Labels lag their points by a request: a refresh replaces `data` while the
   * previous clustering's labels are still in the store, and DBSCAN can
   * renumber every cluster between the two. Annotations have always accepted
   * that lag (clearing them on each refresh would blink every label off and
   * back on), but a consumer making a stronger promise — the hover card names
   * one specific cluster — compares this against `data.visibleHash` and shows
   * nothing rather than another cluster's tags.
   */
  clusterLabelsHash: string | null;
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
    // Orphan any in-flight labels request so its completion (or failure)
    // cannot touch the next account's labels.
    labelsSequence += 1;
    statusInflight = null;
    indexEventSequence += 1;
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
  imageMapStore.patchSnapshot({ loadState: 'loading' });

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
    imageMapStore.patchSnapshot({ clusterLabels: null, clusterLabelsHash: null });

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
    imageMapStore.patchSnapshot({ clusterLabels: null, clusterLabelsHash: null });

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
        imageMapStore.patchSnapshot({ clusterLabels: response.labels, clusterLabelsHash: response.visibleHash });
      }
    })
    .catch(() => {
      // Same staleness rule as success: only the newest request may clear the
      // labels. A slow stale request failing after a newer one already set
      // fresh labels must not wipe them.
      if (sequence === labelsSequence) {
        imageMapStore.patchSnapshot({ clusterLabels: null, clusterLabelsHash: null });
      }
    });
};

/**
 * Re-fetch labels for the currently loaded points. For callers outside the
 * points-refresh flow whose action changes what the labels *say* without
 * moving a single point — today that is a supplementary-vocabulary edit, once
 * the server reports its embedding rebuild finished. A no-op while labels are
 * toggled off or no points are loaded.
 */
export const refetchClusterLabels = (): void => {
  const { data } = imageMapStore.getSnapshot();

  if (data) {
    refreshClusterLabels(data);
  }
};

/**
 * Fold one status report into the snapshot.
 *
 * `measure` marks a socket-delivered report. Only those bump the sequence a
 * status fetch checks itself against, so a fetch cannot rewind counts an event
 * delivered while it was in flight.
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
  // Only real progress restarts the clock. A re-read that returns the same
  // work done — which is every re-read while the indexer waits out a
  // generation, since `total` moves as that generation saves images — would
  // otherwise hide exactly the stall the note exists to report.
  const updatedAt = hasProgressed(previous, counts) ? at : (previousAt ?? at);

  imageMapStore.patchSnapshot({ indexCounts: counts, indexUpdatedAt: updatedAt });
};

/**
 * Pull the counts from the status endpoint.
 *
 * Status events only fire as batches complete, so without this a panel opened
 * while the worker is parked (it waits out every generation) shows no progress
 * at all until the queue moves again — and a client that was disconnected when
 * the run's final `pending === 0` event went out would otherwise show a
 * finished backfill as still running until the page is reloaded. Called on the
 * first load, on every socket reconnect, and from the panel's own retry.
 *
 * Best-effort: a failure just leaves the socket to fill the counts in.
 */
export const refreshImageIndexStatus = (): void => {
  // One at a time, like the point set above. Every widget mount, every retry
  // click and every reconnect calls this, and concurrent requests resolve in
  // no particular order: without the dedup the *oldest* response can land last
  // and rewind the counts to what the server said before the newer one asked.
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
      // Progress detail is optional chrome; the map itself reports its own
      // load failures.
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

  // Unguarded, unlike the point set: the counts are one cheap request, and the
  // widget may well be reopened long after the first load — mid-backfill, with
  // the worker parked and no event due — which is exactly the case the panel
  // exists for. The store's own guards stop it rewinding anything newer.
  refreshImageIndexStatus();
};
