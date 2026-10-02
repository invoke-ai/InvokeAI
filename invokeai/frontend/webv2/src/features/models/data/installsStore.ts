import type { ModelInstallJob, ModelInstallStatus } from '@features/models/core/types';

import { createLogger } from '@platform/logging/logger';
import {
  type AccountScope,
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { createExternalStore, createKeyedTransientStore } from '@platform/state/externalStore';
import { createTrailingSingleFlight } from '@platform/state/singleFlight';
import { getApiErrorMessage } from '@platform/transport/http';

import { listModelInstalls } from './api';
import { refreshModels } from './modelsStore';
import { refreshStartersIfLoaded } from './startersStore';

/**
 * REST owns install jobs; lifecycle events refresh them. High-frequency progress bypasses the list and subscribes
 * per job to limit renders.
 */

export interface InstallsSnapshot {
  jobs: ModelInstallJob[];
  status: 'idle' | 'loading' | 'loaded' | 'error';
  error: string | null;
  /** Settled jobs hidden locally; the backend only prunes all finished jobs at once. */
  dismissedJobIds: ReadonlySet<number>;
}

export interface InstallDownloadProgress {
  bytes: number;
  totalBytes: number;
  /** Smoothed transfer rate; null until two spaced samples exist. */
  bytesPerSecond: number | null;
}

/** A just-settled install, surfaced so the UI can toast success/failure. */
export interface InstallOutcome {
  id: number;
  jobId: number;
  kind: 'completed' | 'error' | 'cancelled';
  modelName: string | null;
  source: string;
  error: string | null;
}

/**
 * Normalize typed and socket source labels identically for matching; colocate here to keep taxonomy out of eager
 * chunks.
 */
export const getInstallSourceLabel = (source: unknown): string => {
  if (typeof source === 'string') {
    return source;
  }

  if (source && typeof source === 'object') {
    const record = source as Record<string, unknown>;

    for (const field of ['repo_id', 'url', 'path'] as const) {
      const value = record[field];

      if (typeof value === 'string') {
        return value;
      }
    }
  }

  return 'model';
};

const REFRESH_COALESCE_MS = 250;
// Bound display history with room for completion bursts between toast flushes.
const OUTCOME_LIMIT = 64;
const RATE_SAMPLE_MS = 500;
const RATE_SMOOTHING = 0.3;

const EMPTY_DISMISSED_IDS: ReadonlySet<number> = new Set();
const EMPTY_INSTALLS_SNAPSHOT: InstallsSnapshot = {
  dismissedJobIds: EMPTY_DISMISSED_IDS,
  error: null,
  jobs: [],
  status: 'idle',
};
const EMPTY_INSTALL_OUTCOMES: { outcomes: InstallOutcome[] } = { outcomes: [] };

const store = createExternalStore<InstallsSnapshot>(EMPTY_INSTALLS_SNAPSHOT);
const outcomesStore = createExternalStore<{ outcomes: InstallOutcome[] }>(EMPTY_INSTALL_OUTCOMES);
let nextOutcomeId = 1;

const progressByJobId = createKeyedTransientStore<number, InstallDownloadProgress>();
const rateSamplesByJobId = new Map<number, { bytes: number; at: number }>();

const refreshFlight = createTrailingSingleFlight();
let refreshTimer: ReturnType<typeof setTimeout> | null = null;
let catalogRefreshTimer: ReturnType<typeof setTimeout> | null = null;

registerAccountOwnedResource({
  clear: () => {
    if (refreshTimer !== null) {
      clearTimeout(refreshTimer);
      refreshTimer = null;
    }

    if (catalogRefreshTimer !== null) {
      clearTimeout(catalogRefreshTimer);
      catalogRefreshTimer = null;
    }

    refreshFlight.reset();
    nextOutcomeId = 1;
    progressByJobId.clear();
    rateSamplesByJobId.clear();
    outcomesStore.setSnapshot(EMPTY_INSTALL_OUTCOMES);
    store.setSnapshot(EMPTY_INSTALLS_SNAPSHOT);
  },
  name: 'model-installs',
});

export const refreshInstalls = (owner: AccountScope = captureAccountScope()): Promise<void> =>
  refreshFlight.run(() => {
    store.patchSnapshot({ status: store.getSnapshot().status === 'loaded' ? 'loaded' : 'loading' });

    return listModelInstalls(owner.signal)
      .then((jobs) => {
        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        const activeJobIds = new Set(jobs.map((job) => job.id));

        for (const [jobId] of progressByJobId.entries()) {
          if (!activeJobIds.has(jobId)) {
            progressByJobId.delete(jobId);
            rateSamplesByJobId.delete(jobId);
          }
        }

        const { dismissedJobIds } = store.getSnapshot();
        const retainedDismissed = [...dismissedJobIds].filter((jobId) => activeJobIds.has(jobId));

        store.patchSnapshot({
          dismissedJobIds:
            retainedDismissed.length === dismissedJobIds.size ? dismissedJobIds : new Set(retainedDismissed),
          error: null,
          jobs,
          status: 'loaded',
        });
      })
      .catch((error: unknown) => {
        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        installLogger.warn({
          error,
          message: 'Failed to load the install queue',
          name: 'models.install-queue-load-failed',
        });
        store.patchSnapshot({
          error: getApiErrorMessage(error, 'Failed to load install queue.'),
          status: store.getSnapshot().jobs.length > 0 ? 'loaded' : 'error',
        });
      });
  });

/** Fetch on first use or retry after an error, so one failed load never sticks. */
export const ensureInstallsLoaded = (): void => {
  const { status } = store.getSnapshot();

  if (status === 'idle' || status === 'error') {
    void refreshInstalls();
  }
};

const scheduleRefresh = (): void => {
  if (refreshTimer !== null) {
    return;
  }

  refreshTimer = setTimeout(() => {
    refreshTimer = null;
    void refreshInstalls();
  }, REFRESH_COALESCE_MS);
};

/** Coalesce install-completion bursts into one library/starter revalidation. */
const scheduleCatalogRefresh = (): void => {
  if (catalogRefreshTimer !== null) {
    return;
  }

  catalogRefreshTimer = setTimeout(() => {
    catalogRefreshTimer = null;
    void refreshModels();
    refreshStartersIfLoaded();
  }, REFRESH_COALESCE_MS);
};

/** Optimistically replace one job (e.g. after pause/resume API calls). */
export const replaceInstallJob = (job: ModelInstallJob): void => {
  store.patchSnapshot({
    jobs: store.getSnapshot().jobs.map((existing) => (existing.id === job.id ? job : existing)),
  });
};

/** Hide a settled job locally until the backend stops listing it. */
export const dismissInstallJob = (jobId: number): void => {
  const { dismissedJobIds } = store.getSnapshot();

  if (!dismissedJobIds.has(jobId)) {
    store.patchSnapshot({ dismissedJobIds: new Set([...dismissedJobIds, jobId]) });
  }
};

/** Optimistically add a freshly created job so the queue updates instantly. */
export const addInstallJob = (job: ModelInstallJob): void => {
  if (store.getSnapshot().jobs.some((existing) => existing.id === job.id)) {
    replaceInstallJob(job);
    return;
  }

  store.patchSnapshot({ jobs: [job, ...store.getSnapshot().jobs], status: 'loaded' });
};

const installLogger = createLogger({ area: 'install', namespace: 'models' });

const recordOutcome = (outcome: Omit<InstallOutcome, 'id'>): void => {
  const context = { jobId: outcome.jobId, modelName: outcome.modelName, source: outcome.source };

  if (outcome.kind === 'error') {
    installLogger.error({
      context: { ...context, reason: outcome.error },
      message: `Model install failed: ${outcome.source}`,
      name: 'models.install-failed',
    });
  } else {
    installLogger.info({
      context,
      message:
        outcome.kind === 'completed'
          ? `Model installed: ${outcome.source}`
          : `Model install cancelled: ${outcome.source}`,
      name: outcome.kind === 'completed' ? 'models.install-completed' : 'models.install-cancelled',
    });
  }

  outcomesStore.patchSnapshot({
    outcomes: [{ ...outcome, id: nextOutcomeId }, ...outcomesStore.getSnapshot().outcomes].slice(0, OUTCOME_LIMIT),
  });
  nextOutcomeId += 1;
};

/** Exponential smoothing over spaced samples keeps the rate readable through bursty progress ticks. */
const sampleTransferRate = (jobId: number, bytes: number): number | null => {
  const now = Date.now();
  const previous = rateSamplesByJobId.get(jobId);
  const current = progressByJobId.get(jobId)?.bytesPerSecond ?? null;

  if (!previous) {
    rateSamplesByJobId.set(jobId, { at: now, bytes });
    return current;
  }

  const elapsedMs = now - previous.at;

  if (elapsedMs < RATE_SAMPLE_MS) {
    return current;
  }

  rateSamplesByJobId.set(jobId, { at: now, bytes });

  const instant = Math.max(0, ((bytes - previous.bytes) * 1000) / elapsedMs);

  return current === null ? instant : current * (1 - RATE_SMOOTHING) + instant * RATE_SMOOTHING;
};

interface ModelInstallSocketPayload {
  id: number;
  bytes?: number;
  total_bytes?: number;
  source?: unknown;
  error?: string | null;
  error_type?: string | null;
  config?: { name?: string } | null;
}

export const MODEL_INSTALL_SOCKET_EVENTS = [
  'model_install_started',
  'model_install_download_started',
  'model_install_download_progress',
  'model_install_downloads_complete',
  'model_install_complete',
  'model_install_error',
  'model_install_cancelled',
] as const;

export type ModelInstallSocketEvent = (typeof MODEL_INSTALL_SOCKET_EVENTS)[number];

/** Socket sink — wired into the backend socket by the queue coordinator. */
export const handleModelInstallSocketEvent = (
  event: ModelInstallSocketEvent,
  payload: unknown,
  owner: AccountScope = captureAccountScope()
): void => {
  if (!isAccountScopeCurrent(owner)) {
    return;
  }

  const data = payload as ModelInstallSocketPayload;

  if (typeof data?.id !== 'number') {
    return;
  }

  if (event === 'model_install_download_progress') {
    const bytes = data.bytes ?? 0;

    progressByJobId.set(data.id, {
      bytes,
      bytesPerSecond: sampleTransferRate(data.id, bytes),
      totalBytes: data.total_bytes ?? 0,
    });

    const job = store.getSnapshot().jobs.find((candidate) => candidate.id === data.id);

    if (!job) {
      // The first progress tick may arrive for a job created in another
      // client; make sure the row exists without refetching on every tick.
      scheduleRefresh();
    } else if (job.status === 'waiting') {
      // Bytes are flowing, so the REST snapshot's `waiting` is stale. Patch
      // locally so download controls (pause/cancel) appear immediately.
      replaceInstallJob({ ...job, status: 'downloading' });
    }

    return;
  }

  if (event === 'model_install_complete' || event === 'model_install_error' || event === 'model_install_cancelled') {
    // Retain settled jobs until cleared, but release their inactive byte-progress state.
    progressByJobId.delete(data.id);
    rateSamplesByJobId.delete(data.id);
  }

  if (event === 'model_install_complete') {
    recordOutcome({
      error: null,
      jobId: data.id,
      kind: 'completed',
      modelName: data.config?.name ?? null,
      source: getInstallSourceLabel(data.source),
    });
    scheduleCatalogRefresh();
  } else if (event === 'model_install_error') {
    recordOutcome({
      error: data.error ?? data.error_type ?? 'Unknown install error.',
      jobId: data.id,
      kind: 'error',
      modelName: null,
      source: getInstallSourceLabel(data.source),
    });
  } else if (event === 'model_install_cancelled') {
    recordOutcome({
      error: null,
      jobId: data.id,
      kind: 'cancelled',
      modelName: null,
      source: getInstallSourceLabel(data.source),
    });
  }

  scheduleRefresh();
};

const ACTIVE_STATUSES: ModelInstallStatus[] = ['waiting', 'downloading', 'downloads_done', 'running'];

export const isActiveInstallStatus = (status: ModelInstallStatus): boolean => ACTIVE_STATUSES.includes(status);

export const useInstallsSelector = store.useSelector;

export const getInstallsSnapshot = (): InstallsSnapshot => store.getSnapshot();

/** Cache active source strings by jobs-array identity for installing affordances. */
const areSetsEqual = <Value>(left: ReadonlySet<Value>, right: ReadonlySet<Value>): boolean =>
  left.size === right.size && Array.from(left).every((value) => right.has(value));

const getActiveInstallSources = (jobs: ModelInstallJob[]): ReadonlySet<string> =>
  new Set(
    jobs
      .filter((job) => isActiveInstallStatus(job.status) || job.status === 'paused')
      .map((job) => getInstallSourceLabel(job.source))
  );

export const useActiveInstallSources = (): ReadonlySet<string> =>
  store.useSelector((snapshot) => getActiveInstallSources(snapshot.jobs), areSetsEqual);

export const useInstallProgress = (jobId: number): InstallDownloadProgress | null =>
  progressByJobId.useValue(jobId) ?? null;

export const getInstallProgress = (jobId: number): InstallDownloadProgress | null => progressByJobId.get(jobId) ?? null;

export const useInstallOutcomes = (): InstallOutcome[] => outcomesStore.useSelector((snapshot) => snapshot.outcomes);

export const getInstallOutcomes = (): InstallOutcome[] => outcomesStore.getSnapshot().outcomes;
