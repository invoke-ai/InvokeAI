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
/**
 * Latest locally known status per job, from a socket event or an action result, stamped with a sequence. A list
 * request that started before the stamp may predate that status, so it must not overwrite it; the first list request
 * started after it is authoritative and clears the fence. Starting earlier does not prove the backend served it
 * earlier, though, and some transitions emit no event (cancelling a download, pruning settled jobs). So a final list
 * status always wins, and a stale response may drop a job whose fence is settled, since pruning removes only those,
 * but only when this client knew the job before the request started; otherwise the backend may have served it before
 * the job existed.
 */
const statusFenceByJobId = new Map<number, { sequence: number; status: ModelInstallStatus }>();
/** Sequence at which this client first saw each job, from a list response, a socket event, or a creation result. */
const firstSeenSequenceByJobId = new Map<number, number>();
let nextStatusFenceSequence = 1;
/** No backend transition leaves these; `error` is excluded because a failed install can be restarted. */
const FINAL_STATUSES: ReadonlySet<ModelInstallStatus> = new Set(['completed', 'cancelled']);
const SETTLED_STATUSES: ReadonlySet<ModelInstallStatus> = new Set(['completed', 'error', 'cancelled']);
const CANCELLATION_ACTIVE_STATUSES: ReadonlySet<ModelInstallStatus> = new Set([
  'waiting',
  'downloading',
  'downloads_done',
  'running',
  'paused',
]);

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
    statusFenceByJobId.clear();
    firstSeenSequenceByJobId.clear();
    nextStatusFenceSequence = 1;
    outcomesStore.setSnapshot(EMPTY_INSTALL_OUTCOMES);
    store.setSnapshot(EMPTY_INSTALLS_SNAPSHOT);
  },
  name: 'model-installs',
});

export const refreshInstalls = (owner: AccountScope = captureAccountScope()): Promise<void> =>
  refreshFlight.run(() => {
    const startSequence = nextStatusFenceSequence;
    store.patchSnapshot({ status: store.getSnapshot().status === 'loaded' ? 'loaded' : 'loading' });

    return listModelInstalls(owner.signal)
      .then((jobs) => {
        if (!isAccountScopeCurrent(owner)) {
          return;
        }

        const activeJobIds = new Set(jobs.map((job) => job.id));
        const reconciledJobs: ModelInstallJob[] = jobs.map((job) => {
          const fence = statusFenceByJobId.get(job.id);

          if (fence === undefined) {
            return job;
          }

          if (fence.status === 'cancelled' && CANCELLATION_ACTIVE_STATUSES.has(job.status)) {
            // DELETE can succeed before the worker observes cancellation. Keep the optimistic result until REST
            // reports a terminal state; otherwise a successful refresh can revive the row if its socket was missed.
            const current = store.getSnapshot().jobs.find((candidate) => candidate.id === job.id) ?? job;
            return current.status === 'cancelled' ? current : { ...current, status: 'cancelled' };
          }

          if (fence.sequence >= startSequence && !FINAL_STATUSES.has(job.status)) {
            // This request predates the fenced status, so its status is stale.
            const current = store.getSnapshot().jobs.find((candidate) => candidate.id === job.id) ?? job;
            return current.status === fence.status ? current : { ...current, status: fence.status };
          }

          // A request started after the fence, or a final status, is authoritative.
          statusFenceByJobId.delete(job.id);
          return job;
        });

        for (const [jobId, fence] of statusFenceByJobId) {
          if (activeJobIds.has(jobId)) {
            continue;
          }

          const knownBeforeStart = (firstSeenSequenceByJobId.get(jobId) ?? Infinity) < startSequence;

          if (fence.sequence >= startSequence && !(SETTLED_STATUSES.has(fence.status) && knownBeforeStart)) {
            // The job may have been created after the backend served this request.
            const locallyKnownJob = store.getSnapshot().jobs.find((job) => job.id === jobId);

            if (locallyKnownJob) {
              reconciledJobs.push({ ...locallyKnownJob, status: fence.status });
              activeJobIds.add(jobId);
            }
          } else {
            statusFenceByJobId.delete(jobId);
          }
        }

        for (const jobId of activeJobIds) {
          noteJobSeen(jobId);
        }

        for (const jobId of firstSeenSequenceByJobId.keys()) {
          if (!activeJobIds.has(jobId) && !statusFenceByJobId.has(jobId)) {
            firstSeenSequenceByJobId.delete(jobId);
          }
        }

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
          jobs: reconciledJobs,
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

const noteJobSeen = (jobId: number): void => {
  if (!firstSeenSequenceByJobId.has(jobId)) {
    firstSeenSequenceByJobId.set(jobId, nextStatusFenceSequence++);
  }
};

const fenceJobStatus = (jobId: number, status: ModelInstallStatus): void => {
  statusFenceByJobId.set(jobId, { sequence: nextStatusFenceSequence++, status });
};

const patchJob = (job: ModelInstallJob): void => {
  store.patchSnapshot({
    jobs: store.getSnapshot().jobs.map((existing) => (existing.id === job.id ? job : existing)),
  });
};

const projectJobStatus = (jobId: number, status: ModelInstallStatus): void => {
  if (statusFenceByJobId.get(jobId)?.status === 'cancelled' && CANCELLATION_ACTIVE_STATUSES.has(status)) {
    return;
  }

  fenceJobStatus(jobId, status);

  const job = store.getSnapshot().jobs.find((candidate) => candidate.id === jobId);

  if (job && job.status !== status) {
    patchJob({ ...job, status });
  }
};

/** Replace one job with a backend action result (e.g. pause/resume), newer than any list request in flight. */
export const replaceInstallJob = (job: ModelInstallJob): void => {
  fenceJobStatus(job.id, job.status);
  patchJob(job);
};

/** Project a successful cancel action: the cancel route returns no job, and cancelling a job that is not downloading
 * emits no event. Cancelling a completed job succeeds as a no-op, so a known completion stands. */
export const projectInstallCancelled = (jobId: number, owner: AccountScope = captureAccountScope()): void => {
  if (!isAccountScopeCurrent(owner)) {
    return;
  }

  const isCompleted =
    statusFenceByJobId.get(jobId)?.status === 'completed' ||
    store.getSnapshot().jobs.some((job) => job.id === jobId && job.status === 'completed');

  if (!isCompleted) {
    projectJobStatus(jobId, 'cancelled');
  }
};

/** Hide a settled job locally until the backend stops listing it. */
export const dismissInstallJob = (jobId: number): void => {
  const { dismissedJobIds } = store.getSnapshot();

  if (!dismissedJobIds.has(jobId)) {
    store.patchSnapshot({ dismissedJobIds: new Set([...dismissedJobIds, jobId]) });
  }
};

/** Hide only the jobs included in a successful prune request. */
export const dismissInstallJobs = (ids: number[]): void => {
  if (ids.length > 0) {
    store.patchSnapshot({ dismissedJobIds: new Set([...store.getSnapshot().dismissedJobIds, ...ids]) });
  }
};

/** Optimistically add a freshly created job so the queue updates instantly. */
export const addInstallJob = (job: ModelInstallJob): void => {
  if (store.getSnapshot().jobs.some((existing) => existing.id === job.id)) {
    replaceInstallJob(job);
    return;
  }

  noteJobSeen(job.id);

  // Any event for the job is at least as new as its creation result, so an existing fence keeps its status.
  const fence = statusFenceByJobId.get(job.id);

  if (fence === undefined) {
    fenceJobStatus(job.id, job.status);
  }

  const added = fence === undefined || fence.status === job.status ? job : { ...job, status: fence.status };
  store.patchSnapshot({ jobs: [added, ...store.getSnapshot().jobs], status: 'loaded' });
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
  'model_install_paused',
  'model_install_complete',
  'model_install_error',
  'model_install_cancelled',
] as const;

export type ModelInstallSocketEvent = (typeof MODEL_INSTALL_SOCKET_EVENTS)[number];

const EVENT_STATUS: Record<Exclude<ModelInstallSocketEvent, 'model_install_download_progress'>, ModelInstallStatus> = {
  model_install_cancelled: 'cancelled',
  model_install_complete: 'completed',
  model_install_download_started: 'downloading',
  model_install_downloads_complete: 'downloads_done',
  model_install_error: 'error',
  model_install_paused: 'paused',
  model_install_started: 'running',
};

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

  noteJobSeen(data.id);

  if (event === 'model_install_download_progress') {
    const bytes = data.bytes ?? 0;

    progressByJobId.set(data.id, {
      bytes,
      bytesPerSecond: sampleTransferRate(data.id, bytes),
      totalBytes: data.total_bytes ?? 0,
    });

    const job = store.getSnapshot().jobs.find((candidate) => candidate.id === data.id);
    const fence = statusFenceByJobId.get(data.id);

    if ((!job || job.status === 'waiting') && (fence === undefined || fence.status === 'waiting')) {
      // Progress proves a waiting job is downloading; preserve newer pause, cancel, or terminal state.
      fenceJobStatus(data.id, 'downloading');
    }

    if (!job) {
      // The first progress tick may arrive for a job created in another
      // client; make sure the row exists without refetching on every tick.
      scheduleRefresh();
    } else if (job.status === 'waiting' && (fence === undefined || fence.status === 'waiting')) {
      // Bytes are flowing, so the REST snapshot's `waiting` is stale. Patch
      // locally so download controls (pause/cancel) appear immediately.
      patchJob({ ...job, status: 'downloading' });
    }

    return;
  }

  // Project the status each event implies (the backend's job status when it emits) and fence it, so a list response
  // that started before the event cannot replace it with an older status. The scheduled refresh then fills in the
  // fields events omit, such as a paused job's download_parts naming the part whose failure auto-paused it.
  projectJobStatus(data.id, EVENT_STATUS[event]);

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
