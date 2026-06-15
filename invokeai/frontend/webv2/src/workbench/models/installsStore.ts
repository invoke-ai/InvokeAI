import { useSyncExternalStore } from 'react';

import { createExternalStore, createListenerChannel } from '@workbench/externalStore';
import { listModelInstalls } from './api';
import { refreshModels } from './modelsStore';
import { refreshStartersIfLoaded } from './startersStore';
import type { ModelInstallJob, ModelInstallStatus } from './types';

/**
 * Live store for model install jobs. The job list itself is REST-owned
 * (`/api/v2/models/install`) and refreshed on lifecycle socket events;
 * download progress is high-frequency transient data that bypasses the list
 * (and the workbench reducer) entirely — each queue row subscribes to its own
 * job id and only re-renders when that job's bytes move. This mirrors the
 * generation `progressStore` pattern.
 */

export interface InstallsSnapshot {
  jobs: ModelInstallJob[];
  status: 'idle' | 'loading' | 'loaded' | 'error';
  error: string | null;
}

export interface InstallDownloadProgress {
  bytes: number;
  totalBytes: number;
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

const REFRESH_COALESCE_MS = 250;
const OUTCOME_LIMIT = 16;

const store = createExternalStore<InstallsSnapshot>({ error: null, jobs: [], status: 'idle' });

let outcomes: InstallOutcome[] = [];
let nextOutcomeId = 1;

const progressByJobId = new Map<number, InstallDownloadProgress>();

const progressChannel = createListenerChannel();
const outcomesChannel = createListenerChannel();

let inflightRefresh: Promise<void> | null = null;
let refreshTimer: ReturnType<typeof setTimeout> | null = null;

export const refreshInstalls = (): Promise<void> => {
  if (inflightRefresh) {
    return inflightRefresh;
  }

  store.patchSnapshot({ status: store.getSnapshot().status === 'loaded' ? 'loaded' : 'loading' });

  inflightRefresh = listModelInstalls()
    .then((jobs) => {
      const activeJobIds = new Set(jobs.map((job) => job.id));

      for (const jobId of progressByJobId.keys()) {
        if (!activeJobIds.has(jobId)) {
          progressByJobId.delete(jobId);
        }
      }

      store.patchSnapshot({ error: null, jobs, status: 'loaded' });
      progressChannel.notify();
    })
    .catch((error: unknown) => {
      store.patchSnapshot({
        error: error instanceof Error ? error.message : 'Failed to load install queue.',
        status: store.getSnapshot().jobs.length > 0 ? 'loaded' : 'error',
      });
    })
    .finally(() => {
      inflightRefresh = null;
    });

  return inflightRefresh;
};

export const ensureInstallsLoaded = (): void => {
  if (store.getSnapshot().status === 'idle') {
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

/** Optimistically replace one job (e.g. after pause/resume API calls). */
export const replaceInstallJob = (job: ModelInstallJob): void => {
  store.patchSnapshot({
    jobs: store.getSnapshot().jobs.map((existing) => (existing.id === job.id ? job : existing)),
  });
};

/** Optimistically add a freshly created job so the queue updates instantly. */
export const addInstallJob = (job: ModelInstallJob): void => {
  if (store.getSnapshot().jobs.some((existing) => existing.id === job.id)) {
    replaceInstallJob(job);
    return;
  }

  store.patchSnapshot({ jobs: [job, ...store.getSnapshot().jobs], status: 'loaded' });
};

const recordOutcome = (outcome: Omit<InstallOutcome, 'id'>): void => {
  outcomes = [{ ...outcome, id: nextOutcomeId }, ...outcomes].slice(0, OUTCOME_LIMIT);
  nextOutcomeId += 1;
  outcomesChannel.notify();
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

const describeSource = (source: unknown): string => {
  if (typeof source === 'string') {
    return source;
  }

  if (source && typeof source === 'object') {
    const record = source as Record<string, unknown>;

    for (const field of ['repo_id', 'url', 'path']) {
      if (typeof record[field] === 'string') {
        return record[field];
      }
    }
  }

  return 'model';
};

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
export const handleModelInstallSocketEvent = (event: ModelInstallSocketEvent, payload: unknown): void => {
  const data = payload as ModelInstallSocketPayload;

  if (typeof data?.id !== 'number') {
    return;
  }

  if (event === 'model_install_download_progress') {
    progressByJobId.set(data.id, { bytes: data.bytes ?? 0, totalBytes: data.total_bytes ?? 0 });
    progressChannel.notify();

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

  if (event === 'model_install_complete') {
    recordOutcome({
      error: null,
      jobId: data.id,
      kind: 'completed',
      modelName: data.config?.name ?? null,
      source: describeSource(data.source),
    });
    void refreshModels();
    refreshStartersIfLoaded();
  } else if (event === 'model_install_error') {
    recordOutcome({
      error: data.error ?? data.error_type ?? 'Unknown install error.',
      jobId: data.id,
      kind: 'error',
      modelName: null,
      source: describeSource(data.source),
    });
  } else if (event === 'model_install_cancelled') {
    recordOutcome({
      error: null,
      jobId: data.id,
      kind: 'cancelled',
      modelName: null,
      source: describeSource(data.source),
    });
  }

  scheduleRefresh();
};

const ACTIVE_STATUSES: ModelInstallStatus[] = ['waiting', 'downloading', 'downloads_done', 'running'];

export const isActiveInstallStatus = (status: ModelInstallStatus): boolean => ACTIVE_STATUSES.includes(status);

export const useInstallsSnapshot = (): InstallsSnapshot => store.useSnapshot();

/**
 * Source strings (URL, repo id, or path) of jobs currently in flight, cached
 * per jobs-array so list rows can show an "installing" state by source.
 */
let activeSourcesCache: { jobs: ModelInstallJob[]; sources: ReadonlySet<string> } | null = null;

const getActiveInstallSources = (): ReadonlySet<string> => {
  const { jobs } = store.getSnapshot();

  if (activeSourcesCache?.jobs !== jobs) {
    activeSourcesCache = {
      jobs,
      sources: new Set(
        jobs
          .filter((job) => isActiveInstallStatus(job.status) || job.status === 'paused')
          .map((job) => describeSource(job.source))
      ),
    };
  }

  return activeSourcesCache.sources;
};

export const useActiveInstallSources = (): ReadonlySet<string> =>
  useSyncExternalStore(store.subscribe, getActiveInstallSources);

export const useInstallProgress = (jobId: number): InstallDownloadProgress | null =>
  useSyncExternalStore(progressChannel.subscribe, () => progressByJobId.get(jobId) ?? null);

export const useInstallOutcomes = (): InstallOutcome[] =>
  useSyncExternalStore(outcomesChannel.subscribe, () => outcomes);
