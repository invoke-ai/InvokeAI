import type { ModelInstallJob } from '@features/models/core/types';

import { getInstallSourceLabel, type InstallDownloadProgress } from '@features/models/data/installsStore';

/** Row-level status: backend statuses folded into what the queue shows and offers actions for. */
export type InstallRowStatus =
  | 'downloading'
  | 'installing'
  | 'queued'
  | 'paused'
  | 'installed'
  | 'failed'
  | 'unauthorized'
  | 'cancelled';

export const STATUS_PRESENTATION: Record<InstallRowStatus, { labelKey: string; palette: string }> = {
  cancelled: { labelKey: 'models.statusCancelled', palette: 'gray' },
  downloading: { labelKey: 'models.statusDownloading', palette: 'blue' },
  failed: { labelKey: 'common.failed', palette: 'red' },
  installed: { labelKey: 'models.installed', palette: 'green' },
  installing: { labelKey: 'models.installing', palette: 'blue' },
  paused: { labelKey: 'models.statusPaused', palette: 'gray' },
  queued: { labelKey: 'models.statusQueued', palette: 'gray' },
  unauthorized: { labelKey: 'models.statusUnauthorized', palette: 'orange' },
};

const UNAUTHORIZED_REASONS = new Set(['GatedRepoError', 'HfHubHTTPError', 'UnauthorizedError']);
// HTTPError reasons are bare phrases ("Unauthorized"); HF client errors carry the status code.
const UNAUTHORIZED_TEXT = /^(unauthorized|forbidden)\b|\b40[13] client error\b|\bgated repo\b/i;

/** Credential failures are fixable from the queue; the backend only reports them as generic errors. */
export const isUnauthorizedFailure = (job: ModelInstallJob): boolean =>
  UNAUTHORIZED_REASONS.has(job.error_reason ?? '') ||
  UNAUTHORIZED_TEXT.test(job.error ?? '') ||
  (job.error_traceback ?? '').includes('GatedRepoError');

export const getInstallRowStatus = (job: ModelInstallJob): InstallRowStatus => {
  switch (job.status) {
    case 'downloading':
      return 'downloading';
    case 'downloads_done':
    case 'running':
      return 'installing';
    case 'paused':
      return 'paused';
    case 'completed':
      return 'installed';
    case 'cancelled':
      return 'cancelled';
    case 'error':
      return isUnauthorizedFailure(job) ? 'unauthorized' : 'failed';
    default:
      return 'queued';
  }
};

export const isActiveRowStatus = (status: InstallRowStatus): status is 'downloading' | 'installing' =>
  status === 'downloading' || status === 'installing';

export const isSettledRowStatus = (status: InstallRowStatus): boolean =>
  status === 'installed' || status === 'failed' || status === 'unauthorized' || status === 'cancelled';

export const isHuggingFaceSource = (job: ModelInstallJob): boolean =>
  typeof job.source === 'object' && job.source.type === 'hf';

/** The `source` string the install endpoint accepts, so a settled job can be resubmitted as-is. */
export const getInstallJobSourceString = (job: ModelInstallJob): string => {
  const { source } = job;

  if (typeof source === 'string') {
    return source;
  }

  switch (source.type) {
    case 'hf':
      return `${source.repo_id ?? ''}${source.variant ? `:${source.variant}` : ''}${
        source.subfolder ? `::${source.subfolder}` : ''
      }`;
    case 'external':
      return `external://${source.provider_id ?? ''}/${source.provider_model_id ?? ''}`;
    default:
      return getInstallSourceLabel(source);
  }
};

const lastSegment = (value: string): string => value.split(/[\\/]/).filter(Boolean).at(-1) ?? value;

/** Where the model comes from: repo id (with subfolder), URL, or path. */
export const getInstallJobSourceLabel = (job: ModelInstallJob): string => {
  const { source } = job;

  if (typeof source === 'object' && source.type === 'hf' && source.repo_id && source.subfolder) {
    return `${source.repo_id} :: ${source.subfolder}`;
  }

  return getInstallSourceLabel(source);
};

/** Configured name first, then the file or folder the source points at. */
export const getInstallJobDisplayName = (job: ModelInstallJob): string => {
  const configured = job.config_out?.name ?? job.config_in?.name;

  if (configured) {
    return configured;
  }

  const { source } = job;

  if (typeof source === 'object' && source.type === 'hf') {
    return source.subfolder ? lastSegment(source.subfolder) : (source.repo_id ?? getInstallSourceLabel(source));
  }

  return lastSegment(getInstallSourceLabel(source));
};

const partFileName = (part: { source?: string; local_path?: string }): string =>
  lastSegment(part.source ?? part.local_path ?? 'file');

export interface ProblemDownloadPart {
  key: string;
  fileName: string;
  /** The part's download URL, which `restart_file` matches against; null for parts without one. */
  url: string | null;
  resumeRequired: boolean;
  message: string | null;
}

/** Parts the backend could not resume or that errored; each needs a manual restart. */
export const getProblemDownloadParts = (job: ModelInstallJob): ProblemDownloadPart[] =>
  (job.download_parts ?? [])
    .filter((part) => part.resume_required === true || part.status === 'error')
    .map((part) => ({
      fileName: partFileName(part),
      key: part.source ?? part.local_path ?? partFileName(part),
      message: part.resume_message ?? null,
      resumeRequired: part.resume_required === true,
      url: typeof part.source === 'string' ? part.source : null,
    }));

/**
 * `restart_failed` only acts on errored or paused jobs with restartable parts; a cancelled job's temp dir is gone,
 * so it and anything else must be resubmitted.
 */
export const canRestartFailedParts = (job: ModelInstallJob): boolean =>
  job.status !== 'cancelled' && getProblemDownloadParts(job).length > 0;

/** A one-off token saved with the source wins over the stored provider key. */
export const getInstallJobAccessToken = (job: ModelInstallJob): string | undefined =>
  typeof job.source === 'object' && typeof job.source.access_token === 'string' && job.source.access_token
    ? job.source.access_token
    : undefined;

export interface InstallQueueRow {
  job: ModelInstallJob;
  status: InstallRowStatus;
  /** 1-based position among waiting jobs; null unless queued. */
  queuePosition: number | null;
}

const STATUS_RANK: Record<InstallRowStatus, number> = {
  cancelled: 6,
  downloading: 0,
  failed: 4,
  installed: 5,
  installing: 1,
  paused: 3,
  queued: 2,
  unauthorized: 4,
};

/** Live downloads first, then installs, queued in run order, paused, attention, installed, cancelled. */
export const buildInstallQueueRows = (
  jobs: readonly ModelInstallJob[],
  dismissedJobIds: ReadonlySet<number>
): InstallQueueRow[] => {
  const visible = jobs.filter((job) => !dismissedJobIds.has(job.id));
  const waitingIds = visible
    .filter((job) => job.status === 'waiting')
    .map((job) => job.id)
    .sort((left, right) => left - right);

  return visible
    .map((job) => {
      const status = getInstallRowStatus(job);

      return {
        job,
        queuePosition: status === 'queued' ? waitingIds.indexOf(job.id) + 1 : null,
        status,
      };
    })
    .sort((left, right) => {
      const rank = STATUS_RANK[left.status] - STATUS_RANK[right.status];

      if (rank !== 0) {
        return rank;
      }

      return left.status === 'queued' ? left.job.id - right.job.id : right.job.id - left.job.id;
    });
};

export interface InstallQueueSummary {
  downloading: number;
  installing: number;
  queued: number;
  paused: number;
  attention: number;
  installed: number;
  settled: number;
}

/** Terminal jobs the backend still holds, dismissed or not: what "Clear finished" would prune. */
export const countFinishedInstallJobs = (jobs: readonly ModelInstallJob[]): number =>
  jobs.filter((job) => isSettledRowStatus(getInstallRowStatus(job))).length;

export const summarizeInstallQueue = (rows: readonly InstallQueueRow[]): InstallQueueSummary => {
  const summary: InstallQueueSummary = {
    attention: 0,
    downloading: 0,
    installed: 0,
    installing: 0,
    paused: 0,
    queued: 0,
    settled: 0,
  };

  for (const { status } of rows) {
    if (status === 'downloading') {
      summary.downloading += 1;
    } else if (status === 'installing') {
      summary.installing += 1;
    } else if (status === 'queued') {
      summary.queued += 1;
    } else if (status === 'paused') {
      summary.paused += 1;
    } else if (status === 'failed' || status === 'unauthorized') {
      summary.attention += 1;
    } else if (status === 'installed') {
      summary.installed += 1;
    }

    if (isSettledRowStatus(status)) {
      summary.settled += 1;
    }
  }

  return summary;
};

const MIN_ETA_RATE_BYTES_PER_SECOND = 8 * 1024;

export interface InstallByteProgress {
  bytes: number;
  totalBytes: number;
  ratio: number | null;
  bytesPerSecond: number | null;
  /** Seconds until done at the smoothed rate; null without a rate or total. */
  etaSeconds: number | null;
}

/** Live socket bytes win over the REST snapshot; the snapshot still carries paused/settled totals. */
export const resolveInstallProgress = (
  job: ModelInstallJob,
  live: InstallDownloadProgress | null
): InstallByteProgress => {
  const bytes = live?.bytes ?? job.bytes ?? 0;
  const totalBytes = live?.totalBytes ?? job.total_bytes ?? 0;
  const ratio = totalBytes > 0 ? Math.min(1, bytes / totalBytes) : null;
  const bytesPerSecond = live?.bytesPerSecond ?? null;
  // A decayed or stalled rate would print an absurd ETA; below the floor the estimate is unknown.
  const etaSeconds =
    bytesPerSecond !== null && bytesPerSecond >= MIN_ETA_RATE_BYTES_PER_SECOND && totalBytes > bytes
      ? Math.ceil((totalBytes - bytes) / bytesPerSecond)
      : null;

  return { bytes, bytesPerSecond, etaSeconds, ratio, totalBytes };
};

export type EtaLabel = { unit: 'seconds' | 'minutes' | 'hours'; count: number };

/** Coarse, rounded-up buckets: a download ETA is an estimate, not a countdown. */
export const describeEta = (etaSeconds: number): EtaLabel => {
  if (etaSeconds < 60) {
    return { count: Math.max(1, Math.ceil(etaSeconds / 5) * 5), unit: 'seconds' };
  }

  if (etaSeconds < 3600) {
    return { count: Math.ceil(etaSeconds / 60), unit: 'minutes' };
  }

  return { count: Math.ceil(etaSeconds / 3600), unit: 'hours' };
};
