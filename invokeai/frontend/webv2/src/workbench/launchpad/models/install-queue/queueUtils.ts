import type { ModelInstallJob, ModelInstallStatus } from '@workbench/models/types';

import { getInstallSourceLabel } from '@workbench/models/taxonomy';

export const STATUS_BADGES: Record<ModelInstallStatus, { label: string; palette: string }> = {
  cancelled: { label: 'Cancelled', palette: 'gray' },
  completed: { label: 'Installed', palette: 'green' },
  downloading: { label: 'Downloading', palette: 'blue' },
  downloads_done: { label: 'Downloaded', palette: 'blue' },
  error: { label: 'Failed', palette: 'red' },
  paused: { label: 'Paused', palette: 'orange' },
  running: { label: 'Installing', palette: 'blue' },
  waiting: { label: 'Waiting', palette: 'gray' },
};

export const getInstallJobDisplayName = (job: ModelInstallJob): string =>
  job.config_out?.name ?? getInstallSourceLabel(job.source);
