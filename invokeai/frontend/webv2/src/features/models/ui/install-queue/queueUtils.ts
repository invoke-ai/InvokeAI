import type { ModelInstallJob, ModelInstallStatus } from '@features/models/core/types';

import { getInstallSourceLabel } from '@features/models/data/installsStore';

/** Labels are i18n keys — translate with `t(labelKey)` at the render site. */
export const STATUS_BADGES: Record<ModelInstallStatus, { labelKey: string; palette: string }> = {
  cancelled: { labelKey: 'models.statusCancelled', palette: 'gray' },
  completed: { labelKey: 'models.installed', palette: 'green' },
  downloading: { labelKey: 'models.statusDownloading', palette: 'blue' },
  downloads_done: { labelKey: 'models.statusDownloaded', palette: 'blue' },
  error: { labelKey: 'common.failed', palette: 'red' },
  paused: { labelKey: 'models.statusPaused', palette: 'orange' },
  running: { labelKey: 'models.installing', palette: 'blue' },
  waiting: { labelKey: 'models.statusWaiting', palette: 'gray' },
};

export const getInstallJobDisplayName = (job: ModelInstallJob): string =>
  job.config_out?.name ?? getInstallSourceLabel(job.source);
