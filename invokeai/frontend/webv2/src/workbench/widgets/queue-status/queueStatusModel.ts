import type { QueueActivity } from '@features/queue/contracts';

import { getDeterminateProgressFraction } from '@features/queue/contracts';

export interface QueueStatusChip {
  count: number;
  labelKey: 'idle' | 'loadingModel' | 'offline' | 'paused' | 'queued';
  tone: 'neutral' | 'offline' | 'paused' | 'running';
}

export const getQueueStatusChip = (summary: { remaining: number }, activity: QueueActivity): QueueStatusChip => {
  switch (activity) {
    case 'idle':
      return { count: 0, labelKey: 'idle', tone: 'neutral' };
    // Work waiting on a lost connection must not read as running: it is neither progressing nor done.
    case 'offline':
      return { count: summary.remaining, labelKey: 'offline', tone: 'offline' };
    case 'paused':
      return { count: summary.remaining, labelKey: 'paused', tone: 'paused' };
    case 'loading-model':
      return { count: summary.remaining, labelKey: 'loadingModel', tone: 'running' };
    case 'running':
      return { count: summary.remaining, labelKey: 'queued', tone: 'running' };
  }
};

/**
 * Show a progress hairline only while running; paused or offline bars look stalled. getDeterminateProgressFraction
 * handles zero as indeterminate.
 */
export const getQueueStatusProgress = (
  chip: QueueStatusChip,
  percentage: number | null | undefined
): { value: number | null } | undefined =>
  chip.tone === 'running' ? { value: getDeterminateProgressFraction(percentage) } : undefined;
