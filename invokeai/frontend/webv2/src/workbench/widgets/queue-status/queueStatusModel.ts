import { getDeterminateProgressFraction } from '@features/queue/contracts';

export interface QueueStatusChip {
  count: number;
  labelKey: 'idle' | 'paused' | 'queued';
  tone: 'neutral' | 'paused' | 'running';
}

export const getQueueStatusChip = (
  summary: { remaining: number; total: number },
  isPaused: boolean
): QueueStatusChip => {
  if (summary.total === 0) {
    return { count: 0, labelKey: 'idle', tone: 'neutral' };
  }

  return isPaused
    ? { count: summary.remaining, labelKey: 'paused', tone: 'paused' }
    : { count: summary.remaining, labelKey: 'queued', tone: 'running' };
};

/**
 * Show a progress hairline only while running; paused bars look stalled. getDeterminateProgressFraction handles
 * zero as indeterminate.
 */
export const getQueueStatusProgress = (
  chip: QueueStatusChip,
  percentage: number | null | undefined
): { value: number | null } | undefined =>
  chip.tone === 'running' ? { value: getDeterminateProgressFraction(percentage) } : undefined;
