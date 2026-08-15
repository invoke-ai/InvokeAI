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
