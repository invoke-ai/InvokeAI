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
 * The chip's hairline, or `undefined` for no bar at all.
 *
 * Only the running tone gets one: an idle queue has nothing to fill, and a
 * paused one would show a bar that never moves, which reads as a hang. Zero and
 * missing percentages become the quiet indeterminate fill for the same reason
 * the top bar rail does — that is the model-loading window, not a stall.
 */
export const getQueueStatusProgress = (
  chip: QueueStatusChip,
  percentage: number | null | undefined
): { value: number | null } | undefined =>
  chip.tone === 'running' ? { value: typeof percentage === 'number' && percentage > 0 ? percentage : null } : undefined;
