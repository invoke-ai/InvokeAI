import { describe, expect, it } from 'vitest';

import { getQueueStatusChip, getQueueStatusProgress } from './queueStatusModel';

describe('getQueueStatusChip', () => {
  it('is idle with an empty queue', () => {
    expect(getQueueStatusChip({ remaining: 0, total: 0 }, false)).toEqual({
      count: 0,
      labelKey: 'idle',
      tone: 'neutral',
    });
  });

  it('shows the remaining count while running', () => {
    expect(getQueueStatusChip({ remaining: 3, total: 5 }, false)).toEqual({
      count: 3,
      labelKey: 'queued',
      tone: 'running',
    });
  });

  it('flags a paused processor with open work', () => {
    expect(getQueueStatusChip({ remaining: 3, total: 5 }, true)).toEqual({
      count: 3,
      labelKey: 'paused',
      tone: 'paused',
    });
  });
});

describe('getQueueStatusProgress', () => {
  const running = getQueueStatusChip({ remaining: 3, total: 5 }, false);

  it('fills to the reported percentage while running', () => {
    expect(getQueueStatusProgress(running, 0.42)).toEqual({ value: 0.42 });
  });

  it('is indeterminate before the first step lands', () => {
    expect(getQueueStatusProgress(running, 0)).toEqual({ value: null });
    expect(getQueueStatusProgress(running, null)).toEqual({ value: null });
    expect(getQueueStatusProgress(running, undefined)).toEqual({ value: null });
  });

  it('draws no bar when idle or paused', () => {
    expect(getQueueStatusProgress(getQueueStatusChip({ remaining: 0, total: 0 }, false), 0.4)).toBeUndefined();
    expect(getQueueStatusProgress(getQueueStatusChip({ remaining: 3, total: 5 }, true), 0.4)).toBeUndefined();
  });
});
