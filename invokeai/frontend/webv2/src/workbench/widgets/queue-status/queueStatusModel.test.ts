import { describe, expect, it } from 'vitest';

import { getQueueStatusChip, getQueueStatusProgress } from './queueStatusModel';

describe('getQueueStatusChip', () => {
  it('is idle with an empty queue', () => {
    expect(getQueueStatusChip({ remaining: 0 }, 'idle')).toEqual({ count: 0, labelKey: 'idle', tone: 'neutral' });
  });

  it('shows the remaining count while running', () => {
    expect(getQueueStatusChip({ remaining: 3 }, 'running')).toEqual({ count: 3, labelKey: 'queued', tone: 'running' });
  });

  it('flags a paused processor with open work', () => {
    expect(getQueueStatusChip({ remaining: 3 }, 'paused')).toEqual({ count: 3, labelKey: 'paused', tone: 'paused' });
  });

  it('says open work is waiting on the connection instead of reading as running', () => {
    expect(getQueueStatusChip({ remaining: 3 }, 'offline')).toEqual({ count: 3, labelKey: 'offline', tone: 'offline' });
  });

  it('says a model is loading while the run waits for it', () => {
    expect(getQueueStatusChip({ remaining: 2 }, 'loading-model')).toEqual({
      count: 2,
      labelKey: 'loadingModel',
      tone: 'running',
    });
  });
});

describe('getQueueStatusProgress', () => {
  const running = getQueueStatusChip({ remaining: 3 }, 'running');

  it('fills to the reported percentage while running', () => {
    expect(getQueueStatusProgress(running, 0.42)).toEqual({ value: 0.42 });
  });

  it('is indeterminate before the first step lands', () => {
    expect(getQueueStatusProgress(running, 0)).toEqual({ value: null });
    expect(getQueueStatusProgress(running, null)).toEqual({ value: null });
    expect(getQueueStatusProgress(running, undefined)).toEqual({ value: null });
  });

  it('draws no bar when idle, paused or offline', () => {
    expect(getQueueStatusProgress(getQueueStatusChip({ remaining: 0 }, 'idle'), 0.4)).toBeUndefined();
    expect(getQueueStatusProgress(getQueueStatusChip({ remaining: 3 }, 'paused'), 0.4)).toBeUndefined();
    expect(getQueueStatusProgress(getQueueStatusChip({ remaining: 3 }, 'offline'), 0.4)).toBeUndefined();
  });
});
