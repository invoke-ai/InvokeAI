import { describe, expect, it } from 'vitest';

import { getQueueStatusChip, getQueueStatusProgress } from './queueStatusModel';

describe('getQueueStatusChip', () => {
  it.each([
    ['is idle with an empty queue', 0, 'idle', { count: 0, labelKey: 'idle', tone: 'neutral' }],
    ['shows the remaining count while running', 3, 'running', { count: 3, labelKey: 'queued', tone: 'running' }],
    ['flags a paused processor with open work', 3, 'paused', { count: 3, labelKey: 'paused', tone: 'paused' }],
    [
      'says open work is waiting on the connection instead of reading as running',
      3,
      'offline',
      { count: 3, labelKey: 'offline', tone: 'offline' },
    ],
    [
      'says a model is loading while the run waits for it',
      2,
      'loading-model',
      { count: 2, labelKey: 'loadingModel', tone: 'running' },
    ],
  ] as const)('%s', (_, remaining, status, expected) => {
    expect(getQueueStatusChip({ remaining }, status)).toEqual(expected);
  });
});

describe('getQueueStatusProgress', () => {
  const running = getQueueStatusChip({ remaining: 3 }, 'running');

  it.each([
    ['fills to the reported percentage while running', running, 0.42, { value: 0.42 }],
    ['is indeterminate before the first step lands (zero)', running, 0, { value: null }],
    ['is indeterminate before the first step lands (null)', running, null, { value: null }],
    ['is indeterminate before the first step lands (undefined)', running, undefined, { value: null }],
    ['draws no bar when idle', getQueueStatusChip({ remaining: 0 }, 'idle'), 0.4, undefined],
    ['draws no bar when paused', getQueueStatusChip({ remaining: 3 }, 'paused'), 0.4, undefined],
    ['draws no bar when offline', getQueueStatusChip({ remaining: 3 }, 'offline'), 0.4, undefined],
  ])('%s', (_, chip, progress, expected) => {
    expect(getQueueStatusProgress(chip, progress)).toEqual(expected);
  });
});
