import type { QueueItem } from '@features/queue';

import { describe, expect, it } from 'vitest';

import {
  getDeterminateProgressFraction,
  getDeterminateProgressPercent,
  getProgressRailModel,
  getProgressRailSegmentValue,
  selectProjectProgressItemIds,
} from './progressRail';

const createQueueItem = ({
  backendItemIds,
  id,
  status = 'pending',
}: {
  backendItemIds?: number[];
  id: string;
  status?: QueueItem['status'];
}): QueueItem => ({
  backendItemIds,
  cancellable: true,
  id,
  snapshot: {
    backendSubmission: { error: 'not submitted in rail tests', kind: 'invalid' },
    destination: 'gallery',
    filterIntermediateResults: false,
    galleryBoardId: null,
    graph: { id: 'graph-1', label: 'Graph' },
    presentation: { batchCount: 1, height: 1024, width: 1024 },
    sourceId: 'generate',
    submittedAt: '2026-08-15T00:00:00.000Z',
  },
  status,
});

describe('getProgressRailModel', () => {
  it('hides while disconnected, even with live sessions', () => {
    expect(getProgressRailModel({ hasOpenWork: true, isConnected: false, sessionItemIds: [1] })).toEqual({
      kind: 'hidden',
    });
  });

  it('hides when the project has no open work', () => {
    expect(getProgressRailModel({ hasOpenWork: false, isConnected: true, sessionItemIds: [] })).toEqual({
      kind: 'hidden',
    });
  });

  it('shows a single pending segment between enqueue and the first progress event', () => {
    expect(getProgressRailModel({ hasOpenWork: true, isConnected: true, sessionItemIds: [] })).toEqual({
      kind: 'pending',
    });
  });

  it('gives every live session its own segment', () => {
    expect(getProgressRailModel({ hasOpenWork: true, isConnected: true, sessionItemIds: [7, 8] })).toEqual({
      itemIds: [7, 8],
      kind: 'sessions',
    });
  });
});

describe('getDeterminateProgressFraction', () => {
  it('treats zero and missing as indeterminate', () => {
    expect(getDeterminateProgressFraction(0)).toBeNull();
    expect(getDeterminateProgressFraction(null)).toBeNull();
    expect(getDeterminateProgressFraction(undefined)).toBeNull();
  });

  it('passes real fractions through and rounds percents', () => {
    expect(getDeterminateProgressFraction(0.42)).toBe(0.42);
    expect(getDeterminateProgressPercent(0.428)).toBe(43);
    expect(getDeterminateProgressPercent(0)).toBeNull();
  });
});

describe('getProgressRailSegmentValue', () => {
  it.each<[string, boolean, number | null | undefined, number | null]>([
    ['passes a real percentage through', false, 0.42, 0.42],
    ['treats a null percentage as indeterminate', false, null, null],
    ['treats a missing percentage as indeterminate', false, undefined, null],
    ['treats zero as indeterminate rather than as an empty bar', false, 0, null],
    ['stays indeterminate while models load, whatever the last percentage was', true, 0.9, null],
  ])('%s', (_label, isLoadingModels, percentage, expected) => {
    expect(getProgressRailSegmentValue({ isLoadingModels, percentage })).toBe(expected);
  });
});

describe('selectProjectProgressItemIds', () => {
  it.each<[string, QueueItem[], number[], number[]]>([
    [
      'keeps only the ids this project enqueued',
      [createQueueItem({ backendItemIds: [10, 11], id: 'batch-1', status: 'running' })],
      [9, 10, 11, 12],
      [10, 11],
    ],
    [
      'preserves ascending order so segments do not swap places between frames',
      [
        createQueueItem({ backendItemIds: [30], id: 'batch-2', status: 'running' }),
        createQueueItem({ backendItemIds: [20], id: 'batch-1', status: 'running' }),
      ],
      [20, 30],
      [20, 30],
    ],
    [
      'ignores items that are no longer open',
      [
        createQueueItem({ backendItemIds: [40], id: 'batch-1', status: 'completed' }),
        createQueueItem({ backendItemIds: [41], id: 'batch-2', status: 'running' }),
      ],
      [40, 41],
      [41],
    ],
    [
      'returns nothing when the store has no live sessions',
      [createQueueItem({ backendItemIds: [50], id: 'batch-1', status: 'running' })],
      [],
      [],
    ],
    [
      'tolerates open items that have not been assigned backend ids yet',
      [createQueueItem({ id: 'batch-1', status: 'pending' })],
      [60],
      [],
    ],
  ])('%s', (_label, items, sessionItemIds, expected) => {
    expect(selectProjectProgressItemIds(items, sessionItemIds)).toEqual(expected);
  });
});
