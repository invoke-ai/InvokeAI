import type { QueueItem } from '@features/queue';

import { describe, expect, it } from 'vitest';

import { getProgressRailModel, getProgressRailSegmentValue, selectProjectProgressItemIds } from './progressRail';

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
    graph: { edges: [], id: 'graph-1', label: 'Graph', nodes: [], updatedAt: '2026-08-15T00:00:00.000Z', version: 1 },
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

describe('getProgressRailSegmentValue', () => {
  it('passes a real percentage through', () => {
    expect(getProgressRailSegmentValue({ isLoadingModels: false, percentage: 0.42 })).toBe(0.42);
  });

  it('treats a missing percentage as indeterminate', () => {
    expect(getProgressRailSegmentValue({ isLoadingModels: false, percentage: null })).toBeNull();
    expect(getProgressRailSegmentValue({ isLoadingModels: false, percentage: undefined })).toBeNull();
  });

  it('treats zero as indeterminate rather than as an empty bar', () => {
    expect(getProgressRailSegmentValue({ isLoadingModels: false, percentage: 0 })).toBeNull();
  });

  it('stays indeterminate while models load, whatever the last percentage was', () => {
    expect(getProgressRailSegmentValue({ isLoadingModels: true, percentage: 0.9 })).toBeNull();
  });
});

describe('selectProjectProgressItemIds', () => {
  it('keeps only the ids this project enqueued', () => {
    const items = [createQueueItem({ backendItemIds: [10, 11], id: 'batch-1', status: 'running' })];

    expect(selectProjectProgressItemIds(items, [9, 10, 11, 12])).toEqual([10, 11]);
  });

  it('preserves ascending order so segments do not swap places between frames', () => {
    const items = [
      createQueueItem({ backendItemIds: [30], id: 'batch-2', status: 'running' }),
      createQueueItem({ backendItemIds: [20], id: 'batch-1', status: 'running' }),
    ];

    expect(selectProjectProgressItemIds(items, [20, 30])).toEqual([20, 30]);
  });

  it('ignores items that are no longer open', () => {
    const items = [
      createQueueItem({ backendItemIds: [40], id: 'batch-1', status: 'completed' }),
      createQueueItem({ backendItemIds: [41], id: 'batch-2', status: 'running' }),
    ];

    expect(selectProjectProgressItemIds(items, [40, 41])).toEqual([41]);
  });

  it('returns nothing when the store has no live sessions', () => {
    const items = [createQueueItem({ backendItemIds: [50], id: 'batch-1', status: 'running' })];

    expect(selectProjectProgressItemIds(items, [])).toEqual([]);
  });

  it('tolerates open items that have not been assigned backend ids yet', () => {
    const items = [createQueueItem({ id: 'batch-1', status: 'pending' })];

    expect(selectProjectProgressItemIds(items, [60])).toEqual([]);
  });
});
