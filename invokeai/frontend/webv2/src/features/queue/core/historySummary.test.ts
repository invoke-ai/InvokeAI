import type { QueueItem } from '@features/queue';
import type { QueueItemProgress } from '@features/queue/core/types';

import { describe, expect, it } from 'vitest';

import {
  getProjectQueueIndicatorState,
  getQueueProgressBarState,
  getQueueProgressBarValue,
  getQueueSummary,
} from './historySummary';

const createQueueItem = ({
  backendItemIds,
  batchCount = 1,
  id,
  sourceId = 'generate',
  status = 'pending',
  submittedAt,
}: {
  backendItemIds?: number[];
  batchCount?: number;
  id: string;
  sourceId?: QueueItem['snapshot']['sourceId'];
  status?: QueueItem['status'];
  submittedAt: string;
}): QueueItem => ({
  backendItemIds,
  cancellable: true,
  id,
  snapshot: {
    backendSubmission: { error: 'not submitted in summary tests', kind: 'invalid' },
    destination: 'gallery',
    filterIntermediateResults: false,
    galleryBoardId: null,
    graph: { id: 'graph-1', label: 'Graph' },
    presentation: { batchCount, height: 1024, width: 1024 },
    sourceId,
    submittedAt,
  },
  status,
});

describe('getQueueSummary', () => {
  it('counts image slots inside the running batch', () => {
    const summary = getQueueSummary([
      createQueueItem({
        backendItemIds: [1, 2, 3, 4],
        batchCount: 4,
        id: 'batch-1',
        status: 'running',
        submittedAt: '2026-06-10T00:00:00.000Z',
      }),
    ]);

    expect(summary).toEqual({ current: 1, remaining: 4, runningQueueItemId: 'batch-1', total: 4 });
  });

  it('keeps the running ordinal stable when newer work is queued', () => {
    const summary = getQueueSummary([
      createQueueItem({
        batchCount: 4,
        id: 'batch-2',
        status: 'pending',
        submittedAt: '2026-06-10T00:01:00.000Z',
      }),
      createQueueItem({
        backendItemIds: [1, 2, 3, 4],
        batchCount: 4,
        id: 'batch-1',
        status: 'running',
        submittedAt: '2026-06-10T00:00:00.000Z',
      }),
    ]);

    expect(summary).toEqual({ current: 1, remaining: 8, runningQueueItemId: 'batch-1', total: 8 });
  });

  it('uses live progress to advance within a batch', () => {
    const progress: QueueItemProgress = {
      activeItemIndex: 2,
      completedItemCount: 1,
      message: '',
      percentage: null,
      totalItemCount: 4,
    };
    const summary = getQueueSummary(
      [
        createQueueItem({
          backendItemIds: [1, 2, 3, 4],
          batchCount: 4,
          id: 'batch-1',
          status: 'running',
          submittedAt: '2026-06-10T00:00:00.000Z',
        }),
        createQueueItem({
          batchCount: 4,
          id: 'batch-2',
          status: 'pending',
          submittedAt: '2026-06-10T00:01:00.000Z',
        }),
      ],
      progress
    );

    expect(summary).toEqual({ current: 2, remaining: 7, runningQueueItemId: 'batch-1', total: 8 });
  });

  it('does not move the queue ordinal backwards as concurrent iterations report out of order', () => {
    const item = createQueueItem({
      backendItemIds: [10, 11, 12, 13, 14],
      batchCount: 5,
      id: 'batch-concurrent',
      status: 'running',
      submittedAt: '2026-06-10T00:00:00.000Z',
    });
    const reports = [
      { activeItemIndex: 4, completedItemCount: 0 },
      { activeItemIndex: 1, completedItemCount: 0 },
      { activeItemIndex: 5, completedItemCount: 1 },
      { activeItemIndex: 2, completedItemCount: 1 },
      { activeItemIndex: 3, completedItemCount: 2 },
    ];

    expect(
      reports.map(
        ({ activeItemIndex, completedItemCount }) =>
          getQueueSummary([item], { activeItemIndex, completedItemCount, message: '', percentage: null }).current
      )
    ).toEqual([1, 1, 2, 2, 3]);
  });

  it('returns zeroes when all batches are terminal', () => {
    const summary = getQueueSummary([
      createQueueItem({ batchCount: 4, id: 'batch-1', status: 'completed', submittedAt: '2026-06-10T00:00:00.000Z' }),
    ]);

    expect(summary).toEqual({ current: 0, remaining: 0, runningQueueItemId: null, total: 0 });
  });
});

describe('getProjectQueueIndicatorState', () => {
  it('returns indeterminate state for running work without percentage progress', () => {
    const queueItems = [createQueueItem({ id: 'running', status: 'running', submittedAt: '2026-06-10T00:00:00.000Z' })];

    expect(
      getProjectQueueIndicatorState({ isConnected: true, loadingModelsCount: 0, progress: null, queueItems })
    ).toEqual({
      hasOpenQueueWork: true,
      progressState: { kind: 'indeterminate', value: null },
      runningQueueItemId: 'running',
    });
  });

  it('returns determinate state for running work with percentage progress', () => {
    const queueItems = [createQueueItem({ id: 'running', status: 'running', submittedAt: '2026-06-10T00:00:00.000Z' })];
    const progress: QueueItemProgress = { message: 'Denoising', percentage: 0.42 };

    expect(getProjectQueueIndicatorState({ isConnected: true, loadingModelsCount: 0, progress, queueItems })).toEqual({
      hasOpenQueueWork: true,
      progressState: { kind: 'determinate', value: 0.42 },
      runningQueueItemId: 'running',
    });
  });
});

describe('getQueueProgressBarValue', () => {
  it('matches legacy indeterminate conditions', () => {
    expect(
      getQueueProgressBarState({ isConnected: true, isRunning: false, loadingModelsCount: 1, progress: null })
    ).toEqual({ kind: 'indeterminate', value: null });
    expect(
      getQueueProgressBarValue({ isConnected: true, isRunning: false, loadingModelsCount: 1, progress: null })
    ).toBeNull();
    expect(
      getQueueProgressBarValue({ isConnected: true, isRunning: true, loadingModelsCount: 0, progress: null })
    ).toBeNull();
    expect(
      getQueueProgressBarValue({
        isConnected: true,
        isRunning: true,
        loadingModelsCount: 0,
        progress: { message: '', percentage: null },
      })
    ).toBeNull();
    expect(
      getQueueProgressBarValue({
        isConnected: true,
        isRunning: true,
        loadingModelsCount: 0,
        progress: { message: '', percentage: 0 },
      })
    ).toBeNull();
  });

  it('resets to zero while disconnected', () => {
    expect(
      getQueueProgressBarState({
        isConnected: false,
        isRunning: true,
        loadingModelsCount: 1,
        progress: { message: 'Denoising', percentage: 0.42 },
      })
    ).toEqual({ kind: 'idle', value: 0 });
    expect(
      getQueueProgressBarValue({
        isConnected: false,
        isRunning: true,
        loadingModelsCount: 1,
        progress: { message: 'Denoising', percentage: 0.42 },
      })
    ).toBe(0);
  });
});
