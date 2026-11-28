import type { QueueItem } from '@features/queue';
import type { QueueItemProgress } from '@features/queue/core/types';

import { describe, expect, it } from 'vitest';

import {
  getFinishedImageCount,
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

describe('getFinishedImageCount', () => {
  const item = {
    ...createQueueItem({
      backendItemIds: [1, 2, 3, 4],
      id: 'batch-1',
      status: 'running',
      submittedAt: '2026-06-10T00:00:00Z',
    }),
    completedBackendItemIds: [1, 1, 2],
    cancelledBackendItemIds: [2, 3],
  };

  it('deduplicates completed and cancelled backend slots', () => {
    expect(getFinishedImageCount(item)).toBe(3);
  });

  it('uses the greater live or recorded settled count', () => {
    expect(
      getFinishedImageCount(item, {
        completedItemCount: 1,
        message: '',
        percentage: null,
      })
    ).toBe(3);

    expect(
      getFinishedImageCount(item, {
        completedItemCount: 4,
        message: '',
        percentage: null,
      })
    ).toBe(4);
  });

  it('caps the settled count at the expected batch size', () => {
    expect(
      getFinishedImageCount(item, {
        completedItemCount: 6,
        message: '',
        percentage: null,
      })
    ).toBe(4);

    expect(
      getFinishedImageCount({
        ...item,
        completedBackendItemIds: [1, 2, 3, 4, 5],
      })
    ).toBe(4);
  });
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
          getQueueSummary([item], { activeItemIndex, completedItemCount, message: '', percentage: null }, true).current
      )
    ).toEqual([1, 1, 2, 2, 3]);
  });

  it('counts unfinished slots across concurrently running batches', () => {
    const first = {
      ...createQueueItem({
        backendItemIds: [7, 8, 9],
        id: 'first',
        status: 'running',
        submittedAt: '2026-06-10T00:00:00Z',
      }),
      completedBackendItemIds: [7],
    };
    const second = {
      ...createQueueItem({
        backendItemIds: [10, 11, 12],
        id: 'second',
        status: 'running',
        submittedAt: '2026-06-10T00:01:00Z',
      }),
      completedBackendItemIds: [10, 11],
    };

    expect(getQueueSummary([second, first])).toEqual({
      current: 2,
      remaining: 3,
      runningQueueItemId: 'first',
      total: 6,
    });
    // A newer progress reading advances only its owning batch, not both batches.
    expect(getQueueSummary([second, first], { completedItemCount: 2, message: '', percentage: null }).remaining).toBe(
      2
    );
  });

  it('includes older pending work and excludes cancelled slots in another open batch', () => {
    const pending = createQueueItem({ batchCount: 2, id: 'pending', submittedAt: '2026-06-10T00:00:00Z' });
    const first = createQueueItem({
      backendItemIds: [1, 2],
      id: 'first',
      status: 'running',
      submittedAt: '2026-06-10T00:01:00Z',
    });
    const second = {
      ...createQueueItem({
        backendItemIds: [3, 4, 5],
        id: 'second',
        status: 'running',
        submittedAt: '2026-06-10T00:02:00Z',
      }),
      cancelledBackendItemIds: [4],
      completedBackendItemIds: [3],
    };

    expect(getQueueSummary([second, first, pending])).toEqual({
      current: 3,
      remaining: 5,
      runningQueueItemId: 'first',
      total: 7,
    });
  });

  it('counts only unsettled slots when a partially finished batch is pending', () => {
    const pending = {
      ...createQueueItem({ backendItemIds: [1, 2, 3], id: 'pending', submittedAt: '2026-06-10T00:00:00Z' }),
      completedBackendItemIds: [1],
    };
    expect(getQueueSummary([pending])).toEqual({ current: 0, remaining: 2, runningQueueItemId: null, total: 3 });
  });

  it('keeps a single-image run open until result delivery closes it', () => {
    const item = createQueueItem({
      backendItemIds: [1],
      id: 'batch-1',
      status: 'running',
      submittedAt: '2026-06-10T00:00:00Z',
    });
    const progress: QueueItemProgress = { completedItemCount: 1, message: '', percentage: null };
    const delivered = { ...item, completedBackendItemIds: [1] };

    // Terminal backend status, transient progress cleanup, then recorded delivery.
    expect([
      getQueueSummary([item], progress).remaining,
      getQueueSummary([item]).remaining,
      getQueueSummary([delivered]).remaining,
      getQueueSummary([{ ...delivered, status: 'completed' }]).remaining,
    ]).toEqual([1, 1, 1, 0]);
  });

  it('keeps every running batch open while settled results are being delivered', () => {
    const first = {
      ...createQueueItem({
        backendItemIds: [1, 2],
        id: 'first',
        status: 'running',
        submittedAt: '2026-06-10T00:00:00Z',
      }),
      completedBackendItemIds: [1, 2],
    };
    const second = {
      ...createQueueItem({ backendItemIds: [3], id: 'second', status: 'running', submittedAt: '2026-06-10T00:01:00Z' }),
      completedBackendItemIds: [3],
    };
    expect(getQueueSummary([first, second]).remaining).toBe(2);
    expect(getQueueSummary([{ ...first, status: 'completed' }, second]).remaining).toBe(1);
  });

  it('uses the actual sequential image ordinal after an earlier image fails', () => {
    const item = {
      ...createQueueItem({
        backendItemIds: [1, 2, 3, 4],
        id: 'batch-1',
        status: 'running',
        submittedAt: '2026-06-10T00:00:00Z',
      }),
      completedBackendItemIds: [1],
    };
    const progress: QueueItemProgress = {
      activeItemIndex: 3,
      completedItemCount: 2,
      message: 'Denoising',
      percentage: 0.6,
      totalItemCount: 4,
    };
    expect(getQueueSummary([item], progress)).toEqual({
      current: 3,
      remaining: 2,
      runningQueueItemId: 'batch-1',
      total: 4,
    });
  });

  it('does not advance the sequential image ordinal when a later pending image is cancelled', () => {
    const item = {
      ...createQueueItem({
        backendItemIds: [1, 2, 3, 4],
        id: 'batch-1',
        status: 'running',
        submittedAt: '2026-06-10T00:00:00Z',
      }),
      cancelledBackendItemIds: [4],
    };
    const progress: QueueItemProgress = {
      activeItemIndex: 1,
      completedItemCount: 1,
      message: 'Denoising',
      percentage: 0.6,
      totalItemCount: 4,
    };
    expect(getQueueSummary([item], progress)).toEqual({
      current: 1,
      remaining: 3,
      runningQueueItemId: 'batch-1',
      total: 4,
    });
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
