import { describe, expect, it } from 'vitest';

import type { QueueItemProgress } from './backend/progressStore';
import type { QueueItem } from './types';

import {
  getProjectQueueIndicatorState,
  getQueueProgressBarState,
  getQueueProgressBarValue,
  getQueueSummary,
} from './queueSummary';

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
    canvas: {
      document: { height: 1024, layers: [], version: 1, width: 1024 },
      stagingArea: {
        areThumbnailsVisible: false,
        isVisible: false,
        pendingImageIds: [],
        pendingImages: [],
        selectedImageIndex: 0,
      },
      version: 1,
    },
    destination: 'gallery',
    graph: { edges: [], id: 'graph-1', label: 'Graph', nodes: [], updatedAt: submittedAt, version: 1 },
    sourceId,
    submittedAt,
    widgetInstances: {},
    widgetStates: {
      generate: { id: 'generate', label: 'Generate', values: { batchCount }, version: 1 },
    } as unknown as QueueItem['snapshot']['widgetStates'],
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

    expect(summary).toEqual({ current: 1, runningQueueItemId: 'batch-1', total: 4 });
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

    expect(summary).toEqual({ current: 1, runningQueueItemId: 'batch-1', total: 8 });
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

    expect(summary).toEqual({ current: 2, runningQueueItemId: 'batch-1', total: 8 });
  });

  it('returns zeroes when all batches are terminal', () => {
    const summary = getQueueSummary([
      createQueueItem({ batchCount: 4, id: 'batch-1', status: 'completed', submittedAt: '2026-06-10T00:00:00.000Z' }),
    ]);

    expect(summary).toEqual({ current: 0, runningQueueItemId: null, total: 0 });
  });

  it('uses the global batch count for workflow-sourced queue items', () => {
    const summary = getQueueSummary([
      createQueueItem({
        batchCount: 2,
        id: 'workflow-batch',
        sourceId: 'workflow',
        status: 'pending',
        submittedAt: '2026-06-10T00:00:00.000Z',
      }),
    ]);

    expect(summary).toEqual({ current: 0, runningQueueItemId: null, total: 2 });
  });

  it('does not cap oversized batch counts', () => {
    const summary = getQueueSummary([
      createQueueItem({
        batchCount: 10_000,
        id: 'oversized-batch',
        status: 'pending',
        submittedAt: '2026-06-10T00:00:00.000Z',
      }),
    ]);

    expect(summary).toEqual({ current: 0, runningQueueItemId: null, total: 10_000 });
  });
});

describe('getProjectQueueIndicatorState', () => {
  it('returns idle state for projects without open queue work', () => {
    expect(
      getProjectQueueIndicatorState({ isConnected: true, loadingModelsCount: 0, progress: null, queueItems: [] })
    ).toEqual({ hasOpenQueueWork: false, progressState: { kind: 'idle', value: 0 }, runningQueueItemId: null });
  });

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

  it('uses indeterminate state while models load for pending work', () => {
    const queueItems = [createQueueItem({ id: 'pending', status: 'pending', submittedAt: '2026-06-10T00:00:00.000Z' })];

    expect(
      getProjectQueueIndicatorState({ isConnected: true, loadingModelsCount: 1, progress: null, queueItems })
    ).toEqual({
      hasOpenQueueWork: true,
      progressState: { kind: 'indeterminate', value: null },
      runningQueueItemId: null,
    });
  });
});

describe('getQueueProgressBarValue', () => {
  it('returns an idle state when there is no active work', () => {
    expect(
      getQueueProgressBarState({ isConnected: true, isRunning: false, loadingModelsCount: 0, progress: null })
    ).toEqual({ kind: 'idle', value: 0 });
  });

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

  it('returns determinate progress when backend reports a non-zero percentage', () => {
    expect(
      getQueueProgressBarState({
        isConnected: true,
        isRunning: true,
        loadingModelsCount: 0,
        progress: { message: 'Denoising', percentage: 0.42 },
      })
    ).toEqual({ kind: 'determinate', value: 0.42 });
    expect(
      getQueueProgressBarValue({
        isConnected: true,
        isRunning: true,
        loadingModelsCount: 0,
        progress: { message: 'Denoising', percentage: 0.42 },
      })
    ).toBe(0.42);
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
