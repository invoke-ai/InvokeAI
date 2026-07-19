import type { QueueItem } from '@features/queue/core/historyTypes';
import type { QueueBackendPort } from '@features/queue/core/types';

import { buildQueueItemOrigin } from '@features/queue/data/events';
import { describe, expect, it, vi } from 'vitest';

import { createQueueRuntime, type QueueHistoryCommands } from './runtime';

const createPendingQueueItem = (): QueueItem => ({
  cancellable: true,
  id: 'local-queue-item',
  snapshot: {
    backendSubmission: {
      batchCount: 1,
      graph: { edges: [], id: 'backend-graph', nodes: {} },
      kind: 'generate',
      negativePrompt: '',
      negativePromptNodeId: 'negative_prompt',
      positivePrompt: '',
      positivePromptNodeId: 'positive_prompt',
      seed: 0,
      seedNodeId: 'seed',
      shouldRandomizeSeed: false,
    },
    destination: 'canvas',
    filterIntermediateResults: false,
    galleryBoardId: null,
    graph: { id: 'graph', label: 'Generate' },
    presentation: { batchCount: 1, height: 1024, width: 1024 },
    resultNodeIds: ['canvas_output'],
    sourceId: 'generate',
    submittedAt: '2026-07-17T00:00:00.000Z',
  },
  status: 'pending',
});

describe('queue runtime', () => {
  it('adopts a persisted backend run before submission so reload cannot duplicate it', async () => {
    const queueItem = createPendingQueueItem();
    const project = { id: 'project-1', queue: { items: [queueItem] } };
    const enqueueGenerate = vi.fn();
    const listItems = vi.fn().mockResolvedValue([
      {
        batchId: 'backend-batch',
        id: 77,
        origin: buildQueueItemOrigin(queueItem.id, project.id),
        status: 'in_progress',
      },
    ]);
    const backend: QueueBackendPort = {
      cancelCurrentItem: vi.fn(),
      cancelItem: vi.fn(),
      cancelQueueItems: vi.fn(),
      cancelQueueItemsByBatchIds: vi.fn(),
      cancelScopedItems: vi.fn(),
      clearFailedItems: vi.fn(),
      clearItems: vi.fn(),
      emit: vi.fn(),
      enqueueGenerate,
      enqueueWorkflow: vi.fn(),
      getItem: vi.fn(),
      getResultImages: vi.fn(),
      listItems,
      on: vi.fn(() => vi.fn()),
      onConnectionChange: vi.fn((listener) => {
        listener('connected');
        return vi.fn();
      }),
      pauseProcessor: vi.fn(),
      readCurrent: vi.fn(),
      readItemIds: vi.fn(),
      readItemsById: vi.fn(),
      readNext: vi.fn(),
      readStatus: vi.fn(),
      resumeProcessor: vi.fn(),
      retryItems: vi.fn(),
    };
    const commands: QueueHistoryCommands = {
      markBackendCancelled: vi.fn(),
      markBackendSubmitted: ({ backendBatchId, backendItemIds }) => {
        queueItem.backendBatchId = backendBatchId;
        queueItem.backendItemIds = backendItemIds;
        queueItem.status = 'running';
      },
      recordError: vi.fn(),
      refreshBackendData: vi.fn(),
      routePartialResults: vi.fn(),
      routeResults: vi.fn(),
      setConnectionStatus: vi.fn(),
      setStatus: vi.fn(),
    };
    const runtime = createQueueRuntime({
      backend,
      destinations: { addImagesToGalleryBoard: vi.fn() },
      ensureTemplatesLoaded: vi.fn(),
      history: {
        commands,
        getSnapshot: () => ({ connectionStatus: 'connected', isHydrated: true, projects: [project] }),
        subscribe: vi.fn(() => vi.fn()),
      },
      modelLoads: {
        completed: vi.fn(),
        reset: vi.fn(),
        started: vi.fn(),
      },
      nodeExecution: {
        clearAll: vi.fn(),
        completed: vi.fn(),
        failed: vi.fn(),
        progress: vi.fn(),
        settleRunning: vi.fn(),
        started: vi.fn(),
      },
    });

    runtime.start();

    await vi.waitFor(() => {
      expect(queueItem).toMatchObject({
        backendBatchId: 'backend-batch',
        backendItemIds: [77],
        status: 'running',
      });
    });
    expect(listItems).toHaveBeenCalledTimes(1);
    expect(enqueueGenerate).not.toHaveBeenCalled();

    runtime.dispose();
  });
});
