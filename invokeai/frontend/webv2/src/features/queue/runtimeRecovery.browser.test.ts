import { afterEach, describe, expect, it, vi } from 'vitest';

import type { QueueBackendPort, QueueItemReadModel } from './core/types';
import type { QueueHistoryCommands, QueueHistoryPort } from './runtime';

import { buildQueueItemOrigin } from './data/events';
import { createQueueRuntime } from './runtime';
import { runUtilityGraph } from './runtime/utilityGraph';

const disposers: Array<() => Promise<void>> = [];
afterEach(async () => {
  await Promise.all(disposers.splice(0).map((dispose) => dispose()));
});

type Candidate = ReturnType<QueueHistoryPort['listStagedResults']>[number];

/** Separate browser-runtime snapshots share only the server's account-wide delivery state. */
const openCanvasRuntime = (claimItemDelivery: QueueBackendPort['claimItemDelivery'], candidates: Candidate[] = []) => {
  const handlers = new Map<string, Set<(event: never) => void>>();
  const historyListeners = new Set<() => void>();
  const rerun: QueueItemReadModel = {
    batchId: 'retry-batch',
    createdAt: '2026-07-17T00:00:00Z',
    destination: 'canvas',
    id: 99,
    origin: buildQueueItemOrigin('retired-run', 'project'),
    resultImageNames: ['recovered.png'],
    retriedFromItemId: 41,
    sessionId: 'retry-session',
    status: 'completed',
    updatedAt: '2026-07-17T00:00:00Z',
  };
  const commands: QueueHistoryCommands = {
    markBackendCancelled: vi.fn(),
    markBackendSubmitted: vi.fn(),
    recordError: vi.fn(),
    recordSubmissionRetry: vi.fn(),
    refreshBackendData: vi.fn(),
    refreshBoardDestinations: vi.fn(),
    reopenForRetry: vi.fn(),
    restoreFromJournal: vi.fn(),
    routePartialResults: vi.fn(),
    routeResults: vi.fn(),
    setCancellationPending: vi.fn(),
    setConnectionStatus: vi.fn(),
    setLocalRecoveryState: vi.fn(),
    setStatus: vi.fn(),
    stageRetriedCanvasResults: vi.fn(({ backendItemId, projectId, queueItemId }) => {
      candidates.push({ backendItemId, projectId, queueItemId });
      historyListeners.forEach((listener) => listener());
    }),
  };
  const backend: QueueBackendPort = {
    cancelBatch: vi.fn(),
    cancelCurrentItem: vi.fn(),
    cancelItem: vi.fn(),
    cancelQueueItems: vi.fn(),
    cancelQueueItemsByBatchIds: vi.fn(),
    cancelScopedItems: vi.fn(),
    claimItemDelivery,
    clearFailedItems: vi.fn(),
    clearFinishedItems: vi.fn(),
    clearItems: vi.fn(),
    emit: vi.fn(),
    enqueueGenerate: vi.fn(),
    enqueueWorkflow: vi.fn(),
    getItem: vi.fn(),
    getResultImages: () =>
      Promise.resolve([
        {
          imageName: 'recovered.png',
          width: 64,
          height: 48,
          imageUrl: '/recovered.png',
          thumbnailUrl: '/recovered.png',
          queuedAt: '2026-07-17T00:00:00Z',
          sourceQueueItemId: 'retired-run',
        },
      ]),
    getResultVideoNames: () => Promise.resolve([]),
    getResultVideos: () => Promise.resolve([]),
    listItems: () => Promise.resolve([]),
    on: (event, handler) => {
      const listeners = handlers.get(event) ?? new Set();
      listeners.add(handler as (event: never) => void);
      handlers.set(event, listeners);
      return () => listeners.delete(handler as (event: never) => void);
    },
    onConnectionChange: (listener) => {
      listener('connected');
      return () => undefined;
    },
    pauseProcessor: vi.fn(),
    readCurrent: vi.fn(),
    readItemDetail: vi.fn(),
    readItemIds: vi.fn(),
    readItemsById: vi.fn((ids) => Promise.resolve(ids.includes(rerun.id) ? [rerun] : [])),
    readNext: vi.fn(),
    readStatus: vi.fn(),
    resumeProcessor: vi.fn(),
    retryItems: vi.fn(),
  };
  const runtime = createQueueRuntime({
    backend,
    destinations: { addImagesToGalleryBoard: vi.fn(), addVideosToGalleryBoard: vi.fn() },
    ensureTemplatesLoaded: vi.fn(),
    history: {
      commands,
      getSnapshot: () => ({
        connectionStatus: 'connected',
        isHydrated: true,
        projects: [{ id: 'project', queue: { items: [] } }],
      }),
      listStagedResults: () => candidates,
      subscribe: (listener) => {
        historyListeners.add(listener);
        return () => historyListeners.delete(listener);
      },
    },
    modelLoads: { completed: vi.fn(), reset: vi.fn(), started: vi.fn() },
    nodeExecution: {
      clearAll: vi.fn(),
      completed: vi.fn(),
      failed: vi.fn(),
      progress: vi.fn(),
      setOrigin: vi.fn(),
      settleRunning: vi.fn(),
      started: vi.fn(),
    },
  });
  disposers.push(runtime.dispose);
  runtime.start();
  return {
    candidates,
    commands,
    dispose: runtime.dispose,
    emitRerun: () => {
      handlers.get('queue_items_retried')?.forEach((handler) =>
        handler({
          queue_id: 'default',
          retried_item_ids: [41],
          retried_item_ids_by_user: {},
          retried_items: [{ item_id: 99, source_item_id: 41 }],
          timestamp: 0,
          user_ids: [],
        } as never)
      );
      handlers.get('queue_item_status_changed')?.forEach((handler) =>
        handler({
          created_at: '2026-07-17T00:00:00Z',
          destination: 'canvas',
          item_id: 99,
          origin: buildQueueItemOrigin('retired-run', 'project'),
          status: 'completed',
        } as never)
      );
    },
  };
};

describe('browser queue recovery', () => {
  it('reads authoritative completion before an explicit deadline and detaches native listeners', async () => {
    const listeners = new Set<unknown>();
    const cancel = vi.fn();
    const readItemStatus = vi.fn(() => Promise.resolve({ status: 'completed' as const }));
    const abort = new AbortController();
    const result = await runUtilityGraph({
      cancel,
      enqueue: () => Promise.resolve({ enqueued: 1, itemIds: [1] }),
      graph: { edges: [], id: 'utility', nodes: {} },
      hub: {
        on: (_event, listener) => {
          listeners.add(listener);
          return () => {
            listeners.delete(listener);
          };
        },
      },
      readItemStatus,
      reconcileCompletedOutput: () => Promise.resolve({ height: 48, imageName: 'recovered.png', width: 64 }),
      signal: abort.signal,
      timeoutMs: 10,
    });
    expect(result).toMatchObject({ imageName: 'recovered.png', height: 48, width: 64 });
    expect(readItemStatus).toHaveBeenCalledTimes(1);
    expect(listeners.size).toBe(0);
    abort.abort();
    await new Promise((resolve) => {
      setTimeout(resolve, 20);
    });
    expect(cancel).not.toHaveBeenCalled();
  });

  it('retains staging across a lost confirmation and prevents a stale second runtime from staging again', async () => {
    let delivered = false;
    let claimant: string | undefined;
    let confirmations = 0;
    const claim: QueueBackendPort['claimItemDelivery'] = (_id, action, requester) => {
      if (action === 'confirm') {
        delivered = true;
        if (++confirmations === 1) {
          return Promise.reject(new TypeError('server confirmed but its response was lost'));
        }
        return Promise.resolve({ kind: 'delivered' });
      }
      if (delivered) {
        return Promise.resolve({ kind: 'delivered' });
      }
      if (action === 'release') {
        claimant = undefined;
        return Promise.resolve({ kind: 'released' });
      }
      if (claimant && claimant !== requester) {
        return Promise.resolve({ kind: 'held', retryInMs: 10 });
      }
      claimant = requester;
      return Promise.resolve({ kind: 'claimed' });
    };
    const first = openCanvasRuntime(claim);
    first.emitRerun();
    await vi.waitFor(() =>
      expect(first.candidates).toEqual([{ backendItemId: 99, projectId: 'project', queueItemId: 'retired-run' }])
    );
    await vi.waitFor(() => expect(confirmations).toBe(1));
    const stale = openCanvasRuntime(claim);
    stale.emitRerun();
    await vi.waitFor(() => expect(confirmations).toBe(2), { timeout: 3_000 });
    expect(stale.candidates).toEqual([]);
    expect(first.commands.stageRetriedCanvasResults).toHaveBeenCalledTimes(1);

    await first.dispose();
    const restored = openCanvasRuntime(claim, structuredClone(first.candidates));
    await vi.waitFor(() => expect(confirmations).toBe(3));
    expect(restored.candidates).toEqual(first.candidates);
    expect(restored.commands.stageRetriedCanvasResults).not.toHaveBeenCalled();
    expect(restored.commands.recordError).not.toHaveBeenCalled();
  });
});
