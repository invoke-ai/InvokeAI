import type {
  EnqueueGenerateRequest,
  EnqueueGenerateResult,
  EnqueueWorkflowRequest,
  ImageDTO,
  QueueItemDTO,
} from '@workbench/generation/types';
import type { BackendConnectionStatus } from '@workbench/types';

import {
  cancelQueueItems,
  cancelQueueItemsByBatchIds,
  enqueueGenerateGraph,
  enqueueWorkflowGraph,
  getQueueItem,
  getQueueItemResultImages,
  listAllQueueItems,
  type QueueItemResultImageOptions,
} from '@workbench/generation/api';

import type { SocketHub } from './socketHub';

import {
  isTerminalBackendStatus,
  parseQueueItemOrigin,
  type InvocationCompleteEvent,
  type InvocationErrorEvent,
  type InvocationProgressEvent,
  type InvocationStartedEvent,
  type QueueItemStatusChangedEvent,
  type TerminalBackendQueueItemStatus,
} from './events';
import { ApiError } from './http';
import { modelLoadStore } from './modelLoadStore';
import { nodeExecutionStore, type NodeExecutionSink } from './nodeExecutionStore';
import { progressImageStore, type ProgressImageSink, type ProgressImageTarget } from './progressImageStore';
import { queueItemProgressStore, type QueueItemProgressSink } from './progressStore';
import { revealHoldStore, type RevealHoldSink } from './revealHoldStore';

const GALLERY_REFRESH_COALESCE_MS = 400;
const SAFETY_SWEEP_INTERVAL_MS = 30_000;
const TERMINAL_EVENT_BUFFER_LIMIT = 256;

export interface QueueCoordinatorApi {
  cancelQueueItems: typeof cancelQueueItems;
  cancelQueueItemsByBatchIds: typeof cancelQueueItemsByBatchIds;
  enqueueGenerateGraph: typeof enqueueGenerateGraph;
  enqueueWorkflowGraph: typeof enqueueWorkflowGraph;
  getQueueItem: typeof getQueueItem;
  getQueueItemResultImages: typeof getQueueItemResultImages;
  listAllQueueItems: typeof listAllQueueItems;
}

export interface QueueCoordinatorCallbacks {
  /** Fired when one backend item in a local batch completes, before the whole local batch necessarily settles. */
  onBackendItemComplete?(localQueueItemId: string, backendItemId: number): void | Promise<void>;
  /** Fired when one backend item in a local batch is canceled. */
  onBackendItemCancelled?(localQueueItemId: string, backendItemId: number): void;
  /** Coalesced signal that completed generations may have added gallery images. */
  onGalleryRefresh(): void;
}

type TerminalOutcome = { status: 'completed' } | { status: 'failed'; error: string } | { status: 'canceled' };

/** Thrown by `waitForResults` when the backend reports the run was canceled. */
export class QueueItemCancelledError extends Error {
  constructor(localQueueItemId: string) {
    super(`Queue item ${localQueueItemId} was canceled.`);
    this.name = 'QueueItemCancelledError';
  }
}

export interface ReconcileInput {
  id: string;
  status: 'pending' | 'running';
  backendItemIds?: number[];
  backendBatchId?: string;
}

export type ReconcileOutcome =
  /** A pending item the backend already accepted before the reload; do not re-enqueue. */
  | { kind: 'adopted'; backendItemIds: number[]; backendBatchId?: string }
  /** A running item whose backend items were found again; its results are awaitable. */
  | { kind: 'resumed' }
  /** A running item whose backend items no longer exist (queue cleared or pruned). */
  | { kind: 'missing' }
  /** A pending item the backend has never seen; submit it normally. */
  | { kind: 'enqueue' };

export interface CancelRunRequest {
  backendBatchId?: string;
  backendItemIds?: number[];
}

export interface QueueCoordinator {
  connect(): void;
  dispose(): void;
  /**
   * Match persisted pending/running queue items against the live backend queue
   * so a reload neither double-submits nor orphans work. Adopted and resumed
   * items are tracked and can be awaited with `waitForResults`.
   */
  reconcile(items: ReconcileInput[]): Promise<Map<string, ReconcileOutcome>>;
  /** Enqueue a generate batch and track its backend items for event-driven settlement. */
  submitGenerate(localQueueItemId: string, request: EnqueueGenerateRequest): Promise<EnqueueGenerateResult>;
  /** Enqueue a compiled workflow graph and track its backend items the same way. */
  submitWorkflow(localQueueItemId: string, request: EnqueueWorkflowRequest): Promise<EnqueueGenerateResult>;
  /**
   * Resolve once every backend item of the run reaches a terminal status —
   * driven by socket events, with a slow safety sweep as the only polling.
   * Resolves with the result images, throws on failure, and throws
   * `QueueItemCancelledError` on backend-side cancellation.
   */
  waitForResults(
    localQueueItemId: string,
    queuedAt: string,
    options?: QueueItemResultImageOptions
  ): Promise<ImageDTO[]>;
  cancelRun(request: CancelRunRequest): Promise<void>;
}

interface RunState {
  backendItemIds: number[];
  backendBatchId?: string;
  outcomePromises: Promise<TerminalOutcome>[];
}

interface RunProgressState {
  activeBackendItemId?: number;
  backendItemIds: number[];
  cancelledBackendItemIds: Set<number>;
  completedBackendItemIds: Set<number>;
  message: string;
  percentage: number | null;
}

interface WaitState {
  localQueueItemId: string;
  settle: (outcome: TerminalOutcome) => void;
}

const defaultApi: QueueCoordinatorApi = {
  cancelQueueItems,
  cancelQueueItemsByBatchIds,
  enqueueGenerateGraph,
  enqueueWorkflowGraph,
  getQueueItem,
  getQueueItemResultImages,
  listAllQueueItems,
};

const toTerminalOutcome = (
  status: TerminalBackendQueueItemStatus,
  error?: string | null,
  errorType?: string | null
): TerminalOutcome => {
  if (status === 'completed') {
    return { status: 'completed' };
  }

  if (status === 'failed') {
    return { error: error ?? errorType ?? 'Generation failed.', status: 'failed' };
  }

  return { status: 'canceled' };
};

export const createQueueCoordinator = (
  callbacks: QueueCoordinatorCallbacks,
  options: {
    hub: Pick<SocketHub, 'on' | 'emit' | 'onConnectionChange'>;
    api?: Partial<QueueCoordinatorApi>;
    galleryRefreshCoalesceMs?: number;
    nodeExecution?: NodeExecutionSink;
    progress?: QueueItemProgressSink;
    progressImage?: ProgressImageSink;
    revealHold?: RevealHoldSink;
    sweepIntervalMs?: number;
  }
): QueueCoordinator => {
  const api: QueueCoordinatorApi = { ...defaultApi, ...options.api };
  const hub = options.hub;
  const progress = options.progress ?? queueItemProgressStore;
  const nodeExecution = options.nodeExecution ?? nodeExecutionStore;
  const progressImage = options.progressImage ?? progressImageStore;
  const revealHold = options.revealHold ?? revealHoldStore;
  const galleryRefreshCoalesceMs = options.galleryRefreshCoalesceMs ?? GALLERY_REFRESH_COALESCE_MS;
  const sweepIntervalMs = options.sweepIntervalMs ?? SAFETY_SWEEP_INTERVAL_MS;

  const runs = new Map<string, RunState>();
  const runProgress = new Map<string, RunProgressState>();
  const waits = new Map<number, WaitState>();
  /**
   * Terminal events that arrived for items nobody tracks yet. Closes the race
   * where a very fast generation finishes between `enqueue_batch` resolving
   * and the run being registered.
   */
  const recentTerminalOutcomes = new Map<number, TerminalOutcome>();

  const detachers: Array<() => void> = [];
  let isAttached = false;
  let isDisposed = false;
  let galleryRefreshTimer: ReturnType<typeof setTimeout> | null = null;
  let sweepTimer: ReturnType<typeof setInterval> | null = null;
  let isSweeping = false;

  const scheduleGalleryRefresh = (): void => {
    if (isDisposed || galleryRefreshTimer !== null) {
      return;
    }

    galleryRefreshTimer = setTimeout(() => {
      galleryRefreshTimer = null;
      callbacks.onGalleryRefresh();
    }, galleryRefreshCoalesceMs);
  };

  const bufferTerminalOutcome = (backendItemId: number, outcome: TerminalOutcome): void => {
    recentTerminalOutcomes.delete(backendItemId);
    recentTerminalOutcomes.set(backendItemId, outcome);

    while (recentTerminalOutcomes.size > TERMINAL_EVENT_BUFFER_LIMIT) {
      const oldestId = recentTerminalOutcomes.keys().next().value;

      if (oldestId === undefined) {
        return;
      }

      recentTerminalOutcomes.delete(oldestId);
    }
  };

  const publishRunProgress = (localQueueItemId: string): void => {
    const state = runProgress.get(localQueueItemId);

    if (!state) {
      return;
    }

    const terminalBackendItemIds = new Set([...state.completedBackendItemIds, ...state.cancelledBackendItemIds]);
    const activeBackendItemId =
      state.activeBackendItemId !== undefined && !terminalBackendItemIds.has(state.activeBackendItemId)
        ? state.activeBackendItemId
        : undefined;
    const activeItemIndex = activeBackendItemId
      ? state.backendItemIds.indexOf(activeBackendItemId) + 1
      : Math.min(terminalBackendItemIds.size + 1, state.backendItemIds.length);

    progress.set(localQueueItemId, {
      activeItemIndex: Math.max(1, activeItemIndex),
      completedItemCount: terminalBackendItemIds.size,
      message: state.message,
      percentage: state.percentage,
      totalItemCount: state.backendItemIds.length,
    });
  };

  const getProgressImageTarget = (localQueueItemId: string, backendItemId: number): ProgressImageTarget => {
    const backendItemIds = runProgress.get(localQueueItemId)?.backendItemIds ?? [backendItemId];
    const itemIndex = backendItemIds.indexOf(backendItemId);

    return { itemIndex: itemIndex === -1 ? 1 : itemIndex + 1, queueItemId: localQueueItemId };
  };

  const settleWait = (backendItemId: number, outcome: TerminalOutcome): void => {
    const wait = waits.get(backendItemId);

    if (!wait) {
      bufferTerminalOutcome(backendItemId, outcome);
      return;
    }

    waits.delete(backendItemId);
    const progressTarget = getProgressImageTarget(wait.localQueueItemId, backendItemId);
    const clearProgressImage = (): void => progressImage.clear(progressTarget);
    const state = runProgress.get(wait.localQueueItemId);

    if (state) {
      state.activeBackendItemId = undefined;
      if (outcome.status === 'completed') {
        state.completedBackendItemIds.add(backendItemId);
      }
      if (outcome.status === 'canceled') {
        state.cancelledBackendItemIds.add(backendItemId);
      }
      state.message = '';
      state.percentage = null;
      publishRunProgress(wait.localQueueItemId);
    }

    if (outcome.status === 'completed') {
      const routingPromise = callbacks.onBackendItemComplete?.(wait.localQueueItemId, backendItemId);
      // Reveal window: while sibling items keep generating, hold off the next
      // live frames briefly so the just-finished result is actually seen.
      const settleCompleted = (): void => {
        clearProgressImage();

        if (waits.size > 0) {
          revealHold.arm();
        }
      };

      if (routingPromise) {
        void Promise.resolve(routingPromise)
          .finally(settleCompleted)
          .catch(() => undefined);
      } else {
        settleCompleted();
      }
    }

    if (outcome.status === 'canceled') {
      clearProgressImage();
      callbacks.onBackendItemCancelled?.(wait.localQueueItemId, backendItemId);
    }

    if (outcome.status === 'failed') {
      clearProgressImage();
    }

    wait.settle(outcome);
  };

  const settleFromQueueItem = (queueItem: QueueItemDTO): void => {
    if (queueItem.status !== 'pending' && queueItem.status !== 'in_progress') {
      settleWait(queueItem.item_id, toTerminalOutcome(queueItem.status, queueItem.error_message, queueItem.error_type));
    }
  };

  const isTrackedEvent = (event: { item_id: number }): boolean => waits.has(event.item_id);

  const trackBackendItem = (localQueueItemId: string, backendItemId: number): Promise<TerminalOutcome> => {
    const bufferedOutcome = recentTerminalOutcomes.get(backendItemId);

    if (bufferedOutcome) {
      recentTerminalOutcomes.delete(backendItemId);

      return Promise.resolve(bufferedOutcome);
    }

    return new Promise<TerminalOutcome>((settle) => {
      waits.set(backendItemId, { localQueueItemId, settle });
    });
  };

  const beginRun = (localQueueItemId: string, backendItemIds: number[], backendBatchId?: string): void => {
    runs.set(localQueueItemId, {
      backendBatchId,
      backendItemIds,
      outcomePromises: backendItemIds.map((backendItemId) => trackBackendItem(localQueueItemId, backendItemId)),
    });

    runProgress.set(localQueueItemId, {
      backendItemIds,
      cancelledBackendItemIds: new Set(),
      completedBackendItemIds: new Set(),
      message: '',
      percentage: null,
    });
    publishRunProgress(localQueueItemId);
  };

  /** Slow safety net for events lost to disconnects; runs on reconnect and on a long interval. */
  const sweep = async (): Promise<void> => {
    if (isSweeping || waits.size === 0) {
      return;
    }

    isSweeping = true;

    try {
      await Promise.all(
        [...waits.keys()].map(async (backendItemId) => {
          try {
            settleFromQueueItem(await api.getQueueItem(backendItemId));
          } catch (error) {
            if (error instanceof ApiError && error.status === 404) {
              settleWait(backendItemId, {
                error: `Queue item ${backendItemId} is no longer on the backend queue.`,
                status: 'failed',
              });
            }
          }
        })
      );
    } finally {
      isSweeping = false;
    }
  };

  const handleStatusChanged = (event: QueueItemStatusChangedEvent): void => {
    if (!isTerminalBackendStatus(event.status)) {
      return;
    }

    if (!isTrackedEvent(event)) {
      bufferTerminalOutcome(event.item_id, toTerminalOutcome(event.status, event.error_message, event.error_type));
      return;
    }

    nodeExecution.settleRunning();
    settleWait(event.item_id, toTerminalOutcome(event.status, event.error_message, event.error_type));

    if (event.status === 'completed') {
      scheduleGalleryRefresh();
    }
  };

  const handleProgress = (event: InvocationProgressEvent): void => {
    const wait = waits.get(event.item_id);

    if (!wait) {
      return;
    }

    nodeExecution.progress(event.invocation_source_id, event.percentage, event.message);

    if (event.image?.dataURL) {
      progressImage.set(
        { dataUrl: event.image.dataURL, height: event.image.height, width: event.image.width },
        getProgressImageTarget(wait.localQueueItemId, event.item_id)
      );
    }

    const state = runProgress.get(wait.localQueueItemId);

    if (state) {
      state.activeBackendItemId = event.item_id;
      state.message = event.message;
      state.percentage = event.percentage;
      publishRunProgress(wait.localQueueItemId);
    } else {
      progress.set(wait.localQueueItemId, { message: event.message, percentage: event.percentage });
    }
  };

  /**
   * React to the shared socket's connection lifecycle. The hub owns the socket,
   * the connection store, and `modelLoadStore.reset()`; the coordinator only
   * clears its generation-scoped stores and, on (re)connect, schedules a gallery
   * refresh and sweeps for events missed while disconnected.
   */
  const handleConnectionChange = (status: BackendConnectionStatus): void => {
    progress.clearAll?.();
    nodeExecution.clearAll();

    if (status === 'connected') {
      scheduleGalleryRefresh();
      void sweep();
    }
  };

  /** Attach generation listeners to the shared socket hub. */
  const connect = (): void => {
    if (isDisposed || isAttached) {
      return;
    }

    isAttached = true;

    detachers.push(
      hub.on('queue_item_status_changed', handleStatusChanged),
      hub.on('invocation_progress', handleProgress),
      hub.on('invocation_started', (event: InvocationStartedEvent) => {
        if (!isTrackedEvent(event)) {
          return;
        }

        nodeExecution.started(event);
      }),
      hub.on('invocation_complete', (event: InvocationCompleteEvent) => {
        if (!isTrackedEvent(event)) {
          return;
        }

        nodeExecution.completed(event);
      }),
      hub.on('invocation_error', (event: InvocationErrorEvent) => {
        if (!isTrackedEvent(event)) {
          return;
        }

        nodeExecution.failed(event);
      }),
      hub.on('model_load_started', (payload: never) => {
        modelLoadStore.started(payload);
      }),
      hub.on('model_load_complete', (payload: never) => {
        modelLoadStore.completed(payload);
      })
    );

    // Fires synchronously with the current status, so attaching after the hub
    // has already connected still triggers the initial clear + sweep.
    detachers.push(hub.onConnectionChange(handleConnectionChange));

    sweepTimer = setInterval(() => {
      void sweep();
    }, sweepIntervalMs);
  };

  /** Detach generation listeners; the hub keeps the socket alive. */
  const dispose = (): void => {
    isDisposed = true;

    for (const detach of detachers) {
      detach();
    }

    detachers.length = 0;

    if (galleryRefreshTimer !== null) {
      clearTimeout(galleryRefreshTimer);
      galleryRefreshTimer = null;
    }

    if (sweepTimer !== null) {
      clearInterval(sweepTimer);
      sweepTimer = null;
    }
  };

  const reconcile = async (items: ReconcileInput[]): Promise<Map<string, ReconcileOutcome>> => {
    const outcomes = new Map<string, ReconcileOutcome>();

    if (items.length === 0) {
      return outcomes;
    }

    const backendItems = items.every((item) => item.backendItemIds?.length)
      ? (
          await Promise.all(
            items
              .flatMap((item) => item.backendItemIds ?? [])
              .map(async (itemId) => {
                try {
                  return await api.getQueueItem(itemId);
                } catch (error) {
                  if (error instanceof ApiError && error.status === 404) {
                    return undefined;
                  }

                  throw error;
                }
              })
          )
        ).filter((item) => item !== undefined)
      : await api.listAllQueueItems();
    const backendItemsById = new Map(backendItems.map((item) => [item.item_id, item]));
    const backendItemsByLocalId = new Map<string, QueueItemDTO[]>();

    for (const backendItem of backendItems) {
      const localQueueItemId = parseQueueItemOrigin(backendItem.origin);

      if (localQueueItemId) {
        backendItemsByLocalId.set(localQueueItemId, [
          ...(backendItemsByLocalId.get(localQueueItemId) ?? []),
          backendItem,
        ]);
      }
    }

    for (const item of items) {
      const knownBackendItems = item.backendItemIds?.length
        ? item.backendItemIds.map((backendItemId) => backendItemsById.get(backendItemId))
        : (backendItemsByLocalId.get(item.id) ?? []);
      const foundBackendItems = knownBackendItems.filter((backendItem) => backendItem !== undefined);

      if (foundBackendItems.length !== knownBackendItems.length || foundBackendItems.length === 0) {
        // A pending item with no backend trace was never accepted and is safe
        // to submit; a running item with (partially) vanished backend items is
        // unrecoverable.
        outcomes.set(item.id, item.status === 'pending' ? { kind: 'enqueue' } : { kind: 'missing' });
        continue;
      }

      const backendItemIds = foundBackendItems.map((backendItem) => backendItem.item_id);
      const backendBatchId = item.backendBatchId ?? foundBackendItems[0]?.batch_id;

      beginRun(item.id, backendItemIds, backendBatchId);

      for (const backendItem of foundBackendItems) {
        settleFromQueueItem(backendItem);
      }

      outcomes.set(
        item.id,
        item.status === 'running' ? { kind: 'resumed' } : { backendBatchId, backendItemIds, kind: 'adopted' }
      );
    }

    return outcomes;
  };

  const submitGenerate = async (
    localQueueItemId: string,
    request: EnqueueGenerateRequest
  ): Promise<EnqueueGenerateResult> => {
    const result = await api.enqueueGenerateGraph(request);

    if (result.enqueued === 0) {
      throw new Error('The backend queue did not accept this generation. The queue may be full.');
    }

    if (result.requested !== result.enqueued) {
      throw new Error(`The backend queue accepted ${result.enqueued} of ${result.requested} requested items.`);
    }

    beginRun(localQueueItemId, result.itemIds, result.batchId);

    return result;
  };

  const submitWorkflow = async (
    localQueueItemId: string,
    request: EnqueueWorkflowRequest
  ): Promise<EnqueueGenerateResult> => {
    const result = await api.enqueueWorkflowGraph(request);

    if (result.enqueued === 0) {
      throw new Error('The backend queue did not accept this workflow. The queue may be full.');
    }

    if (result.requested !== result.enqueued) {
      throw new Error(`The backend queue accepted ${result.enqueued} of ${result.requested} requested items.`);
    }

    beginRun(localQueueItemId, result.itemIds, result.batchId);

    return result;
  };

  const waitForResults = async (
    localQueueItemId: string,
    queuedAt: string,
    options?: QueueItemResultImageOptions
  ): Promise<ImageDTO[]> => {
    const run = runs.get(localQueueItemId);

    if (!run) {
      throw new Error(`Queue item ${localQueueItemId} has no tracked backend run.`);
    }

    try {
      const outcomes = await Promise.all(run.outcomePromises);
      const failure = outcomes.find((outcome) => outcome.status === 'failed');

      if (failure) {
        throw new Error(failure.error);
      }

      const completedBackendItemIds = run.backendItemIds.filter(
        (_backendItemId, index) => outcomes[index]?.status === 'completed'
      );

      if (completedBackendItemIds.length === 0 && outcomes.some((outcome) => outcome.status === 'canceled')) {
        throw new QueueItemCancelledError(localQueueItemId);
      }

      const imagesPerItem = await Promise.all(
        completedBackendItemIds.map((backendItemId) =>
          options
            ? api.getQueueItemResultImages(backendItemId, localQueueItemId, queuedAt, options)
            : api.getQueueItemResultImages(backendItemId, localQueueItemId, queuedAt)
        )
      );

      return imagesPerItem.flat();
    } finally {
      runs.delete(localQueueItemId);
      runProgress.delete(localQueueItemId);
      progress.clear(localQueueItemId);
    }
  };

  const cancelRun = async ({ backendBatchId, backendItemIds }: CancelRunRequest): Promise<void> => {
    try {
      if (backendItemIds?.length) {
        await api.cancelQueueItems(backendItemIds);
        return;
      }

      if (backendBatchId) {
        await api.cancelQueueItemsByBatchIds([backendBatchId]);
      }
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) {
        return;
      }

      throw error;
    }
  };

  return { cancelRun, connect, dispose, reconcile, submitGenerate, submitWorkflow, waitForResults };
};
