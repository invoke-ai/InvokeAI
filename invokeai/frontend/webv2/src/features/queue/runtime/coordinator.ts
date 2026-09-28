import type {
  QueueBackendItem,
  QueueBackendPort,
  QueueEnqueueGenerateRequest,
  QueueEnqueueResult,
  QueueEnqueueWorkflowRequest,
  QueueResultImage,
  QueueResultImageOptions,
  QueueRunOrigin,
  TerminalQueueItemStatus,
} from '@features/queue/core/types';
import type { BackendConnectionStatus } from '@platform/transport/types';

import {
  activeProgressTargetStore,
  type ActiveProgressTargetSink,
} from '@features/queue/data/activeProgressTargetStore';
import {
  isTerminalBackendStatus,
  parseQueueItemOrigin,
  parseQueueItemOriginProjectId,
  type InvocationCompleteEvent,
  type InvocationErrorEvent,
  type InvocationProgressEvent,
  type InvocationStartedEvent,
  type QueueItemStatusChangedEvent,
  type QueueItemsCanceledEvent,
} from '@features/queue/data/events';
import {
  progressImageStore,
  type ProgressImageSink,
  type ProgressImageTarget,
} from '@features/queue/data/progressImageStore';
import { queueItemProgressStore, type QueueItemProgressSink } from '@features/queue/data/progressStore';
import { mapWithConcurrency } from '@platform/core/concurrency';
import { createLogger } from '@platform/logging/logger';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';

const GALLERY_REFRESH_COALESCE_MS = 400;
const SAFETY_SWEEP_INTERVAL_MS = 30_000;
/** Node-level detail supports the terminal queue-item failure the history owner records. */
const coordinatorLogger = createLogger({ area: 'coordinator', namespace: 'queue' });

const TERMINAL_EVENT_BUFFER_LIMIT = 256;
const NODE_EVENT_BUFFER_ITEM_LIMIT = 64;
const NODE_EVENT_BUFFER_EVENTS_PER_ITEM = 512;
const PROGRESS_EVENT_BUFFER_ITEM_LIMIT = 64;
const BACKEND_READ_CONCURRENCY = 16;
const FRAME_GATES_PER_WAIT_LIMIT = 64;

/**
 * App injects Models' model-load activity sink so Queue can report socket-derived activity without importing its
 * owner.
 */
export interface QueueModelLoadPort {
  completed(payload: unknown): void;
  reset(): void;
  started(payload: unknown): void;
}

export interface QueueNodeExecutionPort {
  /** Forgets every node and the origin of the run they belonged to. */
  clearAll(): void;
  /** Names the project workflow whose run the store now reflects; null when it cannot be attributed. */
  setOrigin(origin: QueueRunOrigin | null): void;
  completed(event: InvocationCompleteEvent): void;
  failed(event: InvocationErrorEvent): void;
  progress(nodeId: string, percentage: number | null, message: string): void;
  settleRunning(nodeIds: Iterable<string>, outcome: 'completed' | 'failed' | 'canceled', error?: string): void;
  started(event: InvocationStartedEvent): void;
}

type NodeEvent = (
  | { kind: 'started'; event: InvocationStartedEvent }
  | { kind: 'completed'; event: InvocationCompleteEvent }
  | { kind: 'failed'; event: InvocationErrorEvent }
) & {
  /** Receipt order; a replayed event keeps the order it arrived in. */
  sequence: number;
};

export type QueueCoordinatorBackendPort = Pick<
  QueueBackendPort,
  | 'cancelQueueItems'
  | 'cancelQueueItemsByBatchIds'
  | 'emit'
  | 'enqueueGenerate'
  | 'enqueueWorkflow'
  | 'getItem'
  | 'getEnqueueReceipt'
  | 'getResultImages'
  | 'listItems'
  | 'on'
  | 'onConnectionChange'
  | 'readProgressPreviews'
>;

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

/** Thrown when an enqueue response definitively reports zero accepted items. */
export class QueueEnqueueNotAcceptedError extends Error {
  constructor(workKind: 'generation' | 'workflow') {
    super(`The backend queue did not accept this ${workKind}. The queue may be full.`);
    this.name = 'QueueEnqueueNotAcceptedError';
  }
}

export interface ReconcileInput {
  id: string;
  projectId?: string;
  status: 'pending' | 'running';
  backendItemIds?: number[];
  backendBatchId?: string;
  /** Persisted compiled graph contains a Call Saved Workflow node. */
  hasWorkflowCall?: boolean;
  origin?: QueueRunOrigin;
}

export type ReconcileOutcome =
  /** A pending item the backend already accepted before the reload; do not re-enqueue. */
  | { kind: 'adopted'; backendItemIds: number[]; backendBatchId?: string; missingBackendItemIds?: number[] }
  /** A running item whose backend items were found again; its results are awaitable. */
  | { kind: 'resumed'; backendItemIds?: number[]; missingBackendItemIds?: number[] }
  /** A running item whose backend items no longer exist (queue cleared or pruned). */
  | { kind: 'missing'; backendItemIds?: number[]; backendBatchId?: string }
  /** A pending item the backend has never seen; submit it normally. */
  | { kind: 'enqueue' };

export interface CancelRunRequest {
  backendBatchId?: string;
  backendItemIds?: number[];
}

export interface QueueCoordinator {
  connect(): void;
  detachRun(localQueueItemId: string): void;
  dispose(): void;
  /** Reconcile persisted work with backend items to avoid duplicate submission or orphaned runs after reload. */
  reconcile(items: ReconcileInput[]): Promise<Map<string, ReconcileOutcome>>;
  /** Enqueue a generate batch and track its backend items for event-driven settlement. */
  submitGenerate(localQueueItemId: string, request: QueueEnqueueGenerateRequest): Promise<QueueEnqueueResult>;
  /** Enqueue a compiled workflow graph and track its backend items the same way. */
  submitWorkflow(
    localQueueItemId: string,
    request: QueueEnqueueWorkflowRequest,
    origin?: QueueRunOrigin
  ): Promise<QueueEnqueueResult>;
  /**
   * Await every backend item's terminal state via sockets and safety sweeps; return images or throw
   * failure/cancellation.
   */
  waitForResults(
    localQueueItemId: string,
    queuedAt: string,
    options?: QueueResultImageOptions
  ): Promise<QueueResultImage[]>;
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
  frameGates: Map<number, { revision: number | null; sessionId: string }>;
  hasWorkflowCall: boolean;
  localQueueItemId: string;
  origin: QueueRunOrigin | null;
  settle: (outcome: TerminalOutcome) => void;
}

interface RunTrackingContext {
  hasWorkflowCall: boolean;
  origin: QueueRunOrigin | null;
}

const toTerminalOutcome = (
  status: TerminalQueueItemStatus,
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
    /** One adapter owns Queue HTTP commands and realtime events. */
    backend: QueueCoordinatorBackendPort;
    activeProgressTarget?: ActiveProgressTargetSink;
    galleryRefreshCoalesceMs?: number;
    modelLoads: QueueModelLoadPort;
    nodeExecution: QueueNodeExecutionPort;
    progress?: QueueItemProgressSink;
    progressImage?: ProgressImageSink;
    sweepIntervalMs?: number;
  }
): QueueCoordinator => {
  const owner = captureAccountScope();
  const backend = options.backend;
  const activeProgressTarget = options.activeProgressTarget ?? activeProgressTargetStore;
  const progress = options.progress ?? queueItemProgressStore;
  const modelLoads = options.modelLoads;
  const nodeExecution = options.nodeExecution;
  const progressImage = options.progressImage ?? progressImageStore;
  const galleryRefreshCoalesceMs = options.galleryRefreshCoalesceMs ?? GALLERY_REFRESH_COALESCE_MS;
  const sweepIntervalMs = options.sweepIntervalMs ?? SAFETY_SWEEP_INTERVAL_MS;

  const runs = new Map<string, RunState>();
  const runProgress = new Map<string, RunProgressState>();
  const waits = new Map<number, WaitState>();
  const markWorkflowCallRoot = (event: {
    item_id: number;
    root_item_id?: number | null;
    workflow_call_parent_source_id?: string | null;
  }): void => {
    const rootItemId = event.root_item_id;
    if (rootItemId !== null && rootItemId !== undefined && rootItemId !== event.item_id) {
      const wait = waits.get(rootItemId);
      if (wait && event.workflow_call_parent_source_id !== null && event.workflow_call_parent_source_id !== undefined) {
        wait.hasWorkflowCall = true;
      }
    }
  };
  const clearFrameGate = (itemId: number): void => {
    for (const wait of waits.values()) {
      if (wait.frameGates.delete(itemId)) {
        return;
      }
    }
  };
  /**
   * Terminal events that arrived for items nobody tracks yet. Closes the race
   * where a very fast generation finishes between `enqueue_batch` resolving
   * and the run being registered.
   */
  const recentTerminalOutcomes = new Map<number, TerminalOutcome>();
  /**
   * Node lifecycle events for items nobody tracks yet, replayed once the item
   * is registered: a fast run's nodes start and finish while `enqueue_batch`
   * is still resolving.
   */
  const pendingNodeEvents = new Map<number, NodeEvent[]>();
  /** Latest preview for items whose enqueue response has not registered them yet. */
  const pendingProgressEvents = new Map<number, InvocationProgressEvent>();
  /** Nodes each backend item has driven; settled with the item's terminal outcome. */
  const nodeIdsByBackendItem = new Map<number, Set<string>>();
  /** The root backend item whose node events the execution store currently reflects. */
  let nodeExecutionRootItemId: number | null = null;
  let nodeExecutionRootItemSequence = 0;
  let nodeEventSequence = 0;
  /** Enqueue requests awaiting a response; early node events and previews are buffered while one is in flight. */
  let inFlightSubmissions = 0;
  const latestStatusSequences = new Map<number, number>();
  const getTrackedBackendItemId = (event: { item_id: number; root_item_id?: number | null }): number =>
    event.root_item_id ?? event.item_id;

  const getNodeSourceId = (event: { invocation_source_id: string; workflow_call_parent_source_id?: string | null }) =>
    event.workflow_call_parent_source_id ?? event.invocation_source_id;

  const routeNodeEvent = <T extends NodeEvent['event']>(
    event: T,
    backendItemId: number,
    invocationSourceId: string
  ): T => {
    if (backendItemId === event.item_id && invocationSourceId === event.invocation_source_id) {
      return event;
    }

    return { ...event, invocation_source_id: invocationSourceId, item_id: backendItemId } as T;
  };

  const detachers: Array<() => void> = [];
  let isAttached = false;
  let isDisposed = false;
  let galleryRefreshTimer: ReturnType<typeof setTimeout> | null = null;
  let sweepTimer: ReturnType<typeof setInterval> | null = null;
  let isSweeping = false;
  let isSweepRequested = false;
  const isActive = (): boolean => !isDisposed && isAccountScopeCurrent(owner);

  const scheduleGalleryRefresh = (): void => {
    if (!isActive() || galleryRefreshTimer !== null) {
      return;
    }

    galleryRefreshTimer = setTimeout(() => {
      galleryRefreshTimer = null;

      if (isActive()) {
        callbacks.onGalleryRefresh();
      }
    }, galleryRefreshCoalesceMs);
  };

  const bufferTerminalOutcome = (backendItemId: number, outcome: TerminalOutcome): void => {
    if (!isActive()) {
      return;
    }

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
    if (!isActive()) {
      return;
    }

    const state = runProgress.get(localQueueItemId);

    if (!state) {
      return;
    }

    const terminalBackendItemIds = new Set([...state.completedBackendItemIds, ...state.cancelledBackendItemIds]);
    const activeBackendItemId =
      state.activeBackendItemId !== undefined && !terminalBackendItemIds.has(state.activeBackendItemId)
        ? state.activeBackendItemId
        : undefined;
    // activeItemIndex means "slot currently executing" — omitted while the
    // item merely waits in the queue, so idle slots never present as live.
    const activeItemIndex =
      activeBackendItemId !== undefined
        ? Math.max(1, state.backendItemIds.indexOf(activeBackendItemId) + 1)
        : undefined;

    progress.set(localQueueItemId, {
      activeItemIndex,
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

  const settleNodes = (backendItemId: number, outcome: TerminalOutcome): void => {
    const nodeIds = nodeIdsByBackendItem.get(backendItemId);

    nodeIdsByBackendItem.delete(backendItemId);

    // A late settle for an item the store no longer reflects must not touch the current run's nodes.
    if (nodeIds && isActive() && nodeExecutionRootItemId === backendItemId) {
      if (outcome.status === 'failed') {
        nodeExecution.settleRunning(nodeIds, outcome.status, outcome.error);
      } else {
        nodeExecution.settleRunning(nodeIds, outcome.status);
      }
    }
  };

  /** Backend order may vary; reject only replayed events predating the current item's takeover. */
  const isStaleNodeEvent = (nodeEvent: NodeEvent): boolean =>
    nodeExecutionRootItemId !== null &&
    getTrackedBackendItemId(nodeEvent.event) !== nodeExecutionRootItemId &&
    nodeEvent.sequence < nodeExecutionRootItemSequence;

  const trackNodeForItem = (backendItemId: number, nodeId: string, sequence: number): void => {
    if (nodeExecutionRootItemId !== backendItemId) {
      nodeExecutionRootItemId = backendItemId;
      nodeExecutionRootItemSequence = sequence;
      nodeExecution.clearAll();
      nodeExecution.setOrigin(waits.get(backendItemId)?.origin ?? null);
    }

    const nodeIds = nodeIdsByBackendItem.get(backendItemId);

    if (nodeIds) {
      nodeIds.add(nodeId);
    } else {
      nodeIdsByBackendItem.set(backendItemId, new Set([nodeId]));
    }
  };

  const applyNodeEvent = (nodeEvent: NodeEvent): void => {
    if (isStaleNodeEvent(nodeEvent)) {
      return;
    }

    const backendItemId = getTrackedBackendItemId(nodeEvent.event);
    markWorkflowCallRoot(nodeEvent.event);
    const invocationSourceId = getNodeSourceId(nodeEvent.event);

    trackNodeForItem(backendItemId, invocationSourceId, nodeEvent.sequence);

    switch (nodeEvent.kind) {
      case 'started': {
        const routedEvent = routeNodeEvent(nodeEvent.event, backendItemId, invocationSourceId);
        const wait = waits.get(backendItemId);
        if (wait) {
          activeProgressTarget.set(getProgressImageTarget(wait.localQueueItemId, backendItemId));
        }
        nodeExecution.started(routedEvent);
        return;
      }
      case 'completed':
        // A called workflow's child lifecycle is represented by the visible
        // Call Saved Workflow node. Only the root invocation may settle that
        // node or replace its latest output.
        waits.get(backendItemId)?.frameGates.delete(nodeEvent.event.item_id);
        if (backendItemId !== nodeEvent.event.item_id) {
          return;
        }
        nodeExecution.completed(routeNodeEvent(nodeEvent.event, backendItemId, invocationSourceId));
        return;
      case 'failed':
        waits.get(backendItemId)?.frameGates.delete(nodeEvent.event.item_id);
        if (backendItemId !== nodeEvent.event.item_id) {
          return;
        }
        coordinatorLogger.debug({
          context: {
            errorMessage: nodeEvent.event.error_message,
            errorType: nodeEvent.event.error_type,
            itemId: nodeEvent.event.item_id,
            nodeId: nodeEvent.event.invocation_source_id,
            sessionId: nodeEvent.event.session_id,
          },
          message: `Invocation failed: ${nodeEvent.event.error_type}`,
          name: 'queue.invocation-error',
        });
        nodeExecution.failed(routeNodeEvent(nodeEvent.event, backendItemId, invocationSourceId));
        return;
    }
  };

  const bufferNodeEvent = (nodeEvent: NodeEvent): void => {
    const itemId = getTrackedBackendItemId(nodeEvent.event);
    const events = pendingNodeEvents.get(itemId) ?? [];

    if (events.length >= NODE_EVENT_BUFFER_EVENTS_PER_ITEM) {
      return;
    }

    pendingNodeEvents.delete(itemId);
    pendingNodeEvents.set(itemId, [...events, nodeEvent]);

    while (pendingNodeEvents.size > NODE_EVENT_BUFFER_ITEM_LIMIT) {
      const oldestId = pendingNodeEvents.keys().next().value;

      if (oldestId === undefined) {
        break;
      }

      pendingNodeEvents.delete(oldestId);
    }
  };

  const bufferProgressEvent = (event: InvocationProgressEvent): void => {
    const backendItemId = getTrackedBackendItemId(event);

    pendingProgressEvents.delete(backendItemId);
    pendingProgressEvents.set(backendItemId, event);

    while (pendingProgressEvents.size > PROGRESS_EVENT_BUFFER_ITEM_LIMIT) {
      const oldestId = pendingProgressEvents.keys().next().value;

      if (oldestId === undefined) {
        break;
      }

      pendingProgressEvents.delete(oldestId);
    }
  };

  const replayNodeEvents = (backendItemId: number): void => {
    const events = pendingNodeEvents.get(backendItemId);

    pendingNodeEvents.delete(backendItemId);

    for (const nodeEvent of events ?? []) {
      applyNodeEvent(nodeEvent);
    }
  };

  const handleNodeEvent = (nodeEvent: NodeEvent): void => {
    if (!isActive()) {
      return;
    }

    if (isTrackedEvent(nodeEvent.event)) {
      applyNodeEvent(nodeEvent);
    } else if (inFlightSubmissions > 0) {
      bufferNodeEvent(nodeEvent);
    }
  };

  /** Runs an enqueue call while buffering events that may land before its response registers the items. */
  const withSubmissionInFlight = async <T>(submit: () => Promise<T>): Promise<T> => {
    inFlightSubmissions += 1;

    try {
      return await submit();
    } finally {
      inFlightSubmissions -= 1;

      if (inFlightSubmissions === 0) {
        pendingNodeEvents.clear();
        pendingProgressEvents.clear();
      }
    }
  };

  const settleWait = (backendItemId: number, outcome: TerminalOutcome): void => {
    const wait = waits.get(backendItemId);

    if (!wait) {
      bufferTerminalOutcome(backendItemId, outcome);
      return;
    }

    wait.frameGates.clear();
    waits.delete(backendItemId);
    settleNodes(backendItemId, outcome);
    const progressTarget = getProgressImageTarget(wait.localQueueItemId, backendItemId);
    const releaseProgressSlot = (): void => {
      if (isActive()) {
        activeProgressTarget.clear(progressTarget);
        progressImage.clear(progressTarget);
      }
    };
    const state = runProgress.get(wait.localQueueItemId);

    if (state) {
      if (state.activeBackendItemId === backendItemId) {
        state.activeBackendItemId = undefined;
        state.message = '';
        state.percentage = null;
      }
      if (outcome.status === 'completed') {
        state.completedBackendItemIds.add(backendItemId);
      }
      if (outcome.status === 'canceled') {
        state.cancelledBackendItemIds.add(backendItemId);
      }
      publishRunProgress(wait.localQueueItemId);
    }

    if (outcome.status === 'completed') {
      // Hold the frame before routing so decoded results and the next frameless slot can replace it without
      // blanking.
      progressImage.hold(progressTarget);
      const routingPromise = callbacks.onBackendItemComplete?.(wait.localQueueItemId, backendItemId);

      if (routingPromise) {
        // Keep following until routing lands; terminal events precede finished-image selection by network round
        // trips.
        activeProgressTarget.settle(progressTarget);
        void Promise.resolve(routingPromise)
          .finally(releaseProgressSlot)
          .catch(() => undefined);
      } else {
        releaseProgressSlot();
      }
    } else {
      releaseProgressSlot();
    }

    if (outcome.status === 'canceled') {
      callbacks.onBackendItemCancelled?.(wait.localQueueItemId, backendItemId);
    }

    wait.settle(outcome);
  };

  const settleFromQueueItem = (queueItem: QueueBackendItem): void => {
    const wait = waits.get(queueItem.id);
    if (wait) {
      const target = getProgressImageTarget(wait.localQueueItemId, queueItem.id);
      if (queueItem.status === 'in_progress' || (queueItem.status === 'waiting' && wait.hasWorkflowCall)) {
        activeProgressTarget.set(target);
      } else if ((queueItem.status === 'pending' || queueItem.status === 'waiting') && !wait.hasWorkflowCall) {
        activeProgressTarget.clear(target);
      }
    }
    if (isTerminalBackendStatus(queueItem.status)) {
      settleWait(queueItem.id, toTerminalOutcome(queueItem.status, queueItem.errorMessage, queueItem.errorType));
    }
  };

  const isTrackedEvent = (event: { item_id: number; root_item_id?: number | null }): boolean =>
    waits.has(getTrackedBackendItemId(event));

  const trackBackendItem = (
    localQueueItemId: string,
    backendItemId: number,
    context: RunTrackingContext
  ): Promise<TerminalOutcome> => {
    const bufferedOutcome = recentTerminalOutcomes.get(backendItemId);

    if (bufferedOutcome) {
      replayNodeEvents(backendItemId);
      recentTerminalOutcomes.delete(backendItemId);
      pendingProgressEvents.delete(backendItemId);
      settleNodes(backendItemId, bufferedOutcome);

      return Promise.resolve(bufferedOutcome);
    }

    return new Promise<TerminalOutcome>((settle) => {
      waits.set(backendItemId, { frameGates: new Map(), ...context, localQueueItemId, settle });
      replayNodeEvents(backendItemId);
      replayProgressEvent(backendItemId);
    });
  };

  const beginRun = (
    localQueueItemId: string,
    backendItemIds: number[],
    backendBatchId?: string,
    context: Partial<RunTrackingContext> = {}
  ): void => {
    if (!isActive()) {
      throw new QueueItemCancelledError(localQueueItemId);
    }

    const trackingContext: RunTrackingContext = {
      hasWorkflowCall:
        context.hasWorkflowCall === true ||
        backendItemIds.some((backendItemId) => waits.get(backendItemId)?.hasWorkflowCall),
      origin: context.origin ?? null,
    };

    runProgress.set(localQueueItemId, {
      backendItemIds,
      cancelledBackendItemIds: new Set(),
      completedBackendItemIds: new Set(),
      message: '',
      percentage: null,
    });
    runs.set(localQueueItemId, {
      backendBatchId,
      backendItemIds,
      outcomePromises: backendItemIds.map((backendItemId) =>
        trackBackendItem(localQueueItemId, backendItemId, trackingContext)
      ),
    });

    publishRunProgress(localQueueItemId);
  };

  /**
   * Sweep on reconnect, visibility, and a slow interval. Requests during an active sweep guarantee a trailing pass
   * after network recovery.
   */
  const sweep = async (): Promise<void> => {
    if (!isActive() || waits.size === 0) {
      return;
    }

    if (isSweeping) {
      isSweepRequested = true;
      return;
    }

    isSweeping = true;

    try {
      await Promise.all(
        [...waits.keys()].map(async (backendItemId) => {
          try {
            const queueItem = await backend.getItem(backendItemId);

            if (isActive()) {
              settleFromQueueItem(queueItem);
            }
          } catch (error) {
            if (isActive() && error instanceof ApiError && error.status === 404) {
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

      if (isSweepRequested) {
        isSweepRequested = false;
        void sweep();
      }
    }
  };

  /**
   * Fetch current previews through the revision-gated live handler. Snapshot failure is nonfatal; replay or the
   * next step can recover.
   */
  const refreshProgressPreviews = async (): Promise<void> => {
    if (!isActive() || waits.size === 0 || !backend.readProgressPreviews) {
      return;
    }

    let previews: Awaited<ReturnType<NonNullable<typeof backend.readProgressPreviews>>>;

    try {
      previews = await backend.readProgressPreviews();
    } catch {
      return;
    }

    if (!isActive()) {
      return;
    }

    for (const preview of previews) {
      // Structurally the socket payload; the port cannot name the event type.
      handleProgress(preview as unknown as InvocationProgressEvent);
    }
  };

  const handleStatusChanged = (event: QueueItemStatusChangedEvent): void => {
    if (!isActive()) {
      return;
    }

    const sequence = event.status_sequence;
    const previousSequence = latestStatusSequences.get(event.item_id);
    if (sequence !== null && previousSequence !== undefined && sequence < previousSequence) {
      return;
    }
    if (sequence !== null) {
      latestStatusSequences.set(event.item_id, sequence);
    }

    if (!isTerminalBackendStatus(event.status)) {
      const wait = waits.get(event.item_id);
      if (wait && event.status === 'in_progress') {
        activeProgressTarget.set(getProgressImageTarget(wait.localQueueItemId, event.item_id));
      }
      // Back to the queue (a workflow-call parent waiting on its child, a retry):
      // whatever frames follow belong to a new leg, and after a backend restart
      // their revisions start over.
      if (event.status === 'pending' || event.status === 'waiting') {
        if (wait && !wait.hasWorkflowCall) {
          activeProgressTarget.clear(getProgressImageTarget(wait.localQueueItemId, event.item_id));
        }
        wait?.frameGates.clear();
      }

      return;
    }

    // Root settlement clears its own wait's gates. Only child item statuses
    // need the cross-wait lookup to remove their per-child preview gate.
    if (!waits.has(event.item_id)) {
      clearFrameGate(event.item_id);
    }

    if (!isTrackedEvent(event)) {
      bufferTerminalOutcome(event.item_id, toTerminalOutcome(event.status, event.error_message, event.error_type));
      return;
    }

    settleWait(event.item_id, toTerminalOutcome(event.status, event.error_message, event.error_type));

    if (event.status === 'completed') {
      scheduleGalleryRefresh();
    }
  };

  const handleItemsCanceled = (event: QueueItemsCanceledEvent): void => {
    if (!isActive()) {
      return;
    }

    for (const itemId of event.canceled_item_ids) {
      if (waits.has(itemId)) {
        settleWait(itemId, { status: 'canceled' });
      }
    }
  };

  /**
   * Whether a frame is older than one already shown for its backend item;
   * records it as the newest when it is not. A new session on the same item
   * starts over.
   */
  const isStaleFrame = (event: InvocationProgressEvent): boolean => {
    const wait = waits.get(getTrackedBackendItemId(event));
    if (!wait) {
      return false;
    }

    const revision = event.revision ?? null;
    const gate = wait.frameGates.get(event.item_id);

    if (
      gate &&
      gate.sessionId === event.session_id &&
      revision !== null &&
      gate.revision !== null &&
      revision <= gate.revision
    ) {
      return true;
    }

    wait.frameGates.delete(event.item_id);
    wait.frameGates.set(event.item_id, { revision, sessionId: event.session_id });
    while (wait.frameGates.size > FRAME_GATES_PER_WAIT_LIMIT) {
      const oldestItemId = wait.frameGates.keys().next().value;
      if (oldestItemId === undefined) {
        break;
      }
      wait.frameGates.delete(oldestItemId);
    }

    return false;
  };

  const handleProgress = (event: InvocationProgressEvent): void => {
    if (!isActive()) {
      return;
    }

    const backendItemId = getTrackedBackendItemId(event);
    const wait = waits.get(backendItemId);

    if (!wait) {
      if (inFlightSubmissions > 0) {
        bufferProgressEvent(event);
      }
      return;
    }

    markWorkflowCallRoot(event);

    if (event.image?.dataURL && isStaleFrame(event)) {
      return;
    }

    const invocationSourceId = getNodeSourceId(event);
    trackNodeForItem(backendItemId, invocationSourceId, ++nodeEventSequence);
    nodeExecution.progress(invocationSourceId, event.percentage, event.message);

    const target = getProgressImageTarget(wait.localQueueItemId, backendItemId);
    activeProgressTarget.set(target);

    if (event.image?.dataURL) {
      progressImage.set({ dataUrl: event.image.dataURL, height: event.image.height, width: event.image.width }, target);
    }

    const state = runProgress.get(wait.localQueueItemId);

    if (state) {
      state.activeBackendItemId = backendItemId;
      state.message = event.message;
      state.percentage = event.percentage;
      publishRunProgress(wait.localQueueItemId);
    } else {
      progress.set(wait.localQueueItemId, { message: event.message, percentage: event.percentage });
    }
  };

  const replayProgressEvent = (backendItemId: number): void => {
    const event = pendingProgressEvents.get(backendItemId);

    pendingProgressEvents.delete(backendItemId);

    if (event) {
      handleProgress(event);
    }
  };

  /**
   * Clear transient node/model-load state on connection changes and reconcile on reconnect. Preserve followed
   * slots/frames because backend runs continue offline.
   */
  const handleConnectionChange = (status: BackendConnectionStatus): void => {
    if (!isActive()) {
      return;
    }

    progress.clearAll?.();
    nodeExecution.clearAll();
    nodeExecutionRootItemId = null;
    nodeExecutionRootItemSequence = 0;
    pendingNodeEvents.clear();
    // Keep previews received before enqueue adoption; the HTTP response may still be in flight.
    nodeIdsByBackendItem.clear();
    modelLoads.reset();

    if (status === 'connected') {
      scheduleGalleryRefresh();
      void sweep();
    }
  };

  /** Attach generation listeners to the shared socket hub. */
  const connect = (): void => {
    if (!isActive() || isAttached) {
      return;
    }

    isAttached = true;

    detachers.push(
      backend.on('queue_item_status_changed', handleStatusChanged),
      backend.on('queue_items_canceled', handleItemsCanceled),
      backend.on('invocation_progress', handleProgress),
      backend.on('invocation_started', (event: InvocationStartedEvent) =>
        handleNodeEvent({ event, kind: 'started', sequence: ++nodeEventSequence })
      ),
      backend.on('invocation_complete', (event: InvocationCompleteEvent) =>
        handleNodeEvent({ event, kind: 'completed', sequence: ++nodeEventSequence })
      ),
      backend.on('invocation_error', (event: InvocationErrorEvent) =>
        handleNodeEvent({ event, kind: 'failed', sequence: ++nodeEventSequence })
      ),
      backend.on('model_load_started', (payload: never) => {
        if (isActive()) {
          modelLoads.started(payload);
        }
      }),
      backend.on('model_load_complete', (payload: never) => {
        if (isActive()) {
          modelLoads.completed(payload);
        }
      })
    );

    // Fires synchronously with the current status, so attaching after the hub
    // has already connected still triggers the initial clear + sweep.
    detachers.push(backend.onConnectionChange(handleConnectionChange));

    // Sweep when the tab becomes visible instead of waiting for socket reconnect backoff.
    if (typeof document !== 'undefined') {
      const visibilityDocument = document;
      const handleVisibilityChange = (): void => {
        if (visibilityDocument.visibilityState === 'visible') {
          void sweep();
          void refreshProgressPreviews();
        }
      };

      visibilityDocument.addEventListener('visibilitychange', handleVisibilityChange);
      detachers.push(() => visibilityDocument.removeEventListener('visibilitychange', handleVisibilityChange));
    }

    sweepTimer = setInterval(() => {
      void sweep();
    }, sweepIntervalMs);
  };

  /** Detach generation listeners; the hub keeps the socket alive. */
  const dispose = (): void => {
    isDisposed = true;
    activeProgressTarget.clear();
    progressImage.clear();
    progress.clearAll?.();
    nodeExecution.clearAll();
    modelLoads.reset();

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

    // Dispose settles local waiters as cancelled without backend side effects; later coordinators can re-adopt
    // continuing runs.
    for (const wait of waits.values()) {
      wait.settle({ status: 'canceled' });
    }

    waits.clear();
    runs.clear();
    runProgress.clear();
    recentTerminalOutcomes.clear();
    pendingProgressEvents.clear();
    latestStatusSequences.clear();
  };

  const reconcile = async (items: ReconcileInput[]): Promise<Map<string, ReconcileOutcome>> => {
    const outcomes = new Map<string, ReconcileOutcome>();

    if (!isActive() || items.length === 0) {
      return outcomes;
    }

    const resolvedItems = await mapWithConcurrency(items, BACKEND_READ_CONCURRENCY, async (item) => {
      if (item.backendItemIds?.length || !item.projectId || !backend.getEnqueueReceipt) {
        return item;
      }
      const receipt = await backend.getEnqueueReceipt(item.projectId, item.id);
      return receipt ? { ...item, backendBatchId: receipt.batchId, backendItemIds: receipt.itemIds } : item;
    });
    const canReadExactItems = resolvedItems.every(
      (item) => item.backendItemIds?.length || (item.projectId && backend.getEnqueueReceipt)
    );
    const backendItems = canReadExactItems
      ? (
          await mapWithConcurrency(
            [...new Set(resolvedItems.flatMap((item) => item.backendItemIds ?? []))],
            BACKEND_READ_CONCURRENCY,
            async (itemId) => {
              try {
                return await backend.getItem(itemId);
              } catch (error) {
                if (error instanceof ApiError && error.status === 404) {
                  return undefined;
                }

                throw error;
              }
            }
          )
        ).filter((item) => item !== undefined)
      : await backend.listItems();

    if (!isActive()) {
      return outcomes;
    }

    const backendItemsById = new Map(backendItems.map((item) => [item.id, item]));

    for (const item of resolvedItems) {
      const matchesIdentity = (backendItem: QueueBackendItem | undefined): backendItem is QueueBackendItem =>
        backendItem !== undefined &&
        parseQueueItemOrigin(backendItem.origin) === item.id &&
        (item.projectId === undefined || parseQueueItemOriginProjectId(backendItem.origin) === item.projectId);
      const knownBackendItems = item.backendItemIds?.length
        ? item.backendItemIds.map((backendItemId) => backendItemsById.get(backendItemId))
        : backendItems.filter(matchesIdentity);
      const foundBackendItems = knownBackendItems.filter(matchesIdentity);
      const missingBackendItemIds = item.backendItemIds?.filter(
        (_backendItemId, index) => !matchesIdentity(knownBackendItems[index])
      );

      if (foundBackendItems.length === 0) {
        // A pending item with no backend trace was never accepted and is safe
        // to submit; a running item with (partially) vanished backend items is
        // unrecoverable.
        outcomes.set(
          item.id,
          item.status === 'pending' && !item.backendItemIds?.length
            ? { kind: 'enqueue' }
            : {
                ...(item.backendBatchId ? { backendBatchId: item.backendBatchId } : {}),
                ...(item.backendItemIds?.length ? { backendItemIds: item.backendItemIds } : {}),
                kind: 'missing',
              }
        );
        continue;
      }

      const backendItemIds = foundBackendItems.map((backendItem) => backendItem.id);
      const backendBatchId = item.backendBatchId ?? foundBackendItems[0]?.batchId;

      beginRun(item.id, backendItemIds, backendBatchId, {
        hasWorkflowCall: item.hasWorkflowCall,
        origin: item.origin ?? null,
      });

      for (const backendItem of foundBackendItems) {
        settleFromQueueItem(backendItem);
      }

      outcomes.set(
        item.id,
        item.status === 'running'
          ? {
              kind: 'resumed',
              ...(missingBackendItemIds?.length ? { backendItemIds, missingBackendItemIds } : {}),
            }
          : {
              backendBatchId,
              backendItemIds,
              kind: 'adopted',
              ...(missingBackendItemIds?.length ? { missingBackendItemIds } : {}),
            }
      );
    }

    // Fetch a preview for re-adopted runs rather than waiting for the next socket step.
    void refreshProgressPreviews();

    return outcomes;
  };

  /** Start tracking the accepted backend items. */
  const adoptEnqueueResult = (
    localQueueItemId: string,
    result: QueueEnqueueResult,
    workKind: 'generation' | 'workflow',
    context: Partial<RunTrackingContext> = {}
  ): QueueEnqueueResult => {
    if (result.enqueued === 0) {
      throw new QueueEnqueueNotAcceptedError(workKind);
    }

    beginRun(localQueueItemId, result.itemIds, result.batchId, context);

    return result;
  };

  const submitGenerate = async (
    localQueueItemId: string,
    request: QueueEnqueueGenerateRequest
  ): Promise<QueueEnqueueResult> => {
    if (!isActive()) {
      throw new QueueItemCancelledError(localQueueItemId);
    }

    return await withSubmissionInFlight(async () =>
      adoptEnqueueResult(localQueueItemId, await backend.enqueueGenerate(request), 'generation')
    );
  };

  const submitWorkflow = async (
    localQueueItemId: string,
    request: QueueEnqueueWorkflowRequest,
    origin?: QueueRunOrigin
  ): Promise<QueueEnqueueResult> => {
    if (!isActive()) {
      throw new QueueItemCancelledError(localQueueItemId);
    }

    return await withSubmissionInFlight(async () =>
      adoptEnqueueResult(localQueueItemId, await backend.enqueueWorkflow(request), 'workflow', {
        hasWorkflowCall: Object.values(request.graph.nodes).some((node) => node.type === 'call_saved_workflow'),
        origin: origin ?? null,
      })
    );
  };

  const waitForResults = async (
    localQueueItemId: string,
    queuedAt: string,
    options?: QueueResultImageOptions
  ): Promise<QueueResultImage[]> => {
    if (!isActive()) {
      throw new QueueItemCancelledError(localQueueItemId);
    }

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

      const imagesPerItem = await mapWithConcurrency(
        completedBackendItemIds,
        BACKEND_READ_CONCURRENCY,
        (backendItemId) =>
          options
            ? backend.getResultImages(backendItemId, localQueueItemId, queuedAt, options)
            : backend.getResultImages(backendItemId, localQueueItemId, queuedAt)
      );

      return imagesPerItem.flat();
    } finally {
      runs.delete(localQueueItemId);
      runProgress.delete(localQueueItemId);
      if (isActive()) {
        progress.clear(localQueueItemId);
      }
    }
  };

  const cancelRun = async ({ backendBatchId, backendItemIds }: CancelRunRequest): Promise<void> => {
    try {
      if (backendBatchId) {
        await backend.cancelQueueItemsByBatchIds([backendBatchId]);
        return;
      }

      if (backendItemIds?.length) {
        await backend.cancelQueueItems(backendItemIds);
      }
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) {
        return;
      }

      throw error;
    }
  };

  const detachRun = (localQueueItemId: string): void => {
    const run = runs.get(localQueueItemId);
    for (const backendItemId of run?.backendItemIds ?? []) {
      const wait = waits.get(backendItemId);
      if (wait?.localQueueItemId === localQueueItemId) {
        waits.delete(backendItemId);
        wait.frameGates.clear();
        settleNodes(backendItemId, { status: 'canceled' });
        wait.settle({ status: 'canceled' });
      }
    }
    runs.delete(localQueueItemId);
    runProgress.delete(localQueueItemId);
    for (let itemIndex = 1; itemIndex <= (run?.backendItemIds.length ?? 0); itemIndex += 1) {
      const target = { itemIndex, queueItemId: localQueueItemId };
      activeProgressTarget.clear(target);
      progressImage.clear(target);
    }
    progressImage.clearHeld(localQueueItemId);
    progress.clear(localQueueItemId);
  };

  return { cancelRun, connect, detachRun, dispose, reconcile, submitGenerate, submitWorkflow, waitForResults };
};
