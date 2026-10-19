import type { QueueItem } from '@features/queue/core/historyTypes';
import type {
  QueueBackendPort,
  QueueEnqueueGenerateRequest,
  QueueEnqueueWorkflowRequest,
  QueueItemDeliveryClaim,
  QueueResultImage,
  QueueResultImageOptions,
  QueueResultVideo,
  QueueRunOrigin,
} from '@features/queue/core/types';
import type { BackendConnectionStatus } from '@platform/transport/types';

import { collectGraphInputMediaNames } from '@features/queue/core/graphInputMedia';
import {
  isQueuePromptSeedBehaviour,
  isQueueSeedStep,
  isQueueWorkflowSeed,
  isQueueWorkflowBatchDatum,
  MAX_QUEUE_BATCH_ITEMS,
  type QueueWorkflowSeed,
} from '@features/queue/core/promptBatch';
import { QUEUE_SOURCE_RESULT_NODE_IDS, shouldSubmitPendingQueueItem } from '@features/queue/core/submissionRules';
import {
  isTerminalBackendStatus,
  isUtilityQueueItemOrigin,
  parseQueueItemOrigin,
  parseQueueItemOriginProjectId,
  type QueueItemStatusChangedEvent,
  type QueueItemsRetriedEvent,
} from '@features/queue/data/events';
import { progressImageStore } from '@features/queue/data/progressImageStore';
import {
  createQueueCoordinator,
  QueueEnqueueNotAcceptedError,
  QueueItemCancelledError,
  type QueueCoordinator,
  type QueueModelLoadPort,
  type QueueNodeExecutionPort,
  type ReconcileInput,
  type ReconcileOutcome,
} from '@features/queue/runtime/coordinator';
import { createUuid } from '@platform/browser/randomUuid';
import { mapWithConcurrency } from '@platform/core/concurrency';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { normalizeServerTimestamp } from '@platform/time/serverTimestamp';
import { ApiError, getApiErrorMessage } from '@platform/transport/http';

export interface QueueResultDestinationPort {
  addImagesToGalleryBoard(boardId: string, imageNames: string[]): Promise<void>;
  addVideosToGalleryBoard(boardId: string, videoNames: string[]): Promise<void>;
}

export interface QueueRuntime {
  dispose(): Promise<void>;
  start(): void;
}

export interface QueueRunJournalPort {
  listForProject(
    projectId: string
  ): Promise<
    | { entries: Array<{ item: unknown; projectId: string; queueItemId: string }>; kind: 'available' }
    | { kind: 'unavailable' }
  >;
  record(
    projectId: string,
    item: QueueItem
  ): Promise<{ kind: 'invalid' | 'quota' | 'stored' | 'too-large' | 'unavailable' }>;
  settle(projectId: string, queueItemId: string): Promise<{ kind: 'removed' | 'unavailable' }>;
}

export type QueueRunLock = { kind: 'acquired'; release(): Promise<void> } | { kind: 'contended' | 'unavailable' };

export interface QueueRunLockPort {
  acquire(projectId: string, queueItemId: string): Promise<QueueRunLock>;
  /** Settle once no tab holds the run, or the signal aborts. */
  waitForRunRelease(projectId: string, queueItemId: string, signal: AbortSignal): Promise<void>;
  /** Whether any tab holds the run, asked without requesting it; null when the browser cannot tell. */
  isHeld(projectId: string, queueItemId: string): Promise<boolean | null>;
}

export interface QueueHistoryProject {
  id: string;
  queue: { items: QueueItem[] };
}

export interface QueueHistoryCommands {
  markBackendCancelled(payload: { backendItemId: number; projectId: string; queueItemId: string }): void;
  markBackendSubmitted(payload: {
    backendBatchId?: string;
    backendItemIds: number[];
    projectId: string;
    queueItemId: string;
  }): void;
  setCancellationPending(payload: { pending: boolean; projectId: string; queueItemId: string }): void;
  setLocalRecoveryState(payload: {
    projectId: string;
    queueItemId: string;
    state: NonNullable<QueueItem['localRecoveryState']>;
  }): void;
  recordError(payload: { area: string; message: string; namespace: 'queue'; projectId?: string }): void;
  refreshBackendData(): void;
  /** A board may have refused a run: read the listing results are routed by again, at once. */
  refreshBoardDestinations(): void;
  /** Reopen a finished run around the backend items re-running its failed or canceled ones; refused runs stay. */
  reopenForRetry(payload: {
    projectId: string;
    queueItemId: string;
    retriedItems: { itemId: number; sourceItemId: number }[];
  }): void;
  restoreFromJournal(payload: { items: unknown[]; projectId: string }): void;
  routePartialResults(payload: {
    backendItemId: number;
    images: QueueResultImage[];
    projectId: string;
    queueItemId: string;
    videos: QueueResultVideo[];
  }): void;
  routeResults(payload: {
    images: QueueResultImage[];
    projectId: string;
    queueItemId: string;
    videos: QueueResultVideo[];
  }): void;
  setConnectionStatus(payload: { error?: string; status: BackendConnectionStatus }): void;
  /** Stage a retried Canvas run's results at the current bbox when no open run could adopt it. */
  stageRetriedCanvasResults(payload: {
    backendItemId: number;
    images: QueueResultImage[];
    projectId: string;
    queueItemId: string;
  }): void;
  setStatus(payload: {
    error?: string;
    notify?: boolean;
    projectId: string;
    queueItemId: string;
    status: QueueItem['status'];
  }): void;
}

type QueueItemBackendSubmission =
  | { kind: 'generate'; request: QueueEnqueueGenerateRequest }
  | { kind: 'workflow'; request: QueueEnqueueWorkflowRequest }
  | { error: string; kind: 'invalid' };

const toErrorMessage = (error: unknown): string =>
  getApiErrorMessage(error, error instanceof Error ? error.message : String(error));

export const getQueueItemResultImageOptions = (queueItem: QueueItem): QueueResultImageOptions | undefined => {
  return queueItem.snapshot.resultNodeIds ? { resultNodeIds: queueItem.snapshot.resultNodeIds } : undefined;
};

/** Where a run's node progress belongs; records queued before projects owned several workflows have none. */
const getQueueRunOrigin = (projectId: string, queueItem: QueueItem): QueueRunOrigin | undefined => {
  const submission = (queueItem.snapshot as Partial<QueueItem['snapshot']>).backendSubmission;

  return submission?.kind === 'workflow' && typeof submission.projectWorkflowId === 'string'
    ? { projectId, workflowId: submission.projectWorkflowId }
    : undefined;
};

/** Read compiled graphs to identify input passthroughs; legacy or invalid snapshots may lack them. */
const getQueueItemCompiledGraph = (queueItem: QueueItem): unknown => {
  const submission = (queueItem.snapshot as Partial<QueueItem['snapshot']>).backendSubmission;
  return submission && typeof submission === 'object' && 'graph' in submission ? submission.graph : undefined;
};

const queueItemHasWorkflowCall = (queueItem: QueueItem): boolean => {
  const submission = (queueItem.snapshot as Partial<QueueItem['snapshot']>).backendSubmission;
  if (!submission || typeof submission !== 'object' || !('kind' in submission) || submission.kind !== 'workflow') {
    return false;
  }

  const graph = getQueueItemCompiledGraph(queueItem);
  if (!graph || typeof graph !== 'object' || Array.isArray(graph) || !('nodes' in graph)) {
    return false;
  }

  const nodes = graph.nodes;
  if (!nodes || typeof nodes !== 'object' || Array.isArray(nodes)) {
    return false;
  }

  return Object.values(nodes).some(
    (node) => node !== null && typeof node === 'object' && 'type' in node && node.type === 'call_saved_workflow'
  );
};

/** Recover legacy random-toggle submissions with their original seed expansion rules. */
const readSubmissionSeedStep = (submission: { seedStep?: unknown; shouldRandomizeSeed?: unknown }) =>
  isQueueSeedStep(submission.seedStep)
    ? submission.seedStep
    : typeof submission.shouldRandomizeSeed === 'boolean'
      ? submission.shouldRandomizeSeed
        ? 1
        : 0
      : null;

/** Every recorded seed must name a distinct field on a node the graph still has, in the seed range, stepping ±1. */
const areQueueWorkflowSeedsValid = (seeds: unknown, graph: { nodes?: Record<string, unknown> }): boolean => {
  if (!Array.isArray(seeds)) {
    return false;
  }

  const targets = new Set<string>();

  return seeds.every((seed) => {
    if (!isQueueWorkflowSeed(seed) || !graph.nodes || !(seed.nodeId in graph.nodes)) {
      return false;
    }

    const target = `${seed.nodeId}:${seed.fieldName}`;

    return targets.has(target) ? false : (targets.add(target), true);
  });
};

/**
 * Every group zips datums of one length onto fields of nodes the graph still has, no field is fed twice (by
 * another group or by a seed), and the product stays within what the queue accepts.
 */
const areQueueWorkflowBatchGroupsValid = (
  batchData: unknown,
  graph: { nodes?: Record<string, unknown> },
  seeds: readonly QueueWorkflowSeed[] | undefined,
  batchCount: number
): boolean => {
  if (!Array.isArray(batchData) || batchData.length === 0) {
    return false;
  }

  const targets = new Set((seeds ?? []).map((seed) => `${seed.nodeId}:${seed.fieldName}`));
  let sessions = 1;

  for (const group of batchData) {
    if (!Array.isArray(group) || group.length === 0 || !group.every(isQueueWorkflowBatchDatum)) {
      return false;
    }

    const length = group[0].items.length;

    for (const datum of group) {
      const target = `${datum.nodeId}:${datum.fieldName}`;

      if (datum.items.length !== length || !graph.nodes || !(datum.nodeId in graph.nodes) || targets.has(target)) {
        return false;
      }

      targets.add(target);
    }

    sessions *= length;
  }

  return sessions * batchCount <= MAX_QUEUE_BATCH_ITEMS;
};

export const createQueueItemBackendSubmission = (
  project: Pick<QueueHistoryProject, 'id'>,
  queueItem: QueueItem
): QueueItemBackendSubmission => {
  const submission = (queueItem.snapshot as Partial<QueueItem['snapshot']>).backendSubmission;

  if (!submission || typeof submission !== 'object' || !('kind' in submission)) {
    return { error: 'Queue item is missing a compiled backend submission.', kind: 'invalid' };
  }

  if (submission.kind !== 'invalid' && submission.kind !== 'generate' && submission.kind !== 'workflow') {
    return { error: 'Queue item has an unsupported compiled backend submission.', kind: 'invalid' };
  }

  if (submission.kind === 'invalid') {
    return typeof submission.error === 'string'
      ? submission
      : { error: 'Queue item has an invalid compiled backend submission.', kind: 'invalid' };
  }

  if (!submission.graph || typeof submission.graph !== 'object') {
    return { error: 'Queue item backend submission is missing its compiled graph.', kind: 'invalid' };
  }

  if (
    typeof submission.batchCount !== 'number' ||
    !Number.isFinite(submission.batchCount) ||
    !Number.isSafeInteger(submission.batchCount) ||
    submission.batchCount < 1 ||
    submission.batchCount > MAX_QUEUE_BATCH_ITEMS
  ) {
    return { error: 'Queue item backend submission has an invalid batch count.', kind: 'invalid' };
  }

  if (submission.kind === 'generate') {
    const seedStep = readSubmissionSeedStep(submission);

    if (
      typeof submission.negativePrompt !== 'string' ||
      typeof submission.negativePromptNodeId !== 'string' ||
      typeof submission.positivePrompt !== 'string' ||
      typeof submission.positivePromptNodeId !== 'string' ||
      typeof submission.seed !== 'number' ||
      !Number.isFinite(submission.seed) ||
      typeof submission.seedNodeId !== 'string' ||
      seedStep === null ||
      (submission.positivePrompts !== undefined &&
        (!Array.isArray(submission.positivePrompts) ||
          submission.positivePrompts.some((prompt) => typeof prompt !== 'string'))) ||
      (submission.seedBehaviour !== undefined && !isQueuePromptSeedBehaviour(submission.seedBehaviour))
    ) {
      return { error: 'Queue item has malformed generate submission metadata.', kind: 'invalid' };
    }
    const {
      kind: _,
      shouldRandomizeSeed: _legacyToggle,
      ...compiled
    } = submission as typeof submission & { shouldRandomizeSeed?: unknown };
    return {
      kind: 'generate',
      request: {
        ...compiled,
        destination: queueItem.snapshot.destination,
        ...(isQueueSeedStep(submission.seedStep) ? {} : { legacySeedPlan: true as const }),
        ...(queueItem.snapshot.prepend === true ? { prepend: true } : {}),
        projectId: project.id,
        seedStep,
        sourceQueueItemId: queueItem.id,
      },
    };
  }

  if (submission.seeds !== undefined && !areQueueWorkflowSeedsValid(submission.seeds, submission.graph)) {
    return { error: 'Queue item has malformed workflow seed metadata.', kind: 'invalid' };
  }

  if (
    submission.batchData !== undefined &&
    !areQueueWorkflowBatchGroupsValid(submission.batchData, submission.graph, submission.seeds, submission.batchCount)
  ) {
    return { error: 'Queue item has malformed workflow batch metadata.', kind: 'invalid' };
  }

  // Workflow provenance stays on the snapshot: the backend enqueue accepts neither the originating project
  // workflow nor the library binding older records still carry.
  const {
    kind: _,
    projectWorkflowId: _projectWorkflowId,
    ...compiled
  } = submission as typeof submission & {
    libraryWorkflowId?: string;
  };
  delete (compiled as { libraryWorkflowId?: string }).libraryWorkflowId;
  return {
    kind: 'workflow',
    request: {
      ...compiled,
      destination: queueItem.snapshot.destination,
      ...(queueItem.snapshot.prepend === true ? { prepend: true } : {}),
      projectId: project.id,
      sourceQueueItemId: queueItem.id,
    },
  };
};

export interface QueueHistoryPort {
  commands: QueueHistoryCommands;
  getSnapshot(): {
    connectionStatus: BackendConnectionStatus;
    isHydrated: boolean;
    projects: QueueHistoryProject[];
  };
  subscribe(listener: () => void): () => void;
}

const getRouteTarget = (history: QueueHistoryPort, localQueueItemId: string) =>
  history
    .getSnapshot()
    .projects.map((project) => ({
      project,
      queueItem: project.queue.items.find((item) => item.id === localQueueItemId),
    }))
    .find((target) => target.queueItem !== undefined);

const getRunKey = (projectId: string, queueItemId: string): string => JSON.stringify([projectId, queueItemId]);

const QUEUE_RUNTIME_CONCURRENCY = 16;
/**
 * Reruns no open run adopted, remembered until they finish so their results still reach the Gallery or Canvas; one
 * retry can re-run up to the backend's default `max_queue_size`.
 */
const UNADOPTED_RERUN_LIMIT = 10_000;
/** How long a claimant turned away from a rerun waits at most before asking again whether its holder confirmed. */
const RERUN_CLAIM_POLL_MS = 3_000;
/** How long the server keeps an unconfirmed delivery claim (its contract), and how often a run renews its claims. */
const RERUN_CLAIM_TTL_MS = 120_000;
const RERUN_CLAIM_RENEW_MS = 40_000;

interface RunAttempt {
  generation: number;
  key: string;
  projectId: string;
  queueItemId: string;
}

const getJournalSignature = (item: QueueItem): string =>
  JSON.stringify([
    item.status,
    item.cancellationPending,
    item.backendBatchId,
    item.backendItemIds,
    item.completedBackendItemIds,
    item.cancelledBackendItemIds,
  ]);

export const createQueueRuntime = ({
  backend,
  destinations,
  ensureProjectPersisted,
  ensureTemplatesLoaded,
  history,
  journal,
  locks,
  modelLoads,
  nodeExecution,
  viewerUserId = null,
}: {
  backend: QueueBackendPort;
  destinations: QueueResultDestinationPort;
  ensureProjectPersisted?(projectId: string): Promise<'ready' | 'refused' | 'retry'>;
  ensureTemplatesLoaded: () => void;
  history: QueueHistoryPort;
  journal?: QueueRunJournalPort;
  locks?: QueueRunLockPort;
  modelLoads: QueueModelLoadPort;
  nodeExecution: QueueNodeExecutionPort;
  /** The signed-in user on a multiuser server; null in single-user mode, where every item is the viewer's. */
  viewerUserId?: string | null;
}): QueueRuntime => {
  const owner = captureAccountScope();
  const commands: QueueHistoryCommands = {
    ...history.commands,
    setStatus: (payload) => {
      history.commands.setStatus(payload);
      // A run bound for the gallery, or a workflow run, whose nodes name boards whatever its destination, may have
      // failed on a board: deleted, archived, or no longer the caller's to add to. The listing is read again at once;
      // a run submitted after that read lands is routed by it, one submitted before still goes by the old listing.
      const failedSnapshot =
        payload.status === 'failed' ? getRouteTarget(history, payload.queueItemId)?.queueItem?.snapshot : undefined;
      if (failedSnapshot && (failedSnapshot.destination === 'gallery' || failedSnapshot.sourceId === 'workflow')) {
        history.commands.refreshBoardDestinations();
      }
    },
  };
  const startedRunKeys = new Set<string>();
  const reconcileRetryRunKeys = new Set<string>();
  const detachers: Array<() => void> = [];
  const loadedProjectIds = new Set<string>();
  const loadingProjectIds = new Set<string>();
  const pendingReconcileRunKeys = new Set<string>();
  const trackedRuns = new Map<string, { projectId: string; queueItemId: string; signature: string }>();
  const journalOperations = new Map<string, Promise<void>>();
  const journalRecordFlushes = new Map<string, Promise<void>>();
  const journalSettleFlushes = new Map<string, Promise<void>>();
  const pendingJournalRecords = new Map<
    string,
    { item: QueueItem; projectId: string; restoreLocalOnly: boolean; signature: string; token: number }
  >();
  const pendingJournalSettles = new Map<string, { ownershipEpoch: number; projectId: string; queueItemId: string }>();
  const durableJournalSignatures = new Map<string, string>();
  const reportedJournalFailures = new Set<string>();
  /** Runs whose failed cancellation was already reported; retries stay quiet until the run settles. */
  const reportedCancelFailures = new Set<string>();
  const ownedLocks = new Map<string, Extract<QueueRunLock, { kind: 'acquired' }>>();
  const lockRequests = new Map<string, Promise<boolean>>();
  /** Runs this tab gave up whose browser lock is not yet free; their lock still reads as held. */
  const releasingRuns = new Map<string, Promise<void>>();
  const preparingRunKeys = new Set<string>();
  const activeAttempts = new Map<string, RunAttempt>();
  const ownershipEpochs = new Map<string, number>();
  const latestJournalWriteTokens = new Map<string, number>();
  const pendingFallbackAcks = new Map<string, { projectId: string; queueItemId: string }>();
  const fallbackAckOperations = new Map<string, Promise<void>>();
  let nextAttemptGeneration = 0;
  let nextJournalWriteToken = 0;
  const cancellationOperations = new Map<string, Promise<void>>();
  const submissionOperations = new Map<string, Promise<unknown>>();
  const cancelRequestedRunKeys = new Set<string>();
  let reconcileRetryDelayMs = 100;
  let retryTimer: ReturnType<typeof setTimeout> | undefined;
  let renewTimer: ReturnType<typeof setInterval> | undefined;
  let hasInitialized = false;
  let isInitializing = false;
  let isReconciling = false;
  let isDisposed = false;
  let isStarted = false;
  const isActive = (): boolean => !isDisposed && isAccountScopeCurrent(owner);
  const beginAttempt = (projectId: string, queueItemId: string): RunAttempt => {
    const key = getRunKey(projectId, queueItemId);
    const attempt = {
      generation: ++nextAttemptGeneration,
      key,
      projectId,
      queueItemId,
    };
    ownershipEpochs.set(key, (ownershipEpochs.get(key) ?? 0) + 1);
    activeAttempts.set(attempt.key, attempt);
    return attempt;
  };
  const isAttemptCurrent = (attempt: RunAttempt): boolean =>
    isActive() && activeAttempts.get(attempt.key)?.generation === attempt.generation;
  const invalidateAttempt = (projectId: string, queueItemId: string): void => {
    activeAttempts.delete(getRunKey(projectId, queueItemId));
  };

  const scheduleRetry = (): void => {
    if (!isActive() || retryTimer !== undefined) {
      return;
    }
    const delay = reconcileRetryDelayMs;
    reconcileRetryDelayMs = Math.min(reconcileRetryDelayMs * 2, 5_000);
    retryTimer = setTimeout(() => {
      retryTimer = undefined;
      if (!isActive()) {
        return;
      }
      for (const runKey of reconcileRetryRunKeys) {
        startedRunKeys.delete(runKey);
        pendingReconcileRunKeys.add(runKey);
      }
      reconcileRetryRunKeys.clear();
      for (const runKey of pendingJournalSettles.keys()) {
        void flushJournalSettle(runKey);
      }
      for (const runKey of pendingFallbackAcks.keys()) {
        void flushFallbackAck(runKey);
      }
      synchronize();
    }, delay);
  };

  const reportJournalFailure = (projectId: string, queueItemId: string, kind: string): void => {
    const key = `write:${kind}`;
    if (!isActive() || reportedJournalFailures.has(key)) {
      return;
    }
    reportedJournalFailures.add(key);
    commands.recordError({
      area: 'queue-recovery',
      message: `Local recovery for queue item ${queueItemId} is unavailable (${kind}). The backend run will continue.`,
      namespace: 'queue',
      projectId,
    });
  };

  const reportJournalLoadFailure = (projectId?: string): void => {
    const key = 'load';
    if (!isActive() || reportedJournalFailures.has(key)) {
      return;
    }
    reportedJournalFailures.add(key);
    commands.recordError({
      area: 'queue-recovery',
      message: 'Queue recovery storage is unavailable. Runs can continue but will not survive a reload.',
      namespace: 'queue',
      ...(projectId ? { projectId } : {}),
    });
  };

  const enqueueJournalOperation = (key: string, operation: () => Promise<void>): Promise<void> => {
    const previous = journalOperations.get(key) ?? Promise.resolve();
    const next = previous
      .catch(() => undefined)
      .then(operation)
      .catch(() => undefined)
      .finally(() => {
        if (journalOperations.get(key) === next) {
          journalOperations.delete(key);
        }
      });
    journalOperations.set(key, next);
    return next;
  };

  const flushJournalRecords = (key: string): Promise<void> => {
    if (!journal) {
      return Promise.resolve();
    }
    const activeFlush = journalRecordFlushes.get(key);
    if (activeFlush) {
      return activeFlush;
    }
    const flush = (async () => {
      while (pendingJournalRecords.has(key)) {
        const pending = pendingJournalRecords.get(key)!;
        pendingJournalRecords.delete(key);
        const result = await journal.record(pending.projectId, pending.item);
        if (result.kind === 'stored') {
          durableJournalSignatures.set(key, pending.signature);
          if (isActive()) {
            commands.setLocalRecoveryState({
              projectId: pending.projectId,
              queueItemId: pending.item.id,
              state: 'durable',
            });
          }
        } else {
          durableJournalSignatures.delete(key);
          reportJournalFailure(pending.projectId, pending.item.id, result.kind);
          const target = getRouteTarget(history, pending.item.id);
          if (
            pending.restoreLocalOnly &&
            latestJournalWriteTokens.get(key) === pending.token &&
            isActive() &&
            target?.project.id === pending.projectId &&
            target.queueItem?.localRecoveryState === 'uncertain' &&
            (!locks || ownedLocks.has(key))
          ) {
            commands.setLocalRecoveryState({
              projectId: pending.projectId,
              queueItemId: pending.item.id,
              state: 'local-only',
            });
          }
        }
      }
    })()
      .catch(() => {
        durableJournalSignatures.delete(key);
      })
      .finally(() => {
        if (journalRecordFlushes.get(key) === flush) {
          journalRecordFlushes.delete(key);
        }
        if (pendingJournalRecords.has(key)) {
          void flushJournalRecords(key);
        }
      });
    journalRecordFlushes.set(key, flush);
    return flush;
  };

  const recordActiveRun = async (projectId: string, item: QueueItem): Promise<boolean> => {
    const key = getRunKey(projectId, item.id);
    const signature = getJournalSignature(item);
    const tracked = trackedRuns.get(key);
    if (tracked?.signature === signature) {
      await journalRecordFlushes.get(key);
      return durableJournalSignatures.get(key) === signature;
    }
    trackedRuns.set(key, { projectId, queueItemId: item.id, signature });
    if (!journal) {
      return false;
    }
    const restoreLocalOnly = item.localRecoveryState === 'local-only';
    const token = ++nextJournalWriteToken;
    latestJournalWriteTokens.set(key, token);
    if (restoreLocalOnly) {
      commands.setLocalRecoveryState({ projectId, queueItemId: item.id, state: 'uncertain' });
    }
    pendingJournalRecords.set(key, { item, projectId, restoreLocalOnly, signature, token });
    await flushJournalRecords(key);
    return durableJournalSignatures.get(key) === signature;
  };

  const drainJournal = async (key: string): Promise<void> => {
    while (journalRecordFlushes.has(key) || journalOperations.has(key)) {
      await Promise.all([journalRecordFlushes.get(key), journalOperations.get(key)]);
    }
  };

  const releaseRunOwnership = (projectId: string, queueItemId: string): void => {
    const key = getRunKey(projectId, queueItemId);
    const lock = ownedLocks.get(key);
    if (!lock) {
      return;
    }
    ownedLocks.delete(key);
    const released = lock.release().catch(() => undefined);
    releasingRuns.set(key, released);
    void released.finally(() => {
      if (releasingRuns.get(key) === released) {
        releasingRuns.delete(key);
      }
    });
  };

  const releaseRunOwnershipAfterJournal = (projectId: string, queueItemId: string): void => {
    const key = getRunKey(projectId, queueItemId);
    const epoch = ownershipEpochs.get(key) ?? 0;
    void drainJournal(key).finally(() => {
      if ((ownershipEpochs.get(key) ?? 0) === epoch && !pendingJournalSettles.has(key)) {
        releaseRunOwnership(projectId, queueItemId);
      }
    });
  };

  const flushFallbackAck = (key: string): Promise<void> => {
    const pending = pendingFallbackAcks.get(key);
    if (!pending || !backend.acknowledgeEnqueue || !isActive()) {
      return Promise.resolve();
    }
    if (history.getSnapshot().connectionStatus !== 'connected') {
      scheduleRetry();
      return Promise.resolve();
    }
    const active = fallbackAckOperations.get(key);
    if (active) {
      return active;
    }
    const operation = backend
      .acknowledgeEnqueue(pending.projectId, pending.queueItemId)
      .then(() => {
        if (pendingFallbackAcks.get(key) === pending) {
          pendingFallbackAcks.delete(key);
        }
        reconcileRetryDelayMs = 100;
      })
      .catch(() => scheduleRetry())
      .finally(() => {
        if (fallbackAckOperations.get(key) === operation) {
          fallbackAckOperations.delete(key);
        }
      });
    fallbackAckOperations.set(key, operation);
    return operation;
  };

  const scheduleFallbackAck = (projectId: string, queueItemId: string): void => {
    const key = getRunKey(projectId, queueItemId);
    if (!pendingFallbackAcks.has(key)) {
      pendingFallbackAcks.set(key, { projectId, queueItemId });
    }
    void flushFallbackAck(key);
  };

  const flushJournalSettle = (key: string): Promise<void> => {
    if (!journal) {
      return Promise.resolve();
    }
    const active = journalSettleFlushes.get(key);
    if (active) {
      return active;
    }
    const pending = pendingJournalSettles.get(key);
    if (!pending) {
      return Promise.resolve();
    }
    const recordFlush = journalRecordFlushes.get(key);
    const flush = enqueueJournalOperation(key, async () => {
      await recordFlush;
      let result: Awaited<ReturnType<QueueRunJournalPort['settle']>>;
      try {
        result = await journal.settle(pending.projectId, pending.queueItemId);
      } catch {
        result = { kind: 'unavailable' };
      }
      if (result.kind === 'removed') {
        if (pendingJournalSettles.get(key) === pending) {
          pendingJournalSettles.delete(key);
        }
        reconcileRetryDelayMs = 100;
        if ((ownershipEpochs.get(key) ?? 0) === pending.ownershipEpoch) {
          releaseRunOwnership(pending.projectId, pending.queueItemId);
        }
      } else {
        reportJournalFailure(pending.projectId, pending.queueItemId, result.kind);
      }
    }).finally(() => {
      if (journalSettleFlushes.get(key) === flush) {
        journalSettleFlushes.delete(key);
      }
      if (pendingJournalSettles.has(key)) {
        scheduleRetry();
      }
    });
    journalSettleFlushes.set(key, flush);
    return flush;
  };

  const settleRun = (projectId: string, queueItemId: string): void => {
    const key = getRunKey(projectId, queueItemId);
    const target = getRouteTarget(history, queueItemId);
    if (
      target?.project.id === projectId &&
      target.queueItem?.localRecoveryState === 'local-only' &&
      target.queueItem.backendItemIds?.length
    ) {
      scheduleFallbackAck(projectId, queueItemId);
    }
    trackedRuns.delete(key);
    durableJournalSignatures.delete(key);
    releaseAdoptedReruns(projectId, queueItemId);
    startedRunKeys.delete(key);
    pendingReconcileRunKeys.delete(key);
    reconcileRetryRunKeys.delete(key);
    cancelRequestedRunKeys.delete(key);
    reportedCancelFailures.delete(key);
    invalidateAttempt(projectId, queueItemId);
    pendingJournalRecords.delete(key);
    const ownershipEpoch = ownershipEpochs.get(key) ?? 0;
    if (!journal) {
      releaseRunOwnership(projectId, queueItemId);
      return;
    }
    if (!pendingJournalSettles.has(key)) {
      pendingJournalSettles.set(key, { ownershipEpoch, projectId, queueItemId });
    }
    void flushJournalSettle(key);
  };

  const completeCancellation = (
    projectId: string,
    queueItemId: string,
    acceptedBackendItemIds?: readonly number[]
  ): void => {
    const target = getRouteTarget(history, queueItemId);
    if (
      acceptedBackendItemIds?.length &&
      target?.project.id === projectId &&
      target.queueItem?.localRecoveryState === 'local-only'
    ) {
      scheduleFallbackAck(projectId, queueItemId);
    }
    settleRun(projectId, queueItemId);
    if (isActive()) {
      commands.setCancellationPending({ pending: false, projectId, queueItemId });
    }
  };

  const cancelRun = (
    projectId: string,
    queueItem: QueueItem,
    backendItemIds = queueItem.backendItemIds,
    backendBatchId = queueItem.backendBatchId
  ): Promise<void> => {
    const key = getRunKey(projectId, queueItem.id);
    const active = cancellationOperations.get(key);
    if (active) {
      return active;
    }
    const cancellationItem: QueueItem = {
      ...queueItem,
      backendBatchId,
      backendItemIds,
      cancellationPending: true,
      status: 'cancelled',
    };
    cancelRequestedRunKeys.add(key);
    const operation = recordActiveRun(projectId, cancellationItem)
      .then(() => coordinator.cancelRun({ backendBatchId, backendItemIds }))
      .then(() => {
        completeCancellation(projectId, queueItem.id, backendItemIds);
      })
      .catch((error: unknown) => {
        if (isActive()) {
          if (!reportedCancelFailures.has(key)) {
            reportedCancelFailures.add(key);
            commands.recordError({
              area: 'queue-cancel',
              message: `Could not stop queue item ${queueItem.id} yet; retrying automatically. ${toErrorMessage(error)}`,
              namespace: 'queue',
              projectId,
            });
          }
          reconcileRetryRunKeys.add(key);
          scheduleRetry();
        }
      })
      .finally(() => cancellationOperations.delete(key));
    cancellationOperations.set(key, operation);
    commands.setCancellationPending({ pending: true, projectId, queueItemId: queueItem.id });
    return operation;
  };

  const acquireRunOwnership = (projectId: string, queueItemId: string): Promise<boolean> => {
    const key = getRunKey(projectId, queueItemId);
    if (ownedLocks.has(key) || !locks) {
      return Promise.resolve(true);
    }
    const pending = lockRequests.get(key);
    if (pending) {
      return pending;
    }
    const request = locks
      .acquire(projectId, queueItemId)
      .then(async (result) => {
        if (!isActive()) {
          if (result.kind === 'acquired') {
            await result.release().catch(() => undefined);
          }
          return false;
        }
        if (result.kind === 'acquired') {
          ownedLocks.set(key, result);
          return true;
        }
        return result.kind === 'unavailable';
      })
      .catch(() => false)
      .finally(() => lockRequests.delete(key));
    lockRequests.set(key, request);
    return request;
  };

  interface ResultReadTask {
    execute(): Promise<void>;
  }
  const pendingResultReads: ResultReadTask[] = [];
  let resultReadFlush: Promise<void> | undefined;
  const scheduleResultReadFlush = (): void => {
    if (resultReadFlush) {
      return;
    }
    const flush = Promise.resolve().then(async () => {
      while (pendingResultReads.length > 0) {
        const batch = pendingResultReads.splice(0);
        await mapWithConcurrency(batch, QUEUE_RUNTIME_CONCURRENCY, (task) => task.execute());
      }
    });
    resultReadFlush = flush.finally(() => {
      resultReadFlush = undefined;
      // A read queued after the loop drained but before this settled found the flush still set and did not start one.
      if (pendingResultReads.length > 0) {
        scheduleResultReadFlush();
      }
    });
  };
  const runResultRead = <T>(operation: () => Promise<T>): Promise<T> =>
    new Promise<T>((resolve, reject) => {
      pendingResultReads.push({
        execute: async () => {
          try {
            resolve(await operation());
          } catch (error) {
            reject(error);
          }
        },
      });
      scheduleResultReadFlush();
    });

  /**
   * The board that result images and videos unassigned on the backend land on. A result already on a board keeps it,
   * whether a node saved it there or an earlier settlement or the user put it there.
   */
  const getResultDestinationBoardId = (queueItem: QueueItem): string | null => {
    const boardId = queueItem.snapshot.galleryBoardId;

    return queueItem.snapshot.destination === 'gallery' && boardId && boardId !== 'none' ? boardId : null;
  };

  const addImagesToDestination = async (queueItem: QueueItem, images: QueueResultImage[]): Promise<void> => {
    const boardId = getResultDestinationBoardId(queueItem);
    const imageNames = images.filter((image) => !image.boardId).map((image) => image.imageName);

    if (isActive() && boardId && imageNames.length > 0) {
      await destinations.addImagesToGalleryBoard(boardId, imageNames);
    }
  };

  /**
   * Hydrate generated videos for display and attach the unassigned ones to the destination board. Failures are
   * recorded but never turn successful generation into failure.
   */
  const deliverResultVideos = async (
    projectId: string,
    queueItem: QueueItem,
    backendItemIds: number[]
  ): Promise<QueueResultVideo[]> => {
    // Skip known cancelled backend items. Resumed cancellation records can be incomplete, so remaining attachments
    // are best-effort.
    const deliverableItemIds = backendItemIds.filter(
      (backendItemId) => !queueItem.cancelledBackendItemIds?.includes(backendItemId)
    );

    if (!isActive() || queueItem.snapshot.destination !== 'gallery' || deliverableItemIds.length === 0) {
      return [];
    }

    const recordError = (error: unknown): void => {
      if (isActive()) {
        commands.recordError({
          area: 'queue-results',
          message: toErrorMessage(error),
          namespace: 'queue',
          projectId,
        });
      }
    };
    let namesPerItem: string[][];

    try {
      const imageOptions = getQueueItemResultImageOptions(queueItem);
      const options = queueItem.snapshot.filterIntermediateResults
        ? { ...imageOptions, excludeIntermediate: true }
        : imageOptions;
      namesPerItem = await mapWithConcurrency(deliverableItemIds, QUEUE_RUNTIME_CONCURRENCY, (backendItemId) =>
        runResultRead(() => backend.getResultVideoNames(backendItemId, options))
      );
    } catch (error) {
      recordError(error);
      return [];
    }

    // Exclude video primitives' echoed input clips from generated results.
    const inputMedia = collectGraphInputMediaNames(getQueueItemCompiledGraph(queueItem));
    const videoNames = [...new Set(namesPerItem.flat())].filter((name) => !inputMedia.videoNames.has(name));

    if (videoNames.length === 0 || !isActive()) {
      return [];
    }

    const readVideos = (names: string[]): Promise<QueueResultVideo[]> =>
      runResultRead(() => backend.getResultVideos(names, queueItem.id, queueItem.snapshot.submittedAt));
    let videos: QueueResultVideo[];

    try {
      videos = await readVideos(videoNames);
    } catch (error) {
      recordError(error);
      return [];
    }

    if (!isActive()) {
      return [];
    }

    const boardId = getResultDestinationBoardId(queueItem);
    const unassignedNames = videos.filter((video) => !video.boardId).map((video) => video.videoName);

    // Hydration drops unreadable videos; their board is unknown, so they are neither shown nor attached.
    if (videos.length < videoNames.length) {
      const unreadCount = videoNames.length - videos.length;
      const consequence = boardId ? '; they were not added to the board' : '';
      recordError(new Error(`${unreadCount} of ${videoNames.length} result video(s) could not be read${consequence}.`));
    }

    if (!boardId || unassignedNames.length === 0) {
      return videos;
    }

    try {
      await destinations.addVideosToGalleryBoard(boardId, unassignedNames);
    } catch (error) {
      // A board-attach failure must not keep a finished video out of view.
      recordError(error);
    }

    if (!isActive()) {
      return [];
    }

    // Re-read only the attach candidates so each reports the board it reached: the destination port skips virtual
    // boards, and a partial failure attaches only some videos.
    try {
      const attached = new Map((await readVideos(unassignedNames)).map((video) => [video.videoName, video]));

      return videos.map((video) => attached.get(video.videoName) ?? video);
    } catch (error) {
      recordError(error);
      return videos;
    }
  };

  /** Drop input passthroughs and requested intermediates; echoed source media must not move boards on every run. */
  const selectVisibleImages = (queueItem: QueueItem, allImages: QueueResultImage[]): QueueResultImage[] => {
    const inputMedia = collectGraphInputMediaNames(getQueueItemCompiledGraph(queueItem));
    const producedImages = allImages.filter((image) => !inputMedia.imageNames.has(image.imageName));

    return queueItem.snapshot.filterIntermediateResults
      ? producedImages.filter((image) => !image.isIntermediate)
      : producedImages;
  };

  const deliverVisibleImages = async (
    queueItem: QueueItem,
    allImages: QueueResultImage[]
  ): Promise<QueueResultImage[]> => {
    const images = selectVisibleImages(queueItem, allImages);

    await addImagesToDestination(queueItem, images);

    return images;
  };

  const routeRunResults = async (
    coordinator: QueueCoordinator,
    projectId: string,
    queueItem: QueueItem,
    // Pass actual backend IDs explicitly; the immutable pre-submit queueItem closure cannot contain newly assigned
    // IDs.
    backendItemIds: number[],
    attempt: RunAttempt
  ): Promise<void> => {
    try {
      const allImages = await coordinator.waitForResults(
        queueItem.id,
        queueItem.snapshot.submittedAt,
        getQueueItemResultImageOptions(queueItem)
      );

      if (!isAttemptCurrent(attempt)) {
        return;
      }
      const target = getRouteTarget(history, queueItem.id);
      if (
        target?.project.id !== projectId ||
        !target.queueItem ||
        (target.queueItem.status !== 'pending' && target.queueItem.status !== 'running')
      ) {
        return;
      }

      const images = await deliverVisibleImages(queueItem, allImages);
      // The run-end video pass retries live routing and covers earlier-session completions; it must never throw.
      const videos = await deliverResultVideos(projectId, queueItem, backendItemIds);

      if (!isAttemptCurrent(attempt)) {
        return;
      }

      confirmAdoptedReruns(projectId, queueItem, backendItemIds);
      commands.routeResults({ images, projectId, queueItemId: queueItem.id, videos });
      if (queueItem.snapshot.destination === 'gallery') {
        commands.refreshBackendData();
      }
    } catch (error) {
      if (!isAttemptCurrent(attempt)) {
        return;
      }

      if (error instanceof QueueItemCancelledError) {
        commands.setStatus({ projectId, queueItemId: queueItem.id, status: 'cancelled' });
        return;
      }

      commands.setStatus({
        error: toErrorMessage(error),
        projectId,
        queueItemId: queueItem.id,
        status: 'failed',
      });
    }
  };

  const routeBackendItemResults = async (
    projectId: string,
    queueItem: QueueItem,
    backendItemId: number,
    attempt: RunAttempt
  ): Promise<void> => {
    try {
      const images = isAdoptedRerunDecidedHere(projectId, queueItem, backendItemId)
        ? await runResultRead(() =>
            backend.getResultImages(
              backendItemId,
              queueItem.id,
              queueItem.snapshot.submittedAt,
              getQueueItemResultImageOptions(queueItem)
            )
          )
        : [];

      if (!isAttemptCurrent(attempt)) {
        return;
      }
      const target = getRouteTarget(history, queueItem.id);
      if (
        target?.project.id !== projectId ||
        !target.queueItem ||
        (target.queueItem.status !== 'pending' && target.queueItem.status !== 'running')
      ) {
        return;
      }

      // A renewal may have found the claim lost while the results loaded; another claimant stages the rerun then.
      const visibleImages = isAdoptedRerunDecidedHere(projectId, queueItem, backendItemId)
        ? await deliverVisibleImages(queueItem, images)
        : [];
      // Never throws — a board-attach hiccup must not block routePartialResults below.
      const videos = await deliverResultVideos(projectId, queueItem, [backendItemId]);

      if (!isAttemptCurrent(attempt)) {
        return;
      }

      // The held denoise frame may be painted over exactly these images while
      // they decode — not over an earlier image of the same batch.
      progressImageStore.bindSwapImages(
        queueItem.id,
        visibleImages.map((image) => image.imageName)
      );
      confirmAdoptedReruns(projectId, queueItem, [backendItemId]);
      commands.routePartialResults({
        backendItemId,
        images: visibleImages,
        projectId,
        queueItemId: queueItem.id,
        videos,
      });
      if (queueItem.snapshot.destination === 'gallery') {
        commands.refreshBackendData();
      }
    } catch (error) {
      if (isAttemptCurrent(attempt)) {
        commands.recordError({
          area: 'queue-results',
          message: toErrorMessage(error),
          namespace: 'queue',
          projectId,
        });
      }
    }
  };

  interface PendingResultRoute {
    attempt: RunAttempt;
    backendItemId: number;
    projectId: string;
    queueItem: QueueItem;
    /** Resolves when THIS route has run (or will never run), not when the shared flush drains. */
    settled: Promise<void>;
    settle: () => void;
  }
  const pendingResultRoutes = new Map<string, PendingResultRoute>();
  let resultRoutingFlush: Promise<void> | undefined;
  const scheduleResultRoute = (
    projectId: string,
    queueItem: QueueItem,
    backendItemId: number,
    attempt: RunAttempt
  ): Promise<void> => {
    const key = JSON.stringify([attempt.generation, projectId, queueItem.id, backendItemId]);
    const existing = pendingResultRoutes.get(key);
    let settle: () => void = () => undefined;
    const settled =
      existing?.settled ??
      new Promise<void>((resolve) => {
        settle = resolve;
      });
    pendingResultRoutes.set(key, {
      attempt,
      backendItemId,
      projectId,
      queueItem,
      settle: existing?.settle ?? settle,
      settled,
    });
    if (!resultRoutingFlush) {
      const flush = Promise.resolve().then(async () => {
        while (isActive() && pendingResultRoutes.size > 0) {
          const batch = [...pendingResultRoutes.values()];
          pendingResultRoutes.clear();
          await mapWithConcurrency(batch, QUEUE_RUNTIME_CONCURRENCY, (route) =>
            routeBackendItemResults(route.projectId, route.queueItem, route.backendItemId, route.attempt).finally(
              route.settle
            )
          );
        }
        // Inactive: whatever is left will never run. Release its awaiters — a
        // followed slot must not hang on a route that is never coming.
        for (const route of pendingResultRoutes.values()) {
          route.settle();
        }
        pendingResultRoutes.clear();
      });
      resultRoutingFlush = flush.finally(() => {
        resultRoutingFlush = undefined;
      });
    }
    return settled;
  };

  const coordinatorBackend: QueueBackendPort = {
    ...backend,
    getResultImages: async (itemId, localQueueItemId, ...rest) => {
      const target = getRouteTarget(history, localQueueItemId);
      if (target?.queueItem && !(await isAdoptedRerunStagedHere(target.project.id, target.queueItem, itemId))) {
        return [];
      }
      return runResultRead(() => backend.getResultImages(itemId, localQueueItemId, ...rest));
    },
  };
  const coordinator = createQueueCoordinator(
    {
      onBackendItemCancelled: (localQueueItemId, backendItemId) => {
        if (!isActive()) {
          return;
        }

        const target = getRouteTarget(history, localQueueItemId);

        if (!target?.queueItem || target.queueItem.cancelledBackendItemIds?.includes(backendItemId)) {
          return;
        }

        commands.markBackendCancelled({
          backendItemId,
          projectId: target.project.id,
          queueItemId: localQueueItemId,
        });
      },
      onBackendItemComplete: (localQueueItemId, backendItemId) => {
        if (!isActive()) {
          return;
        }

        const target = getRouteTarget(history, localQueueItemId);

        if (
          !target?.queueItem ||
          target.queueItem.status === 'cancelled' ||
          target.queueItem.status === 'failed' ||
          target.queueItem.status === 'completed' ||
          target.queueItem.completedBackendItemIds?.includes(backendItemId)
        ) {
          return;
        }

        const attempt = activeAttempts.get(getRunKey(target.project.id, target.queueItem.id));
        if (attempt) {
          return scheduleResultRoute(target.project.id, target.queueItem, backendItemId, attempt);
        }
      },
      onGalleryRefresh: () => {
        if (isActive()) {
          commands.refreshBackendData();
        }
      },
    },
    { backend: coordinatorBackend, modelLoads, nodeExecution }
  );

  /** Backend ids of reruns no run in this tab adopted. */
  const unadoptedReruns = new Set<number>();
  const rememberUnadoptedRerun = (itemId: number): void => {
    unadoptedReruns.add(itemId);
    for (const oldest of unadoptedReruns) {
      if (unadoptedReruns.size <= UNADOPTED_RERUN_LIMIT) {
        break;
      }
      unadoptedReruns.delete(oldest);
    }
  };

  /** Ends every wait on another tab's run, or on a rerun another claimant holds, once this runtime stops. */
  const rerunWaits = new AbortController();
  const waitForRerunRetry = (delayMs: number): Promise<void> =>
    new Promise((resolve) => {
      const timer = setTimeout(resolve, delayMs);
      rerunWaits.signal.addEventListener(
        'abort',
        () => {
          clearTimeout(timer);
          resolve();
        },
        { once: true }
      );
    });
  /** This tab, as the claimant of the reruns it stages outside any run. */
  const tabClaimant = `tab:${createUuid()}`;
  /** A run, as the claimant of the reruns it was reopened around; the run restored by a reload claims as it did. */
  const getRunClaimant = (projectId: string, queueItemId: string): string => `run:${getRunKey(projectId, queueItemId)}`;

  /**
   * Whether this claimant stages a retried Canvas item, as the server agrees it for every browser and tab of the
   * account. While another claimant holds the item it asks again, until that one confirms or its claim expires; a
   * request that fails is asked again for as long as a claim could last. A server without claims (a 404) lets every
   * claimant stage it, as before claims existed.
   */
  const claimRerunDelivery = async (itemId: number, claimant: string, canDeliver: () => boolean): Promise<boolean> => {
    for (let failures = 0; ;) {
      let claim: QueueItemDeliveryClaim;
      try {
        claim = await backend.claimItemDelivery(itemId, 'claim', claimant);
      } catch (error) {
        failures += 1;
        if (
          (error instanceof ApiError && error.status === 404) ||
          failures * RERUN_CLAIM_POLL_MS >= RERUN_CLAIM_TTL_MS
        ) {
          return canDeliver();
        }
        claim = { kind: 'held', retryInMs: RERUN_CLAIM_POLL_MS };
      }
      if (claim.kind !== 'held') {
        return claim.kind === 'claimed';
      }
      // Soon enough to see the holder confirm, and spread out so claimants turned away together do not ask together.
      await waitForRerunRetry(Math.min(claim.retryInMs, RERUN_CLAIM_POLL_MS) * (1 + Math.random() / 2));
      if (!canDeliver()) {
        return false;
      }
    }
  };
  /** Whether this claimant still holds an item it claimed, renewing the claim; a server that cannot answer keeps it. */
  const isRerunClaimHeld = async (itemId: number, claimant: string): Promise<boolean> => {
    try {
      return (await backend.claimItemDelivery(itemId, 'claim', claimant)).kind === 'claimed';
    } catch {
      return true;
    }
  };
  const settleRerunClaim = (itemId: number, action: 'confirm' | 'release', claimant: string): void => {
    void backend.claimItemDelivery(itemId, action, claimant).catch(() => undefined);
  };

  /**
   * The Canvas reruns the runs of this tab were reopened around, each claimed by its run as soon as the tab holds the
   * run and renewed while the run lasts, so a tab or browser that sees the rerun finish first leaves it to the run.
   */
  interface AdoptedRerunClaim {
    decision: Promise<boolean>;
    isClaimed: boolean;
    isConfirmed: boolean;
  }
  const adoptedRerunClaims = new Map<string, AdoptedRerunClaim>();
  const getAdoptedRerunKey = (projectId: string, queueItemId: string, backendItemId: number): string =>
    JSON.stringify([projectId, queueItemId, backendItemId]);
  const isAdoptedRerun = (queueItem: QueueItem, backendItemId: number): boolean =>
    queueItem.snapshot.destination === 'canvas' &&
    Boolean(queueItem.retriedBackendItemIds?.length) &&
    queueItem.backendItemIds?.includes(backendItemId) === true;
  const claimAdoptedRerun = (projectId: string, queueItemId: string, backendItemId: number): AdoptedRerunClaim => {
    const key = getAdoptedRerunKey(projectId, queueItemId, backendItemId);
    const existing = adoptedRerunClaims.get(key);
    if (existing) {
      return existing;
    }
    const claim: AdoptedRerunClaim = {
      // A claim the run let go of meanwhile stops asking.
      decision: claimRerunDelivery(
        backendItemId,
        getRunClaimant(projectId, queueItemId),
        () => isActive() && adoptedRerunClaims.get(key) === claim
      ).then((isClaimed) => {
        claim.isClaimed = isClaimed;
        return isClaimed;
      }),
      isClaimed: false,
      isConfirmed: false,
    };
    adoptedRerunClaims.set(key, claim);
    return claim;
  };
  /** Claim the Canvas reruns of a run this tab holds that it has yet to stage. */
  const claimAdoptedReruns = (projectId: string, queueItem: QueueItem): void => {
    for (const backendItemId of queueItem.backendItemIds ?? []) {
      if (isAdoptedRerun(queueItem, backendItemId) && !queueItem.completedBackendItemIds?.includes(backendItemId)) {
        claimAdoptedRerun(projectId, queueItem.id, backendItemId);
      }
    }
  };
  /** Keep the claims of runs still going from lapsing; a claim lost meanwhile leaves its rerun out of the run. */
  const renewAdoptedRerunClaims = (): void => {
    for (const [key, claim] of adoptedRerunClaims) {
      if (!claim.isClaimed || claim.isConfirmed) {
        continue;
      }
      const [projectId, queueItemId, backendItemId] = JSON.parse(key) as [string, string, number];
      void backend
        .claimItemDelivery(backendItemId, 'claim', getRunClaimant(projectId, queueItemId))
        .then((renewal) => {
          if (renewal.kind !== 'claimed' && adoptedRerunClaims.get(key) === claim && !claim.isConfirmed) {
            claim.isClaimed = false;
          }
        })
        .catch(() => undefined);
    }
  };
  /**
   * Whether a run stages a rerun it was reopened around. A rerun another tab or browser staged first, as when the run's
   * tab had closed, is left out of the run's results, so a run restored after that never stages it again.
   */
  const isAdoptedRerunStagedHere = async (
    projectId: string,
    queueItem: QueueItem,
    backendItemId: number
  ): Promise<boolean> => {
    if (!isAdoptedRerun(queueItem, backendItemId)) {
      return true;
    }
    const claim = claimAdoptedRerun(projectId, queueItem.id, backendItemId);
    await claim.decision;
    return claim.isClaimed;
  };
  /**
   * Whether a run stages a rerun now, without waiting on a claim still undecided; the run's end stages it once decided.
   * Result routing is shared by every run of the tab, so one claim must not hold the others up.
   */
  const isAdoptedRerunDecidedHere = (projectId: string, queueItem: QueueItem, backendItemId: number): boolean =>
    !isAdoptedRerun(queueItem, backendItemId) ||
    adoptedRerunClaims.get(getAdoptedRerunKey(projectId, queueItem.id, backendItemId))?.isClaimed === true;
  /**
   * Confirm the reruns a run is about to stage. Called just before their results are routed, since routing them can
   * finish the run, and a finished run gives back the claims it did not confirm.
   */
  const confirmAdoptedReruns = (projectId: string, queueItem: QueueItem, backendItemIds: number[]): void => {
    for (const backendItemId of backendItemIds) {
      const claim = adoptedRerunClaims.get(getAdoptedRerunKey(projectId, queueItem.id, backendItemId));
      if (claim?.isClaimed && !claim.isConfirmed) {
        claim.isConfirmed = true;
        settleRerunClaim(backendItemId, 'confirm', getRunClaimant(projectId, queueItem.id));
      }
    }
  };
  /**
   * A run this tab lets go of, whether it ended or its project closed, gives back the claims it did not confirm, so
   * another tab stages those reruns at once; a later hold of the run claims again.
   */
  const releaseAdoptedReruns = (projectId: string, queueItemId: string): void => {
    for (const [key, claim] of adoptedRerunClaims) {
      const [keyProjectId, keyQueueItemId, backendItemId] = JSON.parse(key) as [string, string, number];
      if (keyProjectId !== projectId || keyQueueItemId !== queueItemId) {
        continue;
      }
      adoptedRerunClaims.delete(key);
      if (!claim.isConfirmed) {
        void claim.decision.then((isClaimed) => {
          if (isClaimed) {
            settleRerunClaim(backendItemId, 'release', getRunClaimant(projectId, queueItemId));
          }
        });
      }
    }
  };

  /**
   * A retry, from any tab or client, re-runs failed or canceled backend items under new ids. The open run that owns
   * the retried items, or once owned them before an earlier retry, reopens around the reruns, so its captured
   * destination receives their results; reconciliation then tracks them like any resumed run.
   */
  const handleItemsRetried = (event: QueueItemsRetriedEvent): void => {
    if (!isActive() || !event.retried_items?.length) {
      return;
    }

    const runsByBackendItemId = new Map<number, { projectId: string; queueItemId: string }>();
    const trackedItemIds = new Set<number>();
    for (const project of history.getSnapshot().projects) {
      for (const queueItem of project.queue.items) {
        for (const backendItemId of [...(queueItem.backendItemIds ?? []), ...(queueItem.retriedBackendItemIds ?? [])]) {
          runsByBackendItemId.set(backendItemId, { projectId: project.id, queueItemId: queueItem.id });
        }
        for (const backendItemId of queueItem.backendItemIds ?? []) {
          trackedItemIds.add(backendItemId);
        }
      }
    }
    // An admin's socket carries every user's retries; only the viewer's own can belong to runs or projects here.
    const viewerSourceIds =
      viewerUserId && event.user_ids.length > 0 ? new Set(event.retried_item_ids_by_user[viewerUserId] ?? []) : null;

    const retriedItemsByRun = new Map<
      string,
      { projectId: string; queueItemId: string; retriedItems: { itemId: number; sourceItemId: number }[] }
    >();
    for (const { item_id: itemId, source_item_id: sourceItemId } of event.retried_items) {
      // The retrying tab sees its pairs twice, from its own response and from the socket; the first one handles them.
      if (
        (viewerSourceIds && !viewerSourceIds.has(sourceItemId)) ||
        trackedItemIds.has(itemId) ||
        unadoptedReruns.has(itemId)
      ) {
        continue;
      }
      const owner = runsByBackendItemId.get(sourceItemId);

      if (!owner) {
        rememberUnadoptedRerun(itemId);
        continue;
      }
      const key = getRunKey(owner.projectId, owner.queueItemId);
      const run = retriedItemsByRun.get(key) ?? { ...owner, retriedItems: [] };
      run.retriedItems.push({ itemId, sourceItemId });
      retriedItemsByRun.set(key, run);
    }

    for (const [key, run] of retriedItemsByRun) {
      commands.reopenForRetry(run);
      const reopened = getRouteTarget(history, run.queueItemId)?.queueItem;
      let isAdopted = false;

      for (const { itemId } of run.retriedItems) {
        if (reopened?.status === 'running' && reopened.backendItemIds?.includes(itemId)) {
          isAdopted = true;
        } else {
          rememberUnadoptedRerun(itemId);
        }
      }
      if (isAdopted) {
        // A settle still waiting on the journal belongs to the finished run; it must not remove the reopened record.
        pendingJournalSettles.delete(key);
        pendingReconcileRunKeys.add(key);
      }
    }
    synchronize();
  };

  /**
   * The images a rerun no run adopted produced. Its run, when this tab still holds it, says which nodes are results;
   * otherwise the product sources' result nodes are tried before every image the rerun created, as for a Workflow.
   */
  const readUnadoptedRerunImages = async (
    event: QueueItemStatusChangedEvent,
    queueItemId: string,
    run: QueueItem | undefined
  ): Promise<QueueResultImage[]> => {
    const read = (options?: QueueResultImageOptions) =>
      runResultRead(() => backend.getResultImages(event.item_id, queueItemId, event.created_at, options));

    if (run) {
      return selectVisibleImages(run, await read(getQueueItemResultImageOptions(run)));
    }

    const productResults = await read({
      resultNodeIds: [
        ...(QUEUE_SOURCE_RESULT_NODE_IDS.generate ?? []),
        ...(QUEUE_SOURCE_RESULT_NODE_IDS.upscale ?? []),
      ],
    });
    if (productResults.length > 0) {
      return productResults;
    }
    // Images older than the rerun are inputs its graph echoed back, not results.
    const queuedAt = Date.parse(normalizeServerTimestamp(event.created_at));
    return (await read()).filter(
      (image) => !image.isIntermediate && (!image.createdAt || Date.parse(image.createdAt) >= queuedAt)
    );
  };

  /**
   * Whether another tab tracks the run behind a Canvas rerun no run adopted. That tab knows the run's result nodes, so
   * it stages the rerun and this one waits for it; the lock is only looked at here, so that tab is never turned away.
   * Tabs that do not track the run, and other browsers, are kept apart by the server's delivery claim instead.
   */
  const isRunTrackedElsewhere = async (projectId: string, queueItemId: string): Promise<boolean> => {
    const key = getRunKey(projectId, queueItemId);
    if (!locks || ownedLocks.has(key) || lockRequests.has(key) || releasingRuns.has(key)) {
      return false;
    }
    try {
      return (await locks.isHeld(projectId, queueItemId)) === true;
    } catch {
      return false;
    }
  };

  /**
   * Deliver a rerun no open run adopted the way its destination would see any untracked result: the Gallery refreshes
   * (onto its run's board when this tab still holds that run), and one tab with its Canvas project open stages it.
   */
  const handleUnadoptedRerunSettled = (event: QueueItemStatusChangedEvent): void => {
    if (!isActive() || !isTerminalBackendStatus(event.status) || !unadoptedReruns.delete(event.item_id)) {
      return;
    }
    if (event.status !== 'completed' || isUtilityQueueItemOrigin(event.origin)) {
      return;
    }

    const projectId = parseQueueItemOriginProjectId(event.origin);
    const queueItemId = parseQueueItemOrigin(event.origin);
    const target = queueItemId ? getRouteTarget(history, queueItemId) : undefined;
    const run = target?.project.id === projectId ? target?.queueItem : undefined;
    const isProjectOpen = history.getSnapshot().projects.some((project) => project.id === projectId);
    const recordError = (error: unknown): void => {
      if (isActive()) {
        commands.recordError({
          area: 'queue-results',
          message: toErrorMessage(error),
          namespace: 'queue',
          ...(projectId ? { projectId } : {}),
        });
      }
    };

    if (event.destination === 'gallery') {
      if (!run || !target || !queueItemId) {
        commands.refreshBackendData();
        return;
      }
      // Video graphs leave their output unboarded, so videos need the same board routing as an adopted rerun's.
      void Promise.all([
        readUnadoptedRerunImages(event, queueItemId, run).then((images) => addImagesToDestination(run, images)),
        deliverResultVideos(target.project.id, run, [event.item_id]),
      ])
        .catch(recordError)
        .finally(() => {
          if (isActive()) {
            commands.refreshBackendData();
          }
        });
      return;
    }
    if (event.destination !== 'canvas' || !projectId || !queueItemId || !isProjectOpen) {
      return;
    }

    const canStage = (): boolean =>
      isActive() && history.getSnapshot().projects.some((project) => project.id === projectId);
    const stage = async (): Promise<void> => {
      // A tab that tracks the run knows its result nodes and stages the rerun itself; this one waits until that tab lets
      // the run go, and then stages only a rerun it left.
      if (await isRunTrackedElsewhere(projectId, queueItemId)) {
        await locks?.waitForRunRelease(projectId, queueItemId, rerunWaits.signal);
      }
      if (!canStage() || !(await claimRerunDelivery(event.item_id, tabClaimant, canStage))) {
        return;
      }
      // Confirmed only once staged, so a tab that cannot stage it, or closes first, leaves the rerun to the next.
      let isStaged = false;
      try {
        const images = await readUnadoptedRerunImages(event, queueItemId, run);
        // A read slower than the claim lasts may have let another claimant stage it meanwhile.
        if (images.length > 0 && canStage() && (await isRerunClaimHeld(event.item_id, tabClaimant)) && canStage()) {
          commands.stageRetriedCanvasResults({ backendItemId: event.item_id, images, projectId, queueItemId });
          isStaged = true;
        }
      } finally {
        settleRerunClaim(event.item_id, isStaged ? 'confirm' : 'release', tabClaimant);
      }
    };
    void stage().catch(recordError);
  };

  const submitQueueItem = async (
    project: QueueHistoryProject,
    queueItem: QueueItem,
    attempt: RunAttempt
  ): Promise<void> => {
    const runKey = getRunKey(project.id, queueItem.id);
    const canSubmit = (item: QueueItem): boolean =>
      shouldSubmitPendingQueueItem(item) || (item.status === 'cancelled' && cancelRequestedRunKeys.has(runKey));
    if (!(await acquireRunOwnership(project.id, queueItem.id))) {
      startedRunKeys.delete(runKey);
      reconcileRetryRunKeys.add(runKey);
      scheduleRetry();
      return;
    }
    if (!isAttemptCurrent(attempt)) {
      if (!activeAttempts.has(attempt.key)) {
        releaseRunOwnershipAfterJournal(project.id, queueItem.id);
      }
      return;
    }
    let target = getRouteTarget(history, queueItem.id);
    if (target?.project.id !== project.id || !target.queueItem || !canSubmit(target.queueItem)) {
      releaseRunOwnership(project.id, queueItem.id);
      return;
    }
    await recordActiveRun(project.id, target.queueItem);
    if (!isAttemptCurrent(attempt)) {
      return;
    }
    target = getRouteTarget(history, queueItem.id);
    if (target?.project.id !== project.id || !target.queueItem || !canSubmit(target.queueItem)) {
      settleRun(project.id, queueItem.id);
      return;
    }

    const currentQueueItem = target.queueItem;
    const projectPersistence = ensureProjectPersisted
      ? await ensureProjectPersisted(project.id).catch(() => 'retry' as const)
      : 'ready';
    if (!isAttemptCurrent(attempt)) {
      return;
    }
    if (projectPersistence === 'retry' || history.getSnapshot().connectionStatus !== 'connected') {
      startedRunKeys.delete(runKey);
      reconcileRetryRunKeys.add(runKey);
      scheduleRetry();
      return;
    }
    if (projectPersistence === 'refused') {
      settleRun(project.id, currentQueueItem.id);
      commands.setStatus({
        error: 'This run was not submitted because its project cannot be saved.',
        projectId: project.id,
        queueItemId: currentQueueItem.id,
        status: 'failed',
      });
      return;
    }
    const submission = createQueueItemBackendSubmission(project, currentQueueItem);

    if (submission.kind === 'invalid') {
      commands.setStatus({
        error: submission.error,
        projectId: project.id,
        queueItemId: currentQueueItem.id,
        status: 'failed',
      });
      return;
    }

    const request =
      submission.kind === 'generate'
        ? coordinator.submitGenerate(currentQueueItem.id, submission.request)
        : coordinator.submitWorkflow(
            currentQueueItem.id,
            submission.request,
            getQueueRunOrigin(project.id, currentQueueItem)
          );
    submissionOperations.set(runKey, request);

    await request
      .then(async ({ batchId, enqueued, itemIds, requested }) => {
        if (!isAttemptCurrent(attempt)) {
          return;
        }

        const target = getRouteTarget(history, currentQueueItem.id);
        if (target?.project.id !== project.id || !target.queueItem) {
          return cancelRun(project.id, currentQueueItem, itemIds, batchId);
        }
        if (target.queueItem.status === 'cancelled') {
          return cancelRun(project.id, target.queueItem, itemIds, batchId);
        }

        if (enqueued < requested) {
          commands.recordError({
            area: 'queue-submission',
            message: `The backend queue accepted ${enqueued} of ${requested} requested items. The accepted items will continue.`,
            namespace: 'queue',
            projectId: project.id,
          });
        }

        commands.markBackendSubmitted({
          backendBatchId: batchId,
          backendItemIds: itemIds,
          projectId: project.id,
          queueItemId: currentQueueItem.id,
        });
        const submittedTarget = getRouteTarget(history, currentQueueItem.id);
        if (submittedTarget?.project.id !== project.id || !submittedTarget.queueItem) {
          return cancelRun(project.id, currentQueueItem, itemIds, batchId);
        }
        await recordActiveRun(project.id, submittedTarget.queueItem);
        void backend.resumeProcessor().catch(() => undefined);

        return routeRunResults(coordinator, project.id, currentQueueItem, itemIds, attempt);
      })
      .catch((error: unknown) => {
        if (!isAttemptCurrent(attempt)) {
          return;
        }
        if (error instanceof QueueEnqueueNotAcceptedError) {
          const target = getRouteTarget(history, currentQueueItem.id);
          if (target?.project.id === project.id && target.queueItem?.cancellationPending === true) {
            completeCancellation(project.id, currentQueueItem.id);
          } else {
            settleRun(project.id, currentQueueItem.id);
            commands.setStatus({
              error: error.message,
              projectId: project.id,
              queueItemId: currentQueueItem.id,
              status: 'failed',
            });
          }
          return;
        }
        if (
          error instanceof ApiError &&
          error.status >= 400 &&
          error.status < 500 &&
          ![408, 429].includes(error.status)
        ) {
          commands.setStatus({
            error: toErrorMessage(error),
            projectId: project.id,
            queueItemId: currentQueueItem.id,
            status: 'failed',
          });
          return;
        }
        startedRunKeys.delete(runKey);
        reconcileRetryRunKeys.add(runKey);
        scheduleRetry();
      })
      .finally(() => {
        if (submissionOperations.get(runKey) === request) {
          submissionOperations.delete(runKey);
        }
      });
  };

  const synchronizeRunPersistence = (): void => {
    if (!hasInitialized || !isActive()) {
      return;
    }
    const activeKeys = new Set<string>();
    const projects = history.getSnapshot().projects;
    const openProjectIds = new Set(projects.map((project) => project.id));
    for (const project of projects) {
      for (const item of project.queue.items) {
        const key = getRunKey(project.id, item.id);
        if (item.status === 'pending' || item.status === 'running') {
          activeKeys.add(key);
          if (!locks || ownedLocks.has(key) || trackedRuns.has(key)) {
            void recordActiveRun(project.id, item);
            claimAdoptedReruns(project.id, item);
          }
        } else if (item.status === 'cancelled' && item.cancellationPending === true) {
          activeKeys.add(key);
          cancelRequestedRunKeys.add(key);
          if (trackedRuns.has(key)) {
            void recordActiveRun(project.id, item);
            if (
              history.getSnapshot().connectionStatus === 'connected' &&
              (item.backendBatchId || item.backendItemIds?.length)
            ) {
              void cancelRun(project.id, item);
            } else if (!submissionOperations.has(key) && !pendingReconcileRunKeys.has(key)) {
              pendingReconcileRunKeys.add(key);
              startedRunKeys.delete(key);
            }
          }
        } else if (trackedRuns.has(key)) {
          settleRun(project.id, item.id);
        }
      }
    }
    for (const [key, tracked] of trackedRuns) {
      if (!activeKeys.has(key)) {
        if (openProjectIds.has(tracked.projectId)) {
          settleRun(tracked.projectId, tracked.queueItemId);
        } else {
          coordinator.detachRun(tracked.queueItemId);
          releaseAdoptedReruns(tracked.projectId, tracked.queueItemId);
          trackedRuns.delete(key);
          startedRunKeys.delete(key);
          pendingReconcileRunKeys.delete(key);
          reconcileRetryRunKeys.delete(key);
          reportedCancelFailures.delete(key);
          pendingJournalRecords.delete(key);
          releaseRunOwnershipAfterJournal(tracked.projectId, tracked.queueItemId);
        }
      }
    }
    for (const [key, attempt] of activeAttempts) {
      if (activeKeys.has(key)) {
        continue;
      }
      activeAttempts.delete(key);
      startedRunKeys.delete(key);
      pendingReconcileRunKeys.delete(key);
      reconcileRetryRunKeys.delete(key);
      coordinator.detachRun(attempt.queueItemId);
      releaseAdoptedReruns(attempt.projectId, attempt.queueItemId);
      if (!trackedRuns.has(key)) {
        releaseRunOwnershipAfterJournal(attempt.projectId, attempt.queueItemId);
      }
    }
  };

  const prepareActiveRuns = (): void => {
    if (!hasInitialized || !isActive()) {
      return;
    }
    const candidates = history.getSnapshot().projects.flatMap((project) =>
      project.queue.items
        .filter((item) => {
          const key = getRunKey(project.id, item.id);
          return (
            !trackedRuns.has(key) &&
            !preparingRunKeys.has(key) &&
            !reconcileRetryRunKeys.has(key) &&
            (item.status === 'pending' || item.status === 'running' || item.cancellationPending === true)
          );
        })
        .map((item) => ({ attempt: beginAttempt(project.id, item.id), item, project }))
    );
    if (candidates.length === 0) {
      return;
    }
    for (const { item, project } of candidates) {
      preparingRunKeys.add(getRunKey(project.id, item.id));
    }
    void mapWithConcurrency(candidates, QUEUE_RUNTIME_CONCURRENCY, async ({ attempt, item, project }) => {
      const key = getRunKey(project.id, item.id);
      try {
        if (!(await acquireRunOwnership(project.id, item.id))) {
          reconcileRetryRunKeys.add(key);
          scheduleRetry();
          return;
        }
        if (!isAttemptCurrent(attempt)) {
          releaseRunOwnershipAfterJournal(project.id, item.id);
          return;
        }
        const target = getRouteTarget(history, item.id);
        if (
          target?.project.id !== project.id ||
          !target.queueItem ||
          (target.queueItem.status !== 'pending' &&
            target.queueItem.status !== 'running' &&
            target.queueItem.cancellationPending !== true)
        ) {
          releaseRunOwnership(project.id, item.id);
          return;
        }
        await recordActiveRun(project.id, target.queueItem);
      } finally {
        preparingRunKeys.delete(key);
      }
    }).finally(() => {
      if (isActive()) {
        synchronize();
      }
    });
  };

  const processQueueItems = (): void => {
    if (!hasInitialized || !isActive() || history.getSnapshot().connectionStatus !== 'connected' || isReconciling) {
      return;
    }

    for (const project of history.getSnapshot().projects) {
      if (!loadedProjectIds.has(project.id)) {
        continue;
      }
      for (const queueItem of project.queue.items) {
        const runKey = getRunKey(project.id, queueItem.id);
        if (
          shouldSubmitPendingQueueItem(queueItem) &&
          trackedRuns.has(runKey) &&
          !preparingRunKeys.has(runKey) &&
          !pendingReconcileRunKeys.has(runKey) &&
          !startedRunKeys.has(runKey)
        ) {
          startedRunKeys.add(runKey);
          void submitQueueItem(project, queueItem, beginAttempt(project.id, queueItem.id));
        }
      }
    }
  };

  const reconcile = (): void => {
    if (
      !hasInitialized ||
      !isActive() ||
      isReconciling ||
      !history.getSnapshot().isHydrated ||
      history.getSnapshot().connectionStatus !== 'connected'
    ) {
      return;
    }

    const openItems = history
      .getSnapshot()
      .projects.flatMap((project) =>
        project.queue.items
          .filter(
            (queueItem) =>
              pendingReconcileRunKeys.has(getRunKey(project.id, queueItem.id)) &&
              !preparingRunKeys.has(getRunKey(project.id, queueItem.id)) &&
              !startedRunKeys.has(getRunKey(project.id, queueItem.id)) &&
              (queueItem.status === 'pending' ||
                queueItem.status === 'running' ||
                (queueItem.status === 'cancelled' && cancelRequestedRunKeys.has(getRunKey(project.id, queueItem.id))))
          )
          .map((queueItem) => ({ project, queueItem }))
      );

    if (openItems.length === 0) {
      return;
    }
    isReconciling = true;
    const attempts = new Map<string, RunAttempt>();
    for (const { project, queueItem } of openItems) {
      const runKey = getRunKey(project.id, queueItem.id);
      startedRunKeys.add(runKey);
      pendingReconcileRunKeys.delete(runKey);
      attempts.set(runKey, beginAttempt(project.id, queueItem.id));
    }

    mapWithConcurrency(openItems, QUEUE_RUNTIME_CONCURRENCY, async (item) => ({
      ...item,
      attempt: attempts.get(getRunKey(item.project.id, item.queueItem.id))!,
      owned: await acquireRunOwnership(item.project.id, item.queueItem.id),
    }))
      .then(async (ownership) => {
        const ownedItems = ownership.flatMap((item) => {
          if (!item.owned) {
            startedRunKeys.delete(item.attempt.key);
            reconcileRetryRunKeys.add(item.attempt.key);
            scheduleRetry();
            return [];
          }
          if (!isAttemptCurrent(item.attempt)) {
            if (!activeAttempts.has(item.attempt.key)) {
              releaseRunOwnershipAfterJournal(item.project.id, item.queueItem.id);
            }
            return [];
          }
          const target = getRouteTarget(history, item.queueItem.id);
          if (
            target?.project.id === item.project.id &&
            target.queueItem &&
            (target.queueItem.status === 'pending' ||
              target.queueItem.status === 'running' ||
              (target.queueItem.status === 'cancelled' &&
                cancelRequestedRunKeys.has(getRunKey(item.project.id, item.queueItem.id))))
          ) {
            return [{ ...item, project: target.project, queueItem: target.queueItem }];
          }
          releaseRunOwnership(item.project.id, item.queueItem.id);
          return [];
        });
        if (!isActive() || ownedItems.length === 0) {
          return { outcomes: new Map<string, ReconcileOutcome>(), ownedItems };
        }
        await mapWithConcurrency(ownedItems, QUEUE_RUNTIME_CONCURRENCY, ({ project, queueItem }) =>
          recordActiveRun(project.id, queueItem)
        );
        if (!isActive()) {
          return { outcomes: new Map<string, ReconcileOutcome>(), ownedItems: [] };
        }
        const inputs: ReconcileInput[] = ownedItems.map(({ project, queueItem }) => ({
          backendBatchId: queueItem.backendBatchId,
          backendItemIds: queueItem.backendItemIds,
          hasWorkflowCall: queueItemHasWorkflowCall(queueItem),
          id: queueItem.id,
          origin: getQueueRunOrigin(project.id, queueItem),
          projectId: project.id,
          status: queueItem.status === 'running' ? 'running' : 'pending',
        }));
        return { outcomes: await coordinator.reconcile(inputs), ownedItems };
      })
      .then(({ outcomes, ownedItems }) => {
        if (!isActive()) {
          return;
        }

        if (ownedItems.length > 0) {
          reconcileRetryDelayMs = 100;
        }
        for (const { project, queueItem } of ownedItems) {
          const attempt = attempts.get(getRunKey(project.id, queueItem.id));
          if (!attempt || !isAttemptCurrent(attempt)) {
            continue;
          }
          const target = getRouteTarget(history, queueItem.id);
          if (
            target?.project.id !== project.id ||
            !target.queueItem ||
            (target.queueItem.status !== 'pending' &&
              target.queueItem.status !== 'running' &&
              !(
                target.queueItem.status === 'cancelled' &&
                cancelRequestedRunKeys.has(getRunKey(project.id, queueItem.id))
              ))
          ) {
            settleRun(project.id, queueItem.id);
            continue;
          }
          const currentQueueItem = target.queueItem;
          const runKey = getRunKey(project.id, currentQueueItem.id);
          const cancelRequested = cancelRequestedRunKeys.has(runKey);
          reconcileRetryRunKeys.delete(runKey);
          const outcome = outcomes.get(queueItem.id);

          if ((outcome?.kind === 'adopted' || outcome?.kind === 'resumed') && outcome.missingBackendItemIds?.length) {
            commands.recordError({
              area: 'queue-reconciliation',
              message: `${outcome.missingBackendItemIds.length} accepted queue item${outcome.missingBackendItemIds.length === 1 ? ' is' : 's are'} no longer available. The surviving items will continue.`,
              namespace: 'queue',
              projectId: project.id,
            });
          }

          if (cancelRequested) {
            if (outcome?.kind === 'adopted') {
              void cancelRun(project.id, currentQueueItem, outcome.backendItemIds, outcome.backendBatchId);
            } else if (outcome?.kind === 'resumed') {
              void cancelRun(project.id, currentQueueItem, outcome.backendItemIds ?? currentQueueItem.backendItemIds);
            } else if (outcome?.kind === 'enqueue') {
              completeCancellation(project.id, currentQueueItem.id);
            } else if (outcome?.kind === 'missing') {
              completeCancellation(project.id, currentQueueItem.id, outcome.backendItemIds);
            }
            continue;
          }

          switch (outcome?.kind) {
            case 'enqueue':
              startedRunKeys.delete(runKey);
              break;
            case 'adopted':
              commands.markBackendSubmitted({
                backendBatchId: outcome.backendBatchId,
                backendItemIds: outcome.backendItemIds,
                projectId: project.id,
                queueItemId: currentQueueItem.id,
              });
              void routeRunResults(coordinator, project.id, currentQueueItem, outcome.backendItemIds, attempt);
              break;
            case 'resumed':
              void routeRunResults(
                coordinator,
                project.id,
                currentQueueItem,
                outcome.backendItemIds ?? currentQueueItem.backendItemIds ?? [],
                attempt
              );
              break;
            case 'missing':
              if (outcome.backendItemIds?.length) {
                commands.markBackendSubmitted({
                  backendBatchId: outcome.backendBatchId,
                  backendItemIds: outcome.backendItemIds,
                  projectId: project.id,
                  queueItemId: currentQueueItem.id,
                });
              }
              commands.setStatus({
                error: 'This run is no longer on the backend queue (it may have been cleared).',
                notify: false,
                projectId: project.id,
                queueItemId: currentQueueItem.id,
                status: 'failed',
              });
              break;
          }
        }
      })
      .catch((error: unknown) => {
        if (!isActive()) {
          return;
        }

        commands.recordError({
          area: 'queue-reconciliation',
          message: `Queue reconciliation failed: ${toErrorMessage(error)}`,
          namespace: 'queue',
        });

        for (const { project, queueItem } of openItems) {
          const attempt = attempts.get(getRunKey(project.id, queueItem.id));
          if (!attempt || !isAttemptCurrent(attempt)) {
            continue;
          }
          reconcileRetryRunKeys.add(attempt.key);
          startedRunKeys.delete(attempt.key);
        }
        scheduleRetry();
      })
      .finally(() => {
        if (isActive()) {
          isReconciling = false;
          synchronize();
        }
      });
  };

  const loadProjectJournal = (projectId: string): void => {
    if (!journal) {
      loadedProjectIds.add(projectId);
      return;
    }
    loadingProjectIds.add(projectId);
    void journal
      .listForProject(projectId)
      .then((result) => {
        if (!isActive()) {
          return;
        }
        if (result.kind === 'available') {
          for (const entry of result.entries) {
            const key = getRunKey(projectId, entry.queueItemId);
            pendingReconcileRunKeys.add(key);
            const item = entry.item as Partial<QueueItem>;
            if (item.status === 'cancelled' && item.cancellationPending === true) {
              cancelRequestedRunKeys.add(key);
            }
          }
          commands.restoreFromJournal({ items: result.entries.map((entry) => entry.item), projectId });
        } else {
          reportJournalLoadFailure(projectId);
        }
      })
      .catch(() => reportJournalLoadFailure(projectId))
      .finally(() => {
        loadingProjectIds.delete(projectId);
        if (!isActive()) {
          return;
        }
        const project = history.getSnapshot().projects.find((candidate) => candidate.id === projectId);
        if (!project) {
          return;
        }
        loadedProjectIds.add(projectId);
        for (const item of project?.queue.items ?? []) {
          if (item.status === 'pending' || item.status === 'running') {
            pendingReconcileRunKeys.add(getRunKey(projectId, item.id));
          } else if (item.cancellationPending === true) {
            const key = getRunKey(projectId, item.id);
            pendingReconcileRunKeys.add(key);
            cancelRequestedRunKeys.add(key);
          }
        }
        synchronize();
      });
  };

  const discoverProjects = (): void => {
    if (!hasInitialized || !isActive()) {
      return;
    }
    const projects = history.getSnapshot().projects;
    const openProjectIds = new Set(projects.map((project) => project.id));
    for (const projectId of loadedProjectIds) {
      if (!openProjectIds.has(projectId)) {
        loadedProjectIds.delete(projectId);
      }
    }
    for (const project of projects) {
      if (!loadedProjectIds.has(project.id) && !loadingProjectIds.has(project.id)) {
        loadProjectJournal(project.id);
      }
    }
  };

  const initialize = (): void => {
    if (hasInitialized || isInitializing || !isActive() || !history.getSnapshot().isHydrated) {
      return;
    }
    isInitializing = true;
    const projectsAtStart = [...history.getSnapshot().projects];
    const initializeProjects = async (): Promise<void> => {
      if (journal) {
        for (const project of projectsAtStart) {
          const result = await journal.listForProject(project.id);
          if (!isActive()) {
            return;
          }
          if (result.kind === 'available') {
            for (const entry of result.entries) {
              const key = getRunKey(project.id, entry.queueItemId);
              pendingReconcileRunKeys.add(key);
              const item = entry.item as Partial<QueueItem>;
              if (item.status === 'cancelled' && item.cancellationPending === true) {
                cancelRequestedRunKeys.add(key);
              }
            }
            commands.restoreFromJournal({ items: result.entries.map((entry) => entry.item), projectId: project.id });
          } else {
            reportJournalLoadFailure(project.id);
          }
        }
      }
      if (!isActive()) {
        return;
      }
      for (const project of projectsAtStart) {
        loadedProjectIds.add(project.id);
        for (const item of project.queue.items) {
          if (item.status === 'pending' || item.status === 'running' || item.cancellationPending === true) {
            const key = getRunKey(project.id, item.id);
            pendingReconcileRunKeys.add(key);
            if (item.cancellationPending === true) {
              cancelRequestedRunKeys.add(key);
            }
          }
        }
      }
      hasInitialized = true;
    };
    void initializeProjects()
      .catch(() => {
        if (!isActive()) {
          return;
        }
        reportJournalLoadFailure();
        for (const project of projectsAtStart) {
          loadedProjectIds.add(project.id);
          for (const item of project.queue.items) {
            if (item.status === 'pending' || item.status === 'running' || item.cancellationPending === true) {
              const key = getRunKey(project.id, item.id);
              pendingReconcileRunKeys.add(key);
              if (item.cancellationPending === true) {
                cancelRequestedRunKeys.add(key);
              }
            }
          }
        }
        hasInitialized = true;
      })
      .finally(() => {
        isInitializing = false;
        if (isActive()) {
          synchronize();
        }
      });
  };

  const synchronize = (): void => {
    initialize();
    discoverProjects();
    prepareActiveRuns();
    synchronizeRunPersistence();
    reconcile();
    processQueueItems();
  };

  const start = (): void => {
    if (isStarted || !isActive()) {
      return;
    }

    isStarted = true;
    coordinator.connect();
    ensureTemplatesLoaded();
    renewTimer = setInterval(renewAdoptedRerunClaims, RERUN_CLAIM_RENEW_MS);
    detachers.push(
      history.subscribe(synchronize),
      backend.on('queue_items_retried', handleItemsRetried),
      backend.on('queue_item_status_changed', handleUnadoptedRerunSettled),
      backend.onConnectionChange((status, error) => {
        if (!isActive()) {
          return;
        }

        commands.setConnectionStatus({ error, status });
        if (status === 'connected') {
          for (const runKey of reconcileRetryRunKeys) {
            reconcileRetryRunKeys.delete(runKey);
            startedRunKeys.delete(runKey);
            pendingReconcileRunKeys.add(runKey);
          }
          for (const runKey of pendingFallbackAcks.keys()) {
            void flushFallbackAck(runKey);
          }
        }
        synchronize();
      })
    );
    synchronize();
  };

  const dispose = async (): Promise<void> => {
    isDisposed = true;
    rerunWaits.abort();
    if (retryTimer !== undefined) {
      clearTimeout(retryTimer);
      retryTimer = undefined;
    }

    for (const detach of detachers.splice(0)) {
      detach();
    }

    coordinator.dispose();
    pendingResultRoutes.clear();
    await resultRoutingFlush?.catch(() => undefined);
    // A settling flush may start another for reads queued while it drained.
    for (let flush = resultReadFlush; flush; flush = resultReadFlush) {
      await flush.catch(() => undefined);
    }
    await Promise.all(
      [...cancellationOperations.values(), ...submissionOperations.values(), ...lockRequests.values()].map(
        (operation) => operation.catch(() => undefined)
      )
    );
    while (journalRecordFlushes.size > 0 || journalOperations.size > 0 || pendingJournalRecords.size > 0) {
      for (const key of pendingJournalRecords.keys()) {
        void flushJournalRecords(key);
      }
      await Promise.all(
        [...journalRecordFlushes.values(), ...journalOperations.values()].map((operation) =>
          operation.catch(() => undefined)
        )
      );
    }
    if (renewTimer !== undefined) {
      clearInterval(renewTimer);
      renewTimer = undefined;
    }
    // A run restored by the next hold claims again as the same claimant; meanwhile another tab may stage the rerun.
    for (const key of adoptedRerunClaims.keys()) {
      const [projectId, queueItemId] = JSON.parse(key) as [string, string];
      releaseAdoptedReruns(projectId, queueItemId);
    }
    startedRunKeys.clear();
    preparingRunKeys.clear();
    pendingJournalSettles.clear();
    journalSettleFlushes.clear();
    pendingFallbackAcks.clear();
    fallbackAckOperations.clear();
    reconcileRetryRunKeys.clear();
    pendingReconcileRunKeys.clear();
    trackedRuns.clear();
    activeAttempts.clear();
    ownershipEpochs.clear();
    latestJournalWriteTokens.clear();
    for (const [key, lock] of ownedLocks) {
      ownedLocks.delete(key);
      await lock.release().catch(() => undefined);
    }
  };

  return { dispose, start };
};
