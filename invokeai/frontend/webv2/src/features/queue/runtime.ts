import type { QueueItem } from '@features/queue/core/historyTypes';
import type {
  QueueBackendPort,
  QueueEnqueueGenerateRequest,
  QueueEnqueueWorkflowRequest,
  QueueResultImage,
  QueueResultImageOptions,
  QueueWorkflowRunSink,
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
import { shouldSubmitPendingQueueItem } from '@features/queue/core/submissionRules';
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
import { mapWithConcurrency } from '@platform/core/concurrency';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
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
  restoreFromJournal(payload: { items: unknown[]; projectId: string }): void;
  routePartialResults(payload: {
    backendItemId: number;
    images: QueueResultImage[];
    projectId: string;
    queueItemId: string;
  }): void;
  routeResults(payload: { images: QueueResultImage[]; projectId: string; queueItemId: string }): void;
  setConnectionStatus(payload: { error?: string; status: BackendConnectionStatus }): void;
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

/** Read compiled graphs to identify input passthroughs; legacy or invalid snapshots may lack them. */
const getQueueItemCompiledGraph = (queueItem: QueueItem): unknown => {
  const submission = (queueItem.snapshot as Partial<QueueItem['snapshot']>).backendSubmission;
  return submission && typeof submission === 'object' && 'graph' in submission ? submission.graph : undefined;
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

  // `libraryWorkflowId` is provenance for the completed-run sink, not something
  // the backend enqueue accepts; it stays on the snapshot and off the request.
  const { kind: _, libraryWorkflowId: _libraryWorkflowId, ...compiled } = submission;
  return {
    kind: 'workflow',
    request: {
      ...compiled,
      destination: queueItem.snapshot.destination,
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
  workflowRuns,
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
  workflowRuns?: QueueWorkflowRunSink;
}): QueueRuntime => {
  const owner = captureAccountScope();
  const commands = history.commands;
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
  const ownedLocks = new Map<string, Extract<QueueRunLock, { kind: 'acquired' }>>();
  const lockRequests = new Map<string, Promise<boolean>>();
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
    void lock.release().catch(() => undefined);
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
    startedRunKeys.delete(key);
    pendingReconcileRunKeys.delete(key);
    reconcileRetryRunKeys.delete(key);
    cancelRequestedRunKeys.delete(key);
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
          commands.recordError({
            area: 'queue-cancel',
            message: toErrorMessage(error),
            namespace: 'queue',
            projectId,
          });
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

  // An image a node saved to its own board stays there; only unassigned images land on the active board.
  const addImagesToDestination = async (queueItem: QueueItem, images: QueueResultImage[]): Promise<void> => {
    if (!isActive() || queueItem.snapshot.destination !== 'gallery') {
      return;
    }

    const boardId = queueItem.snapshot.galleryBoardId;
    const imageNames = images.filter((image) => !image.boardId).map((image) => image.imageName);

    if (boardId && boardId !== 'none' && imageNames.length > 0) {
      await destinations.addImagesToGalleryBoard(boardId, imageNames);
    }
  };

  /**
   * Attach generated videos to the destination board; repeat attachment is idempotent. Categorization failures are
   * recorded but never turn successful generation into failure.
   */
  const addResultVideosToDestination = async (
    projectId: string,
    queueItem: QueueItem,
    backendItemIds: number[]
  ): Promise<void> => {
    // Skip known cancelled backend items. Resumed cancellation records can be incomplete, so remaining attachments
    // are best-effort.
    const deliverableItemIds = backendItemIds.filter(
      (backendItemId) => !queueItem.cancelledBackendItemIds?.includes(backendItemId)
    );

    if (!isActive() || queueItem.snapshot.destination !== 'gallery' || deliverableItemIds.length === 0) {
      return;
    }

    const boardId = queueItem.snapshot.galleryBoardId;

    if (!boardId || boardId === 'none') {
      return;
    }

    try {
      const imageOptions = getQueueItemResultImageOptions(queueItem);
      const options = queueItem.snapshot.filterIntermediateResults
        ? { ...imageOptions, excludeIntermediate: true }
        : imageOptions;
      const namesPerItem = await mapWithConcurrency(deliverableItemIds, QUEUE_RUNTIME_CONCURRENCY, (backendItemId) =>
        runResultRead(() => backend.getResultVideoNames(backendItemId, options))
      );
      // Exclude video primitives' echoed input clips from generated results.
      const inputMedia = collectGraphInputMediaNames(getQueueItemCompiledGraph(queueItem));
      const videoNames = [...new Set(namesPerItem.flat())].filter((name) => !inputMedia.videoNames.has(name));

      if (videoNames.length === 0 || !isActive()) {
        return;
      }

      await destinations.addVideosToGalleryBoard(boardId, videoNames);
    } catch (error) {
      if (isActive()) {
        commands.recordError({
          area: 'queue-results',
          message: toErrorMessage(error),
          namespace: 'queue',
          projectId,
        });
      }
    }
  };

  /**
   * Remove input passthroughs and requested intermediates before routing; echoed source media must not move boards
   * on every run.
   */
  const deliverVisibleImages = async (
    queueItem: QueueItem,
    allImages: QueueResultImage[]
  ): Promise<QueueResultImage[]> => {
    const inputMedia = collectGraphInputMediaNames(getQueueItemCompiledGraph(queueItem));
    const producedImages = allImages.filter((image) => !inputMedia.imageNames.has(image.imageName));
    const images = queueItem.snapshot.filterIntermediateResults
      ? producedImages.filter((image) => !image.isIntermediate)
      : producedImages;

    await addImagesToDestination(queueItem, images);

    return images;
  };

  /**
   * Notify the optional library sink only for bound workflows. Capture is decoration: sink failure must not fail
   * settlement or prevent gallery refresh.
   */
  const notifyWorkflowRunCompleted = (projectId: string, queueItem: QueueItem, images: QueueResultImage[]): void => {
    const submission = (queueItem.snapshot as Partial<QueueItem['snapshot']>).backendSubmission;

    if (
      !workflowRuns ||
      submission?.kind !== 'workflow' ||
      typeof submission.libraryWorkflowId !== 'string' ||
      images.length === 0
    ) {
      return;
    }

    try {
      workflowRuns.onWorkflowRunCompleted({
        imageNames: images.map((image) => image.imageName),
        libraryWorkflowId: submission.libraryWorkflowId,
        projectId,
        queueItemId: queueItem.id,
      });
    } catch {
      // The sink owns its own error reporting; the run is already complete.
    }
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
      await addResultVideosToDestination(projectId, queueItem, backendItemIds);

      if (!isAttemptCurrent(attempt)) {
        return;
      }

      commands.routeResults({ images, projectId, queueItemId: queueItem.id });
      notifyWorkflowRunCompleted(projectId, queueItem, images);
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
      const images = await runResultRead(() =>
        backend.getResultImages(
          backendItemId,
          queueItem.id,
          queueItem.snapshot.submittedAt,
          getQueueItemResultImageOptions(queueItem)
        )
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

      const visibleImages = await deliverVisibleImages(queueItem, images);
      // Never throws — a board-attach hiccup must not block routePartialResults below.
      await addResultVideosToDestination(projectId, queueItem, [backendItemId]);

      if (!isAttemptCurrent(attempt)) {
        return;
      }

      // The held denoise frame may be painted over exactly these images while
      // they decode — not over an earlier image of the same batch.
      progressImageStore.bindSwapImages(
        queueItem.id,
        visibleImages.map((image) => image.imageName)
      );
      commands.routePartialResults({
        backendItemId,
        images: visibleImages,
        projectId,
        queueItemId: queueItem.id,
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
    getResultImages: (...args) => runResultRead(() => backend.getResultImages(...args)),
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
        : coordinator.submitWorkflow(currentQueueItem.id, submission.request);
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
          trackedRuns.delete(key);
          startedRunKeys.delete(key);
          pendingReconcileRunKeys.delete(key);
          reconcileRetryRunKeys.delete(key);
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
          id: queueItem.id,
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
    detachers.push(
      history.subscribe(synchronize),
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
    await resultReadFlush?.catch(() => undefined);
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
