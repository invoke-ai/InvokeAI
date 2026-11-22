import type { Logger } from '@platform/logging/contracts';
import type { HydratedWorkbenchSnapshot } from '@workbench/persistenceContracts';
import type { RefusedWorkbenchProject, WorkbenchState } from '@workbench/projectContracts';

import { createLogger } from '@platform/logging/logger';

import type { UnloadJournalOutcome, WorkbenchLoadOptions, WorkbenchSaveResult } from './projects/syncedPersistence';

import { WorkbenchBackendUnavailableError } from './projects/syncedPersistence';

export interface PersistenceAggregatePort {
  assignProjectBoard(assignment: WorkbenchSaveResult['projectBoardAssignments'][number]): void;
  releaseHeldLayouts(): void;
  getPersistedRevision(): number;
  getState(): WorkbenchState;
  hydrate(state: WorkbenchState): void;
  notifyProjectNotFound(): void;
  reportLoadAvailable(): void;
  reportLoadError(error: string): void;
  reportLoadUnavailable(error: string): void;
  reportRefusedProjects(refused: readonly RefusedWorkbenchProject[]): void;
  saveFailed(error: string): void;
  saveNeedsAttention(savedAt: string): void;
  savePending(error: string): void;
  saveScheduled(): void;
  saveStarted(): void;
  saveSucceeded(savedAt: string): void;
  setHasHydrated(hasHydrated: boolean): void;
  subscribe(listener: () => void): () => void;
}

export interface WorkbenchPersistencePort {
  hasClosedSession(): boolean;
  hasPendingChanges(): boolean;
  subscribeSessionReopened(listener: () => void): () => void;
  journalBeforeUnload(state: WorkbenchState): UnloadJournalOutcome;
  hasChangesUnloadWouldLose(): boolean;
  loadWorkbench(options?: WorkbenchLoadOptions): Promise<HydratedWorkbenchSnapshot | null>;
  saveWorkbench(state: WorkbenchState): Promise<WorkbenchSaveResult>;
}

export interface PersistenceClock {
  clearTimeout(id: unknown): void;
  setTimeout(callback: () => void, delayMs: number): unknown;
}

export interface PageLifecyclePort {
  /** Checkpoints before unload; a true answer has the browser offer to stay, with its own wording. */
  subscribeBeforeUnload(shouldPrompt: () => boolean): () => void;
  /** Checkpoints on hidden/pagehide; the callback must tolerate repeated signals. */
  subscribeHidden(listener: () => void): () => void;
}

export interface PersistenceRuntimeSnapshot {
  error: string | null;
  phase: 'disposed' | 'hydrating' | 'idle' | 'saving' | 'unavailable';
}

export type PersistenceExitOutcome = 'abandoned' | 'failed' | 'nothing-to-save' | 'saved';

export interface PersistenceExit {
  settled: Promise<unknown>;
  supersede(): void;
}

export interface WorkbenchPersistenceRuntime {
  /** `beforeCapture` brings in state held outside the aggregate (bounded); if it moves the state on, it saves again. */
  exit(options?: { beforeCapture?: () => Promise<unknown> }): PersistenceExit & {
    settled: Promise<PersistenceExitOutcome>;
  };
  getSnapshot(): PersistenceRuntimeSnapshot;
  retryLoad(): void;
  start(): void;
  subscribe(listener: () => void): () => void;
}

/** Bounds only a hung request; a checkpoint normally settles in one save round trip. */
const EXIT_CHECKPOINT_WAIT_MS = 10_000;
/** Well under the re-entry wait, so a slow pixel upload cannot hold the checkpoint's second save past it. */
const EXIT_CAPTURE_WAIT_MS = 3_000;

const browserClock: PersistenceClock = {
  clearTimeout: (id) => window.clearTimeout(id as number),
  setTimeout: (callback, delayMs) => window.setTimeout(callback, delayMs),
};

/** Checkpoint before WebKit starts tearing IndexedDB down; hidden still covers pages that are later discarded. */
export const browserPageLifecycle: PageLifecyclePort = {
  subscribeBeforeUnload: (shouldPrompt) => {
    const listener = (event: BeforeUnloadEvent): void => {
      if (shouldPrompt()) {
        event.preventDefault();
        // Browsers that predate `preventDefault` for this event prompt on any assigned value; the text is their own.
        event.returnValue = '';
      }
    };
    window.addEventListener('beforeunload', listener);
    return () => window.removeEventListener('beforeunload', listener);
  },
  subscribeHidden: (listener) => {
    const onVisibilityChange = (): void => {
      if (document.visibilityState === 'hidden') {
        listener();
      }
    };
    window.addEventListener('pagehide', listener);
    document.addEventListener('visibilitychange', onVisibilityChange);
    return () => {
      window.removeEventListener('pagehide', listener);
      document.removeEventListener('visibilitychange', onVisibilityChange);
    };
  },
};

const defaultLogger = createLogger({ area: 'autosave', namespace: 'persistence' });

const errorMessage = (error: unknown, fallback: string): string => (error instanceof Error ? error.message : fallback);

export const createWorkbenchPersistenceRuntime = ({
  aggregate,
  clock = browserClock,
  commitDrafts,
  loadOptions,
  logger = defaultLogger,
  page,
  persistence,
  previousExit,
  saveDelayMs = 500,
  signal,
}: {
  aggregate: PersistenceAggregatePort;
  clock?: PersistenceClock;
  commitDrafts?: () => void;
  loadOptions?: WorkbenchLoadOptions;
  logger?: Logger;
  page?: PageLifecyclePort;
  persistence: WorkbenchPersistencePort;
  previousExit?: PersistenceExit;
  saveDelayMs?: number;
  signal?: AbortSignal;
}): WorkbenchPersistenceRuntime => {
  let snapshot: PersistenceRuntimeSnapshot = { error: null, phase: 'idle' };
  const listeners = new Set<() => void>();
  let started = false;
  let disposed = false;
  let exiting = false;
  let exitHandle: ReturnType<WorkbenchPersistenceRuntime['exit']> | null = null;
  let hasLoaded = false;
  let generation = 0;
  let timeoutId: unknown | null = null;
  let scheduledRevision: number | null = null;
  let failedRevision: number | null = null;
  let lastSavedRevision = aggregate.getPersistedRevision();
  let previousConnectionStatus = aggregate.getState().backendConnection.status;
  let isSaveInFlight = false;
  let inFlightRevision: number | null = null;
  let inFlightSave: Promise<void> | null = null;
  let lastExitRevision: number | null = null;
  /** The newest revision journaled; `pagehide` and becoming hidden both fire on unload. */
  let journaledRevision: number | null = null;
  let retryAttempt = 0;
  let queuedSaveRequireCurrentRevision: boolean | null = null;
  let saveHeldByClosedSession: boolean | null = null;
  let pendingPreviousExit = previousExit ?? null;
  let previousExitWait: Promise<void> | null = null;
  let unsubscribeAggregate: (() => void) | null = null;
  let unsubscribePage: (() => void) | null = null;
  let unsubscribeBeforeUnload: (() => void) | null = null;
  let hasJournalFailed = false;
  let unsubscribeSessionReopened: (() => void) | null = null;

  const isStopped = (): boolean => disposed || exiting;

  const publish = (next: PersistenceRuntimeSnapshot): void => {
    if (isStopped() || (snapshot.error === next.error && snapshot.phase === next.phase)) {
      return;
    }
    snapshot = next;
    for (const listener of listeners) {
      listener();
    }
  };

  const clearScheduledSave = (): void => {
    if (timeoutId !== null) {
      clock.clearTimeout(timeoutId);
      timeoutId = null;
    }
  };

  const applySaveResult = (result: WorkbenchSaveResult): void => {
    for (const assignment of result.projectBoardAssignments) {
      aggregate.assignProjectBoard(assignment);
    }
    // Every state after hydration carries the account that adopted, so any acknowledged session write saved it.
    if (result.accountSaved) {
      aggregate.releaseHeldLayouts();
    }
  };

  const isStaleSave = (revision: number, saveGeneration: number, requireCurrentRevision: boolean): boolean =>
    disposed ||
    saveGeneration !== generation ||
    (requireCurrentRevision && aggregate.getPersistedRevision() !== revision);

  const completeSave = (
    result: WorkbenchSaveResult,
    revision: number,
    saveGeneration: number,
    requireCurrentRevision: boolean
  ): void => {
    if (disposed) {
      return;
    }

    // Check staleness before applying board assignment, which itself advances the compared generation.
    const isStale = isStaleSave(revision, saveGeneration, requireCurrentRevision);

    // Board identity is a server fact and remains safe to apply when this save is stale.
    applySaveResult(result);

    if (isStale) {
      return;
    }
    if (exiting) {
      if (!result.error && !result.shouldRetry) {
        lastSavedRevision = revision;
      }
      return;
    }
    const scheduleRetry = (): void => {
      scheduledRevision = revision;
      failedRevision = null;
      retryAttempt += 1;
      clearScheduledSave();
      timeoutId = clock.setTimeout(() => save(true), Math.min(saveDelayMs * 2 ** retryAttempt, 30_000));
    };
    if (result.error) {
      aggregate.saveFailed(result.error);
      publish({ error: result.error, phase: 'idle' });
      if (result.shouldRetry) {
        scheduleRetry();
      } else {
        failedRevision = revision;
        scheduledRevision = null;
      }
      return;
    }
    if (result.shouldRetry || result.hasRetryablePendingChanges) {
      const message = 'Autosave is pending and will retry.';
      aggregate.savePending(message);
      publish({ error: message, phase: 'idle' });
      scheduleRetry();
      return;
    }
    lastSavedRevision = revision;
    failedRevision = null;
    scheduledRevision = null;
    retryAttempt = 0;
    if (result.awaitsDecision) {
      aggregate.saveNeedsAttention(result.snapshot.savedAt);
    } else {
      aggregate.saveSucceeded(result.snapshot.savedAt);
    }
    publish({ error: null, phase: 'idle' });
  };

  const failSave = (
    error: unknown,
    revision: number,
    saveGeneration: number,
    requireCurrentRevision: boolean
  ): void => {
    if (exiting || isStaleSave(revision, saveGeneration, requireCurrentRevision)) {
      return;
    }
    const message = errorMessage(error, 'Failed to autosave workbench.');
    failedRevision = revision;
    scheduledRevision = null;
    aggregate.saveFailed(message);
    publish({ error: message, phase: 'idle' });
  };

  const save = (requireCurrentRevision: boolean): void => {
    if (isStopped() || !hasLoaded) {
      return;
    }
    timeoutId = null;
    if (persistence.hasClosedSession()) {
      saveHeldByClosedSession = Boolean(saveHeldByClosedSession) || requireCurrentRevision;
      return;
    }

    if (isSaveInFlight) {
      queuedSaveRequireCurrentRevision =
        queuedSaveRequireCurrentRevision === null
          ? requireCurrentRevision
          : queuedSaveRequireCurrentRevision || requireCurrentRevision;
      return;
    }

    const state = aggregate.getState();
    const revision = aggregate.getPersistedRevision();
    generation += 1;
    const saveGeneration = generation;

    isSaveInFlight = true;
    inFlightRevision = revision;
    aggregate.saveStarted();
    publish({ error: null, phase: 'saving' });
    inFlightSave = persistence
      .saveWorkbench(state)
      .then((result) => completeSave(result, revision, saveGeneration, requireCurrentRevision))
      .catch((error: unknown) => failSave(error, revision, saveGeneration, requireCurrentRevision))
      .finally(() => {
        isSaveInFlight = false;
        inFlightRevision = null;
        inFlightSave = null;
        const queuedRequireCurrentRevision = queuedSaveRequireCurrentRevision;
        queuedSaveRequireCurrentRevision = null;

        if (queuedRequireCurrentRevision !== null) {
          save(queuedRequireCurrentRevision);
        }
      });
  };

  const scheduleSave = (): void => {
    if (isStopped() || !hasLoaded) {
      return;
    }
    const revision = aggregate.getPersistedRevision();
    if (revision === lastSavedRevision || revision === scheduledRevision || revision === failedRevision) {
      return;
    }
    failedRevision = null;
    retryAttempt = 0;
    scheduledRevision = revision;
    generation += 1;
    clearScheduledSave();
    timeoutId = clock.setTimeout(() => save(false), saveDelayMs);
    // Reported outside the aggregate's own notification, as a status change is itself an aggregate change.
    queueMicrotask(() => {
      if (!isStopped() && scheduledRevision === revision && !isSaveInFlight) {
        aggregate.saveScheduled();
      }
    });
  };

  /** A draft that cannot be committed must not keep the already committed state from being saved. */
  const commitPendingDrafts = (): void => {
    try {
      commitDrafts?.();
    } catch (error) {
      logger.warn({
        error,
        message: 'Pending editor drafts could not be committed before saving.',
        name: 'persistence.draft-commit-failed',
      });
    }
  };

  /** The only write that can land while the page unloads; reconciled into local recovery on the next load. */
  const journalBeforeUnload = (): void => {
    const revision = aggregate.getPersistedRevision();
    if (
      disposed ||
      !hasLoaded ||
      persistence.hasClosedSession() ||
      revision === lastSavedRevision ||
      revision === journaledRevision
    ) {
      return;
    }
    try {
      const outcome = persistence.journalBeforeUnload(aggregate.getState());
      if (outcome.kind === 'unavailable') {
        logger.warn({
          message: 'Unsaved changes could not be journaled for recovery before the page was hidden.',
          name: 'persistence.unload-journal-unavailable',
        });
        return;
      }
      journaledRevision = revision;
      if (outcome.kind === 'journaled') {
        void outcome.written.then((isWritten) => {
          if (isWritten) {
            return;
          }
          if (journaledRevision === revision) {
            journaledRevision = null;
          }
          logger.warn({
            message: 'Unsaved changes journaled for recovery before the page was hidden were not written.',
            name: 'persistence.unload-journal-aborted',
          });
        });
      }
      if (outcome.kind === 'journaled' && outcome.skippedOversizedProjectIds.length > 0) {
        logger.warn({
          context: { projectIds: outcome.skippedOversizedProjectIds },
          message: 'Projects too large to journal before the page was hidden rely on autosave alone.',
          name: 'persistence.unload-journal-oversized',
        });
      }
      hasJournalFailed = false;
    } catch (error) {
      logger.warn({
        error,
        message: 'Unsaved changes could not be journaled for recovery before the page was hidden.',
        name: 'persistence.unload-journal-unavailable',
      });
      hasJournalFailed = true;
    }
  };

  const wouldUnloadLoseChanges = (): boolean => {
    if (disposed || !hasLoaded || persistence.hasClosedSession()) {
      return false;
    }
    return (
      aggregate.getPersistedRevision() !== lastSavedRevision &&
      (hasJournalFailed || persistence.hasChangesUnloadWouldLose())
    );
  };

  const promptBeforeUnload = (): boolean => {
    if (isStopped() || !hasLoaded || persistence.hasClosedSession()) {
      return false;
    }
    saveBeforeHidden();
    return wouldUnloadLoseChanges();
  };

  /** Best effort only: a hidden page may be frozen or discarded before the request completes. */
  const saveBeforeHidden = (): void => {
    // A closed session must not change until the editor leaves or reopens it, so its drafts stay where they are.
    if (isStopped() || !hasLoaded || persistence.hasClosedSession()) {
      return;
    }
    commitPendingDrafts();
    journalBeforeUnload();
    const revision = aggregate.getPersistedRevision();
    if (
      revision === lastSavedRevision ||
      revision === failedRevision ||
      (isSaveInFlight && revision === inFlightRevision)
    ) {
      return;
    }
    clearScheduledSave();
    scheduledRevision = revision;
    save(false);
  };

  const onAggregateChange = (): void => {
    if (isStopped()) {
      return;
    }
    const connectionStatus = aggregate.getState().backendConnection.status;
    if (connectionStatus !== previousConnectionStatus) {
      previousConnectionStatus = connectionStatus;
      if (connectionStatus === 'connected' && hasLoaded && persistence.hasPendingChanges()) {
        clearScheduledSave();
        scheduledRevision = aggregate.getPersistedRevision();
        save(true);
        return;
      }
    }
    scheduleSave();
  };

  const waitForPreviousExit = (exit: PersistenceExit): Promise<void> =>
    new Promise((resolve) => {
      const waitTimeoutId = clock.setTimeout(() => {
        exit.supersede();
        resolve();
      }, EXIT_CHECKPOINT_WAIT_MS);
      void exit.settled.then(() => {
        clock.clearTimeout(waitTimeoutId);
        resolve();
      });
    });

  const load = async (): Promise<void> => {
    const loadGeneration = generation;
    const revisionBeforeLoad = aggregate.getPersistedRevision();
    publish({ error: null, phase: 'hydrating' });
    let loadedSnapshot: HydratedWorkbenchSnapshot | null = null;

    try {
      if (pendingPreviousExit) {
        previousExitWait = waitForPreviousExit(pendingPreviousExit);
        pendingPreviousExit = null;
        await previousExitWait;
        if (isStopped() || loadGeneration !== generation) {
          return;
        }
      }
      loadedSnapshot = await persistence.loadWorkbench(loadOptions);
      if (isStopped() || loadGeneration !== generation) {
        return;
      }

      // A persisted edit made while loading is newer than the loaded snapshot; keep it for the first autosave.
      const wasEditedDuringLoad = aggregate.getPersistedRevision() !== revisionBeforeLoad;
      if (loadedSnapshot && !wasEditedDuringLoad) {
        const isPendingSnapshot = persistence.hasPendingChanges();
        aggregate.hydrate(loadedSnapshot.state);
        if (!isPendingSnapshot) {
          lastSavedRevision = aggregate.getPersistedRevision();
        }
      }

      const requestedId = loadOptions?.openProjectId;
      const refusedProjects = loadedSnapshot?.refusedProjects ?? [];
      const unrequestedRefusals = refusedProjects.filter((refused) => refused.projectId !== requestedId);
      if (unrequestedRefusals.length > 0) {
        aggregate.reportRefusedProjects(unrequestedRefusals);
      }

      const projects = loadedSnapshot?.state.projects ?? aggregate.getState().projects;
      if (
        requestedId &&
        !projects.some((project) => project.id === requestedId) &&
        !refusedProjects.some((refused) => refused.projectId === requestedId)
      ) {
        aggregate.notifyProjectNotFound();
      }
    } catch (error) {
      if (isStopped() || loadGeneration !== generation) {
        return;
      }
      const message = errorMessage(error, 'Failed to load persisted workbench.');
      if (error instanceof WorkbenchBackendUnavailableError) {
        aggregate.reportLoadUnavailable(message);
        publish({ error: message, phase: 'unavailable' });
        return;
      }
      aggregate.reportLoadError(message);
    }
    if (!isStopped() && loadGeneration === generation) {
      hasLoaded = true;
      aggregate.reportLoadAvailable();
      aggregate.setHasHydrated(true);
      publish({ error: null, phase: 'idle' });
      scheduleSave();
    }
  };

  const onSessionReopened = (): void => {
    const held = saveHeldByClosedSession;
    saveHeldByClosedSession = null;
    if (held !== null && !isStopped()) {
      save(held);
    }
  };

  const stopObserving = (): void => {
    clearScheduledSave();
    queuedSaveRequireCurrentRevision = null;
    saveHeldByClosedSession = null;
    unsubscribeAggregate?.();
    unsubscribeAggregate = null;
    unsubscribePage?.();
    unsubscribePage = null;
    unsubscribeBeforeUnload?.();
    unsubscribeBeforeUnload = null;
    unsubscribeSessionReopened?.();
    unsubscribeSessionReopened = null;
    snapshot = { error: null, phase: 'disposed' };
    listeners.clear();
  };

  const dispose = (): void => {
    if (disposed) {
      return;
    }
    disposed = true;
    generation += 1;
    stopObserving();
    signal?.removeEventListener('abort', dispose);
  };

  const writeLatest = async (
    isSuperseded: () => boolean,
    skipRevision: number | null
  ): Promise<PersistenceExitOutcome> => {
    await inFlightSave;
    const revision = aggregate.getPersistedRevision();
    if (revision === lastSavedRevision || revision === skipRevision || persistence.hasClosedSession()) {
      return 'nothing-to-save';
    }
    // An account change ends the lifetime this data belongs to; nothing of it is written afterwards.
    if (disposed) {
      return 'abandoned';
    }
    if (isSuperseded()) {
      logger.warn({
        context: { revision },
        message: 'A newer editor took over before the exit checkpoint saved; its unsaved changes were not written.',
        name: 'persistence.exit-checkpoint-abandoned',
      });
      return 'abandoned';
    }
    lastExitRevision = revision;
    try {
      const result = await persistence.saveWorkbench(aggregate.getState());
      if (result.error || result.shouldRetry) {
        logger.warn({
          context: {
            localDraftStatus: result.localDraftStatus,
            reason: result.error ?? 'The backend did not acknowledge every change.',
          },
          message: 'The exit checkpoint did not reach the server; unacknowledged changes stay with local recovery.',
          name: 'persistence.exit-checkpoint-incomplete',
        });
        return 'failed';
      }
      lastSavedRevision = revision;
      return 'saved';
    } catch (error) {
      if (disposed) {
        return 'abandoned';
      }
      logger.error({ error, message: 'The exit checkpoint failed.', name: 'persistence.exit-checkpoint-failed' });
      return 'failed';
    }
  };

  const captureBeforeExit = async (beforeCapture: () => Promise<unknown>): Promise<void> => {
    let captureTimeoutId: unknown = null;
    const timedOut = new Promise<'timed-out'>((resolve) => {
      captureTimeoutId = clock.setTimeout(() => resolve('timed-out'), EXIT_CAPTURE_WAIT_MS);
    });
    try {
      // A synchronous throw becomes a rejection here rather than skipping the checkpoint.
      const capture = new Promise((resolve) => {
        resolve(beforeCapture());
      }).then(() => 'captured' as const);
      if ((await Promise.race([capture, timedOut])) === 'timed-out' && !disposed) {
        logger.warn({
          message: 'State held outside the project (such as Canvas pixels) was not captured before leaving.',
          name: 'persistence.exit-capture-incomplete',
        });
      }
    } catch (error) {
      if (!disposed) {
        logger.warn({
          error,
          message: 'State held outside the project (such as Canvas pixels) could not be captured before leaving.',
          name: 'persistence.exit-capture-incomplete',
        });
      }
    } finally {
      clock.clearTimeout(captureTimeoutId);
    }
  };

  const checkpoint = async (
    beforeCapture: (() => Promise<unknown>) | undefined,
    isSuperseded: () => boolean
  ): Promise<PersistenceExitOutcome> => {
    if (!hasLoaded) {
      // Leaving while still waiting must not let the next editor overtake the checkpoint this one waited for.
      await previousExitWait;
      return 'nothing-to-save';
    }
    // Stage committed state now rather than after the capture, which can take network round trips.
    const [first] = await Promise.all([
      writeLatest(isSuperseded, null),
      beforeCapture ? captureBeforeExit(beforeCapture) : undefined,
    ]);
    if (first === 'abandoned') {
      return first;
    }
    const second = await writeLatest(isSuperseded, lastExitRevision);
    return second === 'nothing-to-save' ? first : second;
  };

  const exit: WorkbenchPersistenceRuntime['exit'] = ({ beforeCapture } = {}) => {
    if (exitHandle) {
      return exitHandle;
    }
    if (disposed) {
      exitHandle = { settled: Promise.resolve('abandoned'), supersede: () => undefined };
      return exitHandle;
    }
    exiting = true;
    stopObserving();
    commitPendingDrafts();
    let isSuperseded = false;
    const unsubscribeExitPage =
      page?.subscribeHidden(() => {
        if (!isSuperseded) {
          journalBeforeUnload();
        }
      }) ?? null;
    unsubscribeBeforeUnload =
      page?.subscribeBeforeUnload(() => {
        if (isSuperseded) {
          return false;
        }
        journalBeforeUnload();
        return wouldUnloadLoseChanges();
      }) ?? null;
    const settled = checkpoint(beforeCapture, () => isSuperseded).finally(() => {
      unsubscribeExitPage?.();
      dispose();
    });
    exitHandle = {
      settled,
      supersede: () => {
        isSuperseded = true;
        unsubscribeBeforeUnload?.();
        unsubscribeBeforeUnload = null;
      },
    };
    return exitHandle;
  };

  return {
    exit,
    getSnapshot: () => snapshot,
    retryLoad() {
      if (isStopped() || hasLoaded || snapshot.phase === 'hydrating') {
        return;
      }
      generation += 1;
      void load();
    },
    start() {
      if (started || isStopped()) {
        return;
      }
      if (signal?.aborted) {
        dispose();
        return;
      }
      started = true;
      signal?.addEventListener('abort', dispose, { once: true });
      unsubscribeAggregate = aggregate.subscribe(onAggregateChange);
      unsubscribePage = page?.subscribeHidden(saveBeforeHidden) ?? null;
      unsubscribeBeforeUnload = page?.subscribeBeforeUnload(promptBeforeUnload) ?? null;
      unsubscribeSessionReopened = persistence.subscribeSessionReopened(onSessionReopened);
      void load();
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
};
