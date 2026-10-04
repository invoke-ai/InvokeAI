import type { Logger } from '@platform/logging/contracts';
import type { HydratedWorkbenchSnapshot } from '@workbench/persistenceContracts';
import type { RefusedWorkbenchProject, WorkbenchState } from '@workbench/projectContracts';

import { createLogger } from '@platform/logging/logger';

import type { WorkbenchLoadOptions, WorkbenchSaveResult } from './projects/syncedPersistence';

import { WorkbenchBackendUnavailableError } from './projects/syncedPersistence';

export interface PersistenceAggregatePort {
  /** Point a project at the board the server minted for it. */
  assignProjectBoard(assignment: WorkbenchSaveResult['projectBoardAssignments'][number]): void;
  getPersistedRevision(): number;
  getState(): WorkbenchState;
  hydrate(state: WorkbenchState): void;
  notifyProjectNotFound(): void;
  reportLoadAvailable(): void;
  reportLoadError(error: string): void;
  reportLoadUnavailable(error: string): void;
  /** Persisted projects the canvas version gate refused; they are absent from the hydrated state. */
  reportRefusedProjects(refused: readonly RefusedWorkbenchProject[]): void;
  saveFailed(error: string): void;
  savePending(error: string): void;
  /** Persisted content changed and its save is scheduled: the workbench is dirty until that save is acknowledged. */
  saveScheduled(): void;
  saveStarted(): void;
  saveSucceeded(savedAt: string): void;
  setHasHydrated(hasHydrated: boolean): void;
  subscribe(listener: () => void): () => void;
}

export interface WorkbenchPersistencePort {
  /** True while the editor is leaving after closing its last tab: its session must stay empty, so nothing saves. */
  hasClosedSession(): boolean;
  hasPendingChanges(): boolean;
  loadWorkbench(options?: WorkbenchLoadOptions): Promise<HydratedWorkbenchSnapshot | null>;
  saveWorkbench(state: WorkbenchState): Promise<WorkbenchSaveResult>;
}

export interface PersistenceClock {
  clearTimeout(id: unknown): void;
  setTimeout(callback: () => void, delayMs: number): unknown;
}

/** Signals the moments a page may stop running script without further notice. */
export interface PageLifecyclePort {
  subscribeHidden(listener: () => void): () => void;
}

export interface PersistenceRuntimeSnapshot {
  error: string | null;
  phase: 'disposed' | 'hydrating' | 'idle' | 'saving' | 'unavailable';
}

export type PersistenceExitOutcome = 'abandoned' | 'failed' | 'nothing-to-save' | 'saved';

/** An editor's exit checkpoint, as the next editor of the same account sees it. */
export interface PersistenceExit {
  /** Settles once the checkpoint has finished or been abandoned; never rejects. */
  settled: Promise<unknown>;
  /** The next editor has stopped waiting: start no further write. */
  supersede(): void;
}

export interface WorkbenchPersistenceRuntime {
  /**
   * Ends the editor lifetime. Stops observing the aggregate and saves its latest committed state unless it is already
   * saved. `beforeCapture` brings state still held outside the aggregate in (bounded); if that moves the state on, it
   * is saved once more.
   */
  exit(options?: { beforeCapture?: () => Promise<unknown> }): PersistenceExit & {
    settled: Promise<PersistenceExitOutcome>;
  };
  getSnapshot(): PersistenceRuntimeSnapshot;
  retryLoad(): void;
  start(): void;
  subscribe(listener: () => void): () => void;
}

/**
 * How long a remounted editor waits for the previous editor's exit checkpoint before loading anyway. A checkpoint
 * normally settles in one save round trip; this only bounds a hung request.
 */
const EXIT_CHECKPOINT_WAIT_MS = 10_000;
/** Well under the re-entry wait, so a slow pixel upload cannot hold the checkpoint's second save past it. */
const EXIT_CAPTURE_WAIT_MS = 3_000;

const browserClock: PersistenceClock = {
  clearTimeout: (id) => window.clearTimeout(id as number),
  setTimeout: (callback, delayMs) => window.setTimeout(callback, delayMs),
};

/** `pagehide` and becoming hidden; hidden is often the last event a backgrounded or discarded page receives. */
export const browserPageLifecycle: PageLifecyclePort = {
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
  loadOptions?: WorkbenchLoadOptions;
  logger?: Logger;
  /** Saves the current revision immediately when the page is hidden or unloaded. */
  page?: PageLifecyclePort;
  persistence: WorkbenchPersistencePort;
  /** The previous editor's exit checkpoint for this account; loading waits for it, for at most 10 seconds. */
  previousExit?: PersistenceExit;
  saveDelayMs?: number;
  /** Cancels this runtime when the account lifetime that owns it expires. */
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
  /** The revision the exit checkpoint last tried to write. */
  let lastExitRevision: number | null = null;
  let retryAttempt = 0;
  let queuedSaveRequireCurrentRevision: boolean | null = null;
  let pendingPreviousExit = previousExit ?? null;
  let previousExitWait: Promise<void> | null = null;
  let unsubscribeAggregate: (() => void) | null = null;
  let unsubscribePage: (() => void) | null = null;

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
      // The exit checkpoint only needs to know whether this save covered the state it is about to capture.
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
    if (result.shouldRetry) {
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
    if (result.hasPendingChanges) {
      aggregate.savePending('Autosave requires your attention.');
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
    // The closing tab flow either leaves the editor or reopens the session, which saves it then.
    if (persistence.hasClosedSession()) {
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

  /** Best effort only: a hidden page may be frozen or discarded before the request completes. */
  const saveBeforeHidden = (): void => {
    const revision = aggregate.getPersistedRevision();
    // A failed revision waits for a new edit as it does under the debounce; hiding and unloading both fire.
    if (
      isStopped() ||
      !hasLoaded ||
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

      // A persisted edit made while loading is newer than the loaded snapshot.
      // Preserve it and let the first autosave reconcile it with remote storage.
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
      // A deep-linked refusal is reported by the session controller when it retries the open.
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

  const stopObserving = (): void => {
    clearScheduledSave();
    queuedSaveRequireCurrentRevision = null;
    unsubscribeAggregate?.();
    unsubscribeAggregate = null;
    unsubscribePage?.();
    unsubscribePage = null;
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

  /** Saves the latest committed state unless it is saved already or was the revision `skipRevision` just tried. */
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
    // Stage what is already committed at once (behind any save in flight) instead of after the capture, which can
    // take network round trips; the descendants' unmount flushes land before this first await resumes.
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
    let isSuperseded = false;
    const settled = checkpoint(beforeCapture, () => isSuperseded).finally(dispose);
    exitHandle = {
      settled,
      supersede: () => {
        isSuperseded = true;
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
      void load();
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
};
