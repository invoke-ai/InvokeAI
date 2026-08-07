import type { HydratedWorkbenchSnapshot } from '@workbench/persistenceContracts';
import type { WorkbenchState } from '@workbench/projectContracts';

import type { WorkbenchLoadOptions, WorkbenchSaveResult } from './projects/syncedPersistence';

export interface PersistenceAggregatePort {
  /** Point a project at the board the server minted for it. */
  assignProjectBoard(assignment: WorkbenchSaveResult['projectBoardAssignments'][number]): void;
  getPersistedRevision(): number;
  getState(): WorkbenchState;
  hydrate(state: WorkbenchState): void;
  notifyProjectNotFound(): void;
  reconcileConflict(conflict: WorkbenchSaveResult['conflicts'][number]): void;
  /** Replace a project deleted elsewhere with the fork carrying its unsaved edits. */
  reconcileDeletedProject(fork: WorkbenchSaveResult['deletedProjectForks'][number]): void;
  reportLoadError(error: string): void;
  saveFailed(error: string): void;
  saveStarted(): void;
  saveSucceeded(savedAt: string): void;
  setHasHydrated(hasHydrated: boolean): void;
  subscribe(listener: () => void): () => void;
}

export interface WorkbenchPersistencePort {
  hasPendingChanges(): boolean;
  loadWorkbench(options?: WorkbenchLoadOptions): Promise<HydratedWorkbenchSnapshot | null>;
  saveWorkbench(state: WorkbenchState): Promise<WorkbenchSaveResult>;
}

export interface PersistenceClock {
  clearTimeout(id: unknown): void;
  setTimeout(callback: () => void, delayMs: number): unknown;
}

export interface PersistenceRuntimeSnapshot {
  error: string | null;
  phase: 'disposed' | 'hydrating' | 'idle' | 'saving';
}

export interface WorkbenchPersistenceRuntime {
  dispose(): void;
  getSnapshot(): PersistenceRuntimeSnapshot;
  start(): void;
  subscribe(listener: () => void): () => void;
}

const browserClock: PersistenceClock = {
  clearTimeout: (id) => window.clearTimeout(id as number),
  setTimeout: (callback, delayMs) => window.setTimeout(callback, delayMs),
};

const errorMessage = (error: unknown, fallback: string): string => (error instanceof Error ? error.message : fallback);

export const createWorkbenchPersistenceRuntime = ({
  aggregate,
  clock = browserClock,
  loadOptions,
  persistence,
  saveDelayMs = 500,
  signal,
}: {
  aggregate: PersistenceAggregatePort;
  clock?: PersistenceClock;
  loadOptions?: WorkbenchLoadOptions;
  persistence: WorkbenchPersistencePort;
  saveDelayMs?: number;
  /** Cancels this runtime when the account lifetime that owns it expires. */
  signal?: AbortSignal;
}): WorkbenchPersistenceRuntime => {
  let snapshot: PersistenceRuntimeSnapshot = { error: null, phase: 'idle' };
  const listeners = new Set<() => void>();
  let started = false;
  let disposed = false;
  let hasLoaded = false;
  let generation = 0;
  let timeoutId: unknown | null = null;
  let scheduledRevision: number | null = null;
  let failedRevision: number | null = null;
  let lastSavedRevision = aggregate.getPersistedRevision();
  let previousConnectionStatus = aggregate.getState().backendConnection.status;
  let isSaveInFlight = false;
  let queuedSaveRequireCurrentRevision: boolean | null = null;
  let unsubscribeAggregate: (() => void) | null = null;

  const publish = (next: PersistenceRuntimeSnapshot): void => {
    if (disposed || (snapshot.error === next.error && snapshot.phase === next.phase)) {
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
    for (const conflict of result.conflicts) {
      aggregate.reconcileConflict(conflict);
    }

    for (const fork of result.deletedProjectForks) {
      aggregate.reconcileDeletedProject(fork);
    }

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

    // Staleness is read before anything is applied, because applying is itself an edit: assigning
    // a board dispatches through the reducer and bumps the generation this check compares against.
    const isStale = isStaleSave(revision, saveGeneration, requireCurrentRevision);

    // Applied either way. Every one of these is a fact about the *server* — the board it minted,
    // the id the fork already occupies — rather than a statement about the snapshot that was sent.
    // A save is stale whenever a keystroke lands while it is in flight, which for a project's very
    // first save is the common case; dropping the answer there leaves a new project pointing at no
    // board, and a deleted one recreated under its old id by the next push.
    //
    // What makes that safe for the two that replace a project, and not only for the idempotent
    // board write, is that they no longer carry a document. A fork hands over an *identity*, and
    // the reducer re-labels the live project with it; the content the person can see is never
    // swapped for the snapshot this save started from. See `ProjectRecoveredIdentity`.
    applySaveResult(result);

    if (isStale) {
      return;
    }
    lastSavedRevision = revision;
    failedRevision = null;
    scheduledRevision = null;
    aggregate.saveSucceeded(result.snapshot.savedAt);
    publish({ error: null, phase: 'idle' });
  };

  const failSave = (
    error: unknown,
    revision: number,
    saveGeneration: number,
    requireCurrentRevision: boolean
  ): void => {
    if (isStaleSave(revision, saveGeneration, requireCurrentRevision)) {
      return;
    }
    const message = errorMessage(error, 'Failed to autosave workbench.');
    failedRevision = revision;
    scheduledRevision = null;
    aggregate.saveFailed(message);
    publish({ error: message, phase: 'idle' });
  };

  const save = (requireCurrentRevision: boolean): void => {
    if (disposed || !hasLoaded) {
      return;
    }
    timeoutId = null;

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
    aggregate.saveStarted();
    publish({ error: null, phase: 'saving' });
    void persistence
      .saveWorkbench(state)
      .then((result) => completeSave(result, revision, saveGeneration, requireCurrentRevision))
      .catch((error: unknown) => failSave(error, revision, saveGeneration, requireCurrentRevision))
      .finally(() => {
        isSaveInFlight = false;
        const queuedRequireCurrentRevision = queuedSaveRequireCurrentRevision;
        queuedSaveRequireCurrentRevision = null;

        if (queuedRequireCurrentRevision !== null) {
          save(queuedRequireCurrentRevision);
        }
      });
  };

  const scheduleSave = (): void => {
    if (disposed || !hasLoaded) {
      return;
    }
    const revision = aggregate.getPersistedRevision();
    if (revision === lastSavedRevision || revision === scheduledRevision || revision === failedRevision) {
      return;
    }
    failedRevision = null;
    scheduledRevision = revision;
    generation += 1;
    clearScheduledSave();
    timeoutId = clock.setTimeout(() => save(false), saveDelayMs);
  };

  const onAggregateChange = (): void => {
    if (disposed) {
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

  const load = async (): Promise<void> => {
    const loadGeneration = generation;
    const revisionBeforeLoad = aggregate.getPersistedRevision();
    publish({ error: null, phase: 'hydrating' });
    let loadedSnapshot: HydratedWorkbenchSnapshot | null = null;

    try {
      loadedSnapshot = await persistence.loadWorkbench(loadOptions);
      if (disposed || loadGeneration !== generation) {
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
      const projects = loadedSnapshot?.state.projects ?? aggregate.getState().projects;
      if (requestedId && !projects.some((project) => project.id === requestedId)) {
        aggregate.notifyProjectNotFound();
      }
    } catch (error) {
      if (!disposed) {
        aggregate.reportLoadError(errorMessage(error, 'Failed to load persisted workbench.'));
      }
    } finally {
      if (!disposed && loadGeneration === generation) {
        hasLoaded = true;
        aggregate.setHasHydrated(true);
        publish({ error: null, phase: 'idle' });
        scheduleSave();
      }
    }
  };

  const dispose = (): void => {
    if (disposed) {
      return;
    }
    disposed = true;
    generation += 1;
    queuedSaveRequireCurrentRevision = null;
    clearScheduledSave();
    unsubscribeAggregate?.();
    unsubscribeAggregate = null;
    signal?.removeEventListener('abort', dispose);
    snapshot = { error: null, phase: 'disposed' };
    listeners.clear();
  };

  return {
    dispose,
    getSnapshot: () => snapshot,
    start() {
      if (started || disposed) {
        return;
      }
      if (signal?.aborted) {
        dispose();
        return;
      }
      started = true;
      signal?.addEventListener('abort', dispose, { once: true });
      unsubscribeAggregate = aggregate.subscribe(onAggregateChange);
      void load();
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
};
