import type { HydratedWorkbenchSnapshot } from '@workbench/persistenceContracts';
import type { RefusedWorkbenchProject, WorkbenchState } from '@workbench/projectContracts';

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
  phase: 'disposed' | 'hydrating' | 'idle' | 'saving' | 'unavailable';
}

export interface WorkbenchPersistenceRuntime {
  dispose(): void;
  getSnapshot(): PersistenceRuntimeSnapshot;
  retryLoad(): void;
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
  let retryAttempt = 0;
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
    retryAttempt = 0;
    scheduledRevision = revision;
    generation += 1;
    clearScheduledSave();
    timeoutId = clock.setTimeout(() => save(false), saveDelayMs);
    // Reported outside the aggregate's own notification, as a status change is itself an aggregate change.
    queueMicrotask(() => {
      if (!disposed && scheduledRevision === revision && !isSaveInFlight) {
        aggregate.saveScheduled();
      }
    });
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
      if (disposed || loadGeneration !== generation) {
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
    if (!disposed && loadGeneration === generation) {
      hasLoaded = true;
      aggregate.reportLoadAvailable();
      aggregate.setHasHydrated(true);
      publish({ error: null, phase: 'idle' });
      scheduleSave();
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
    retryLoad() {
      if (disposed || hasLoaded || snapshot.phase === 'hydrating') {
        return;
      }
      generation += 1;
      void load();
    },
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
