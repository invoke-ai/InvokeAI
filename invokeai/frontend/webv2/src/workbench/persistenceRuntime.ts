import type { HydratedWorkbenchSnapshot } from '@workbench/persistenceContracts';
import type { WorkbenchState } from '@workbench/projectContracts';

import type { WorkbenchLoadOptions, WorkbenchSaveResult } from './projects/syncedPersistence';

export interface PersistenceAggregatePort {
  getPersistedRevision(): number;
  getState(): WorkbenchState;
  hydrate(state: WorkbenchState): void;
  notifyProjectNotFound(): void;
  reconcileConflict(conflict: WorkbenchSaveResult['conflicts'][number]): void;
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
}: {
  aggregate: PersistenceAggregatePort;
  clock?: PersistenceClock;
  loadOptions?: WorkbenchLoadOptions;
  persistence: WorkbenchPersistencePort;
  saveDelayMs?: number;
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
    if (isStaleSave(revision, saveGeneration, requireCurrentRevision)) {
      return;
    }
    lastSavedRevision = revision;
    failedRevision = null;
    scheduledRevision = null;
    aggregate.saveSucceeded(result.snapshot.savedAt);
    applySaveResult(result);
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
    const state = aggregate.getState();
    const revision = aggregate.getPersistedRevision();
    generation += 1;
    const saveGeneration = generation;

    aggregate.saveStarted();
    publish({ error: null, phase: 'saving' });
    void persistence
      .saveWorkbench(state)
      .then((result) => completeSave(result, revision, saveGeneration, requireCurrentRevision))
      .catch((error: unknown) => failSave(error, revision, saveGeneration, requireCurrentRevision));
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

  return {
    dispose() {
      if (disposed) {
        return;
      }
      disposed = true;
      generation += 1;
      clearScheduledSave();
      unsubscribeAggregate?.();
      unsubscribeAggregate = null;
      snapshot = { error: null, phase: 'disposed' };
      listeners.clear();
    },
    getSnapshot: () => snapshot,
    start() {
      if (started || disposed) {
        return;
      }
      started = true;
      unsubscribeAggregate = aggregate.subscribe(onAggregateChange);
      void load();
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
};
