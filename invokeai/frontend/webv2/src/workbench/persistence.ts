import type { HydratedWorkbenchSnapshot, PersistedWorkbenchSnapshotV1 } from '@workbench/persistenceContracts';
import type { WorkbenchState } from '@workbench/projectContracts';

import { getUserStorageScope } from '@features/identity';

import { timeWorkbenchPerf } from './performanceMarks';

const BASE_STORAGE_KEY = 'invokeai:v7:webv2:workbench';
const WORKBENCH_SCHEMA_VERSION = 1;

/**
 * Projects and the editor session are account-owned on multi-user backends, so
 * each signed-in user gets their own storage bucket on this browser.
 * Single-user mode keeps the unscoped key, so existing data is untouched.
 */
const getStorageKey = (): string => `${BASE_STORAGE_KEY}${getUserStorageScope()}`;

export interface WorkbenchPersistenceService {
  loadWorkbench(): Promise<HydratedWorkbenchSnapshot | null>;
  saveWorkbench(state: WorkbenchState): Promise<HydratedWorkbenchSnapshot>;
  clearWorkbench(): Promise<void>;
}

const isBrowser = (): boolean => typeof window !== 'undefined' && typeof window.localStorage !== 'undefined';

export const stripTransientWorkbenchState = (state: WorkbenchState): WorkbenchState => {
  const { errorLog: _legacyErrorLog, ...nextState } = state as WorkbenchState & { errorLog?: string[] };

  return {
    ...nextState,
    notifications: [],
    // Project undo/redo is deliberately session-only. Normalize legacy cache
    // snapshots immediately and never let full-project undo entries consume
    // localStorage quota or grow across browser sessions.
    projects: nextState.projects.map((project) => ({
      ...project,
      undoRedo: { future: [], past: [] },
    })),
  };
};

const createSnapshot = (state: WorkbenchState): HydratedWorkbenchSnapshot => ({
  savedAt: new Date().toISOString(),
  state: stripTransientWorkbenchState(state),
  version: WORKBENCH_SCHEMA_VERSION,
});

const isWorkbenchState = (value: unknown): value is WorkbenchState => {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const record = value as Record<string, unknown>;

  return Array.isArray(record.projects) && typeof record.activeProjectId === 'string';
};

export const hydratePersistedWorkbenchSnapshot = (value: unknown): HydratedWorkbenchSnapshot | null => {
  if (!value || typeof value !== 'object') {
    return null;
  }

  const record = value as Partial<PersistedWorkbenchSnapshotV1> & { schemaVersion?: number };
  const version = record.version ?? record.schemaVersion;

  if (version !== WORKBENCH_SCHEMA_VERSION || !isWorkbenchState(record.state)) {
    return null;
  }

  return {
    savedAt: typeof record.savedAt === 'string' ? record.savedAt : new Date().toISOString(),
    state: stripTransientWorkbenchState(record.state),
    version: WORKBENCH_SCHEMA_VERSION,
  };
};

export const serializeWorkbenchPersistenceSnapshot = (
  snapshot: HydratedWorkbenchSnapshot
): PersistedWorkbenchSnapshotV1 => ({
  savedAt: snapshot.savedAt,
  state: snapshot.state,
  version: WORKBENCH_SCHEMA_VERSION,
});

export const localStorageWorkbenchPersistence: WorkbenchPersistenceService = {
  clearWorkbench() {
    if (!isBrowser()) {
      return Promise.resolve();
    }

    window.localStorage.removeItem(getStorageKey());

    return Promise.resolve();
  },
  loadWorkbench() {
    if (!isBrowser()) {
      return Promise.resolve(null);
    }

    const value = window.localStorage.getItem(getStorageKey());

    if (!value) {
      return Promise.resolve(null);
    }

    try {
      return Promise.resolve(hydratePersistedWorkbenchSnapshot(JSON.parse(value)));
    } catch {
      window.localStorage.removeItem(getStorageKey());

      return Promise.resolve(null);
    }
  },
  saveWorkbench(state) {
    const snapshot = createSnapshot(state);

    if (!isBrowser()) {
      return Promise.resolve(snapshot);
    }

    try {
      window.localStorage.setItem(
        getStorageKey(),
        timeWorkbenchPerf(
          'workbench:persistence-localstorage-stringify',
          { area: 'persistence', kind: 'workbench', projectId: state.activeProjectId },
          () => JSON.stringify(serializeWorkbenchPersistenceSnapshot(snapshot))
        )
      );
    } catch {
      // The backend remains the source of truth; localStorage is only an offline cache.
    }

    return Promise.resolve(snapshot);
  },
};
