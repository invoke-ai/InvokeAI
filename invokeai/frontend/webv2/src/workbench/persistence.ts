import { getUserStorageScope } from './auth/session';
import type { WorkbenchPersistenceSnapshot, WorkbenchState } from './types';

const BASE_STORAGE_KEY = 'invokeai:v7:webv2:workbench';
const WORKBENCH_SCHEMA_VERSION = 1;

/**
 * Projects and the editor session are account-owned on multi-user backends, so
 * each signed-in user gets their own storage bucket on this browser.
 * Single-user mode keeps the unscoped key, so existing data is untouched.
 */
const getStorageKey = (): string => `${BASE_STORAGE_KEY}${getUserStorageScope()}`;

export interface WorkbenchPersistenceService {
  loadWorkbench(): Promise<WorkbenchPersistenceSnapshot | null>;
  saveWorkbench(state: WorkbenchState): Promise<WorkbenchPersistenceSnapshot>;
  clearWorkbench(): Promise<void>;
}

const isBrowser = (): boolean => typeof window !== 'undefined' && typeof window.localStorage !== 'undefined';

const createSnapshot = (state: WorkbenchState): WorkbenchPersistenceSnapshot => ({
  savedAt: new Date().toISOString(),
  state,
  version: WORKBENCH_SCHEMA_VERSION,
});

const isWorkbenchState = (value: unknown): value is WorkbenchState => {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const record = value as Record<string, unknown>;

  return Array.isArray(record.projects) && typeof record.activeProjectId === 'string';
};

export const migrateWorkbenchPersistenceSnapshot = (value: unknown): WorkbenchPersistenceSnapshot | null => {
  if (!value || typeof value !== 'object') {
    return null;
  }

  const record = value as Partial<WorkbenchPersistenceSnapshot> & { schemaVersion?: number };
  const version = record.version ?? record.schemaVersion;

  if (version !== WORKBENCH_SCHEMA_VERSION || !isWorkbenchState(record.state)) {
    return null;
  }

  return {
    savedAt: typeof record.savedAt === 'string' ? record.savedAt : new Date().toISOString(),
    state: record.state,
    version: WORKBENCH_SCHEMA_VERSION,
  };
};

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

    return Promise.resolve(migrateWorkbenchPersistenceSnapshot(JSON.parse(value)));
  },
  saveWorkbench(state) {
    const snapshot = createSnapshot(state);

    if (!isBrowser()) {
      return Promise.resolve(snapshot);
    }

    window.localStorage.setItem(getStorageKey(), JSON.stringify(snapshot));

    return Promise.resolve(snapshot);
  },
};
