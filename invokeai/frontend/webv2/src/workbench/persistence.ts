import type { WorkbenchPersistenceSnapshot, WorkbenchState } from './types';

const STORAGE_KEY = 'invokeai:v7:webv2:workbench';

export interface WorkbenchPersistenceService {
  loadWorkbench(): Promise<WorkbenchPersistenceSnapshot | null>;
  saveWorkbench(state: WorkbenchState): Promise<WorkbenchPersistenceSnapshot>;
  clearWorkbench(): Promise<void>;
}

const isBrowser = (): boolean => typeof window !== 'undefined' && typeof window.localStorage !== 'undefined';

const createSnapshot = (state: WorkbenchState): WorkbenchPersistenceSnapshot => ({
  savedAt: new Date().toISOString(),
  state,
  version: 1,
});

export const localStorageWorkbenchPersistence: WorkbenchPersistenceService = {
  clearWorkbench() {
    if (!isBrowser()) {
      return Promise.resolve();
    }

    window.localStorage.removeItem(STORAGE_KEY);

    return Promise.resolve();
  },
  loadWorkbench() {
    if (!isBrowser()) {
      return Promise.resolve(null);
    }

    const value = window.localStorage.getItem(STORAGE_KEY);

    if (!value) {
      return Promise.resolve(null);
    }

    return Promise.resolve(JSON.parse(value) as WorkbenchPersistenceSnapshot);
  },
  saveWorkbench(state) {
    const snapshot = createSnapshot(state);

    if (!isBrowser()) {
      return Promise.resolve(snapshot);
    }

    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshot));

    return Promise.resolve(snapshot);
  },
};
