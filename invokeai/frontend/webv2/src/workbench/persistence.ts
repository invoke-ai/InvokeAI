import type { Project, WorkbenchPersistenceSnapshot, WorkbenchState } from './types';

const STORAGE_KEY = 'invokeai:v7:webv2:workbench';
const WORKBENCH_SCHEMA_VERSION = 1;

export interface WorkbenchPersistenceService {
  loadWorkbench(): Promise<WorkbenchPersistenceSnapshot | null>;
  saveWorkbench(state: WorkbenchState): Promise<WorkbenchPersistenceSnapshot>;
  clearWorkbench(): Promise<void>;
}

export interface ProjectPersistenceService {
  loadProjects(): Promise<Project[]>;
  saveProjects(projects: Project[]): Promise<Project[]>;
  clearProjects(): Promise<void>;
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

    return Promise.resolve(migrateWorkbenchPersistenceSnapshot(JSON.parse(value)));
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

export const localStorageProjectPersistence: ProjectPersistenceService = {
  async clearProjects() {
    await localStorageWorkbenchPersistence.clearWorkbench();
  },
  async loadProjects() {
    const snapshot = await localStorageWorkbenchPersistence.loadWorkbench();

    return snapshot?.state.projects ?? [];
  },
  async saveProjects(projects) {
    const snapshot = await localStorageWorkbenchPersistence.loadWorkbench();

    if (!snapshot) {
      return projects;
    }

    await localStorageWorkbenchPersistence.saveWorkbench({ ...snapshot.state, projects });

    return projects;
  },
};
