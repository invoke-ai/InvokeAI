import type { Project, WorkbenchState } from '@workbench/projectContracts';

import { createDraftProject, createInitialWorkbenchState } from '@workbench/workbenchState';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type * as libraryModule from './library';
import type * as persistenceModule from './syncedPersistence';

/**
 * Service-level tests for the library/session split: the open set drives
 * hydration, saving never deletes, and explicit deletes cannot be undone by
 * racing autosaves. The REST module is replaced by an in-memory server.
 */

const api = vi.hoisted(() => {
  interface MockRecord {
    project_id: string;
    name: string;
    revision: number;
    created_at: string;
    updated_at: string;
    data: Record<string, unknown>;
  }

  const records = new Map<string, MockRecord>();
  const clientState = new Map<string, string>();

  const conflictError = (): Error => Object.assign(new Error('conflict'), { __status: 409 });
  const notFoundError = (): Error => Object.assign(new Error('not found'), { __status: 404 });
  const toSummary = (record: MockRecord) => ({
    created_at: record.created_at,
    name: record.name,
    project_id: record.project_id,
    revision: record.revision,
    updated_at: record.updated_at,
  });
  const clone = (record: MockRecord): MockRecord => structuredClone(record);

  const mock = {
    __clientState: clientState,
    __records: records,
    __seed: (data: Record<string, unknown>): void => {
      const id = data.id as string;

      records.set(id, {
        created_at: '2026-06-10 08:00:00.000',
        data: structuredClone(data),
        name: data.name as string,
        project_id: id,
        revision: 1,
        updated_at: '2026-06-10 08:00:00.000',
      });
    },
    createProject: vi.fn((request: { project_id?: string; name: string; data: Record<string, unknown> }) => {
      const id = request.project_id ?? `generated-${records.size}`;

      if (records.has(id)) {
        return Promise.reject(conflictError());
      }

      const record: MockRecord = {
        created_at: '2026-06-10 09:00:00.000',
        data: structuredClone(request.data),
        name: request.name,
        project_id: id,
        revision: 1,
        updated_at: '2026-06-10 09:00:00.000',
      };

      records.set(id, record);

      return Promise.resolve(clone(record));
    }),
    deleteClientStateValue: vi.fn((key: string) => {
      clientState.delete(key);

      return Promise.resolve();
    }),
    deleteProject: vi.fn((projectId: string) => {
      records.delete(projectId);

      return Promise.resolve();
    }),
    getClientStateValue: vi.fn((key: string) => Promise.resolve(clientState.get(key) ?? null)),
    getProject: vi.fn((projectId: string) => {
      const record = records.get(projectId);

      return record ? Promise.resolve(clone(record)) : Promise.reject(notFoundError());
    }),
    isProjectConflictError: (error: unknown): boolean => (error as { __status?: number }).__status === 409,
    isProjectNotFoundError: (error: unknown): boolean => (error as { __status?: number }).__status === 404,
    listProjects: vi.fn(() => Promise.resolve([...records.values()].map(toSummary))),
    setClientStateValue: vi.fn((key: string, value: string) => {
      clientState.set(key, value);

      return Promise.resolve();
    }),
    updateProject: vi.fn(
      (projectId: string, request: { name: string; data: Record<string, unknown>; expected_revision: number }) => {
        const record = records.get(projectId);

        if (!record) {
          return Promise.reject(notFoundError());
        }

        if (record.revision !== request.expected_revision) {
          return Promise.reject(conflictError());
        }

        const updated: MockRecord = {
          ...record,
          data: structuredClone(request.data),
          name: request.name,
          revision: record.revision + 1,
          updated_at: '2026-06-10 10:00:00.000',
        };

        records.set(projectId, updated);

        return Promise.resolve(clone(updated));
      }
    ),
  };

  return mock;
});

vi.mock('./api', () => api);

const storage = new Map<string, string>();

vi.stubGlobal('window', {
  localStorage: {
    getItem: (key: string): string | null => storage.get(key) ?? null,
    removeItem: (key: string): void => {
      storage.delete(key);
    },
    setItem: (key: string, value: string): void => {
      storage.set(key, value);
    },
  },
});

const SESSION_KEY = 'webv2:workbench-account';

let persistence: typeof persistenceModule;
let library: typeof libraryModule;
let service: persistenceModule.SyncedWorkbenchPersistence;

const seedSessionBlob = (blob: Record<string, unknown>): void => {
  api.__clientState.set(SESSION_KEY, JSON.stringify(blob));
};

const seedServerProject = (name: string): Project => {
  const draft = { ...createDraftProject([]), name };

  api.__seed(persistence.serializeProjectDocument(draft));

  return draft;
};

const stateWithProjects = (projects: Project[], activeProjectId = projects[0]?.id ?? ''): WorkbenchState => ({
  ...createInitialWorkbenchState(),
  activeProjectId,
  projects,
});

beforeEach(async () => {
  vi.resetModules();
  api.__records.clear();
  api.__clientState.clear();
  storage.clear();
  api.createProject.mockClear();
  api.deleteProject.mockClear();
  api.getProject.mockClear();
  api.listProjects.mockClear();
  api.updateProject.mockClear();

  persistence = await import('./syncedPersistence');
  service = persistence.createSyncedWorkbenchPersistence();
  library = await import('./library');
});

describe('loadWorkbench session hydration', () => {
  it('constructs isolated synchronization lifetimes instead of sharing pending state', async () => {
    const first = persistence.createSyncedWorkbenchPersistence();
    const second = persistence.createSyncedWorkbenchPersistence();
    api.listProjects.mockRejectedValueOnce(new Error('offline'));

    await first.loadWorkbench();

    expect(first.hasPendingChanges()).toBe(true);
    expect(second.hasPendingChanges()).toBe(false);
  });

  it('hydrates only the open set and seeds the full library', async () => {
    const first = seedServerProject('First');
    const second = seedServerProject('Second');
    const third = seedServerProject('Third');
    const account = createInitialWorkbenchState().account;

    seedSessionBlob({ account, activeProjectId: second.id, openProjectIds: [second.id] });

    const snapshot = await service.loadWorkbench();

    expect(snapshot?.state.projects.map((project) => project.id)).toEqual([second.id]);
    expect(snapshot?.state.activeProjectId).toBe(second.id);
    expect(api.getProject).toHaveBeenCalledTimes(1);
    expect(service.hasPendingChanges()).toBe(false);

    const libraryIds = library.getProjectLibrary().summaries.map((summary) => summary.id);

    expect(libraryIds).toHaveLength(3);
    expect(libraryIds).toEqual(expect.arrayContaining([first.id, second.id, third.id]));
  });

  it('opens every project for sessions from before the split (no open set in the blob)', async () => {
    seedServerProject('First');
    seedServerProject('Second');

    const account = createInitialWorkbenchState().account;

    seedSessionBlob({ account, activeProjectId: 'missing' });

    const snapshot = await service.loadWorkbench();

    expect(snapshot?.state.projects).toHaveLength(2);
  });

  it('ignores a corrupt local cache and still hydrates from the backend', async () => {
    seedServerProject('Backend Project');
    storage.set('invokeai:v7:webv2:workbench', '{not json');

    const snapshot = await service.loadWorkbench();

    expect(snapshot?.state.projects.map((project) => project.name)).toEqual(['Backend Project']);
  });

  it('boots a fresh draft when the session is empty', async () => {
    const existing = seedServerProject('Closed project');
    const account = createInitialWorkbenchState().account;

    seedSessionBlob({ account, activeProjectId: '', openProjectIds: [] });

    const snapshot = await service.loadWorkbench();

    expect(snapshot?.state.projects).toHaveLength(1);
    expect(snapshot?.state.projects[0].id).not.toBe(existing.id);
    expect(snapshot?.state.activeProjectId).toBe(snapshot?.state.projects[0].id);
    expect(service.hasPendingChanges()).toBe(true);
  });

  it('joins a deep-linked project into the open set and focuses it', async () => {
    const first = seedServerProject('First');
    const second = seedServerProject('Second');
    const account = createInitialWorkbenchState().account;

    seedSessionBlob({ account, activeProjectId: first.id, openProjectIds: [first.id] });

    const snapshot = await service.loadWorkbench({ openProjectId: second.id });

    expect(snapshot?.state.projects.map((project) => project.id)).toEqual([first.id, second.id]);
    expect(snapshot?.state.activeProjectId).toBe(second.id);
    expect(service.hasPendingChanges()).toBe(true);
  });

  it('appends and activates a draft when a new project is requested', async () => {
    const first = seedServerProject('First');
    const account = createInitialWorkbenchState().account;

    seedSessionBlob({ account, activeProjectId: first.id, openProjectIds: [first.id] });

    const snapshot = await service.loadWorkbench({ createNew: true });

    expect(snapshot?.state.projects).toHaveLength(2);
    expect(snapshot?.state.activeProjectId).not.toBe(first.id);
    expect(service.hasPendingChanges()).toBe(true);
  });
});

describe('saveWorkbench', () => {
  it('never deletes server projects that are absent from state', async () => {
    const first = seedServerProject('First');
    const second = seedServerProject('Second');
    const account = createInitialWorkbenchState().account;

    seedSessionBlob({ account, activeProjectId: first.id, openProjectIds: [first.id, second.id] });

    const snapshot = await service.loadWorkbench();
    const open = snapshot?.state.projects ?? [];

    expect(open).toHaveLength(2);

    // Close the second tab: it leaves state, but must stay on the server.
    const closed = stateWithProjects(
      open.filter((project) => project.id !== second.id),
      first.id
    );

    await service.saveWorkbench(closed);

    expect(api.deleteProject).not.toHaveBeenCalled();
    expect(api.__records.has(second.id)).toBe(true);
  });

  it('persists the open set in the session blob', async () => {
    const first = seedServerProject('First');
    const account = createInitialWorkbenchState().account;

    seedSessionBlob({ account, activeProjectId: first.id, openProjectIds: [first.id] });

    const snapshot = await service.loadWorkbench();
    const open = snapshot?.state.projects ?? [];
    const draft = createDraftProject(open);

    await service.saveWorkbench(stateWithProjects([...open, draft], draft.id));

    const blob = JSON.parse(api.__clientState.get(SESSION_KEY) ?? '{}') as {
      activeProjectId?: string;
      openProjectIds?: string[];
    };

    expect(blob.openProjectIds).toEqual([first.id, draft.id]);
    expect(blob.activeProjectId).toBe(draft.id);
  });

  it('skips pushes for projects marked deleted, so a racing autosave cannot resurrect them', async () => {
    const project = createDraftProject([]);
    const state = stateWithProjects([project]);

    service.markProjectDeleted(project.id);
    await service.saveWorkbench(state);

    expect(api.createProject).not.toHaveBeenCalled();
    expect(api.updateProject).not.toHaveBeenCalled();
    expect(api.__records.has(project.id)).toBe(false);
  });
});

describe('persistEmptySession', () => {
  it('writes an empty open set without touching project records', async () => {
    const first = seedServerProject('First');
    const account = createInitialWorkbenchState().account;

    seedSessionBlob({ account, activeProjectId: first.id, openProjectIds: [first.id] });

    const snapshot = await service.loadWorkbench();

    await service.persistEmptySession(snapshot?.state ?? createInitialWorkbenchState());

    const blob = JSON.parse(api.__clientState.get(SESSION_KEY) ?? '{}') as { openProjectIds?: string[] };

    expect(blob.openProjectIds).toEqual([]);
    expect(api.__records.has(first.id)).toBe(true);
    expect(api.deleteProject).not.toHaveBeenCalled();
  });
});

describe('hydrateProjectFromServer', () => {
  it('returns an openable project and registers its revision for future saves', async () => {
    const project = seedServerProject('Closed project');
    const hydrated = await service.hydrateProjectFromServer(project.id);

    expect(hydrated?.id).toBe(project.id);
    expect(hydrated?.undoRedo).toEqual({ future: [], past: [] });

    // A subsequent save updates in place rather than re-creating.
    const renamed = { ...hydrated!, name: 'Renamed after reopen' };

    await service.saveWorkbench(stateWithProjects([renamed]));

    expect(api.createProject).not.toHaveBeenCalled();
    expect(api.__records.get(project.id)?.name).toBe('Renamed after reopen');
  });

  it('returns null for unknown projects', async () => {
    expect(await service.hydrateProjectFromServer('nope')).toBeNull();
  });
});
