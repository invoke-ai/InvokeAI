import type * as accountLifecycleModule from '@platform/state/accountLifecycle';
import type { Project, WorkbenchState } from '@workbench/projectContracts';

import { createWorkbenchPersistenceRuntime, type PersistenceClock } from '@workbench/persistenceRuntime';
import { createDraftProject, createInitialWorkbenchState } from '@workbench/workbenchState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
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
    /** Every project owns one, and the server is the only thing that decides which. */
    board_id: string;
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
    board_id: record.board_id,
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
        board_id: `board-for-${id}`,
        created_at: '2026-06-10 08:00:00.000',
        data: structuredClone(data),
        name: data.name as string,
        project_id: id,
        revision: 1,
        updated_at: '2026-06-10 08:00:00.000',
      });
    },
    createProject: vi.fn(
      (request: { project_id?: string; board_id?: string; name: string; data: Record<string, unknown> }) => {
        const id = request.project_id ?? `generated-${records.size}`;

        if (records.has(id)) {
          return Promise.reject(conflictError());
        }

        const record: MockRecord = {
          board_id: request.board_id ?? `board-for-${id}`,
          created_at: '2026-06-10 09:00:00.000',
          data: structuredClone(request.data),
          name: request.name,
          project_id: id,
          revision: 1,
          updated_at: '2026-06-10 09:00:00.000',
        };

        records.set(id, record);

        return Promise.resolve(clone(record));
      }
    ),
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

const deferred = <T>() => {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });

  return { promise, reject, resolve };
};

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
let account: typeof accountLifecycleModule;
let service: persistenceModule.SyncedWorkbenchPersistence;
const defaultUpdateProject = api.updateProject.getMockImplementation()!;
const defaultSetClientStateValue = api.setClientStateValue.getMockImplementation()!;

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
  api.updateProject.mockReset();
  api.updateProject.mockImplementation(defaultUpdateProject);
  api.setClientStateValue.mockReset();
  api.setClientStateValue.mockImplementation(defaultSetClientStateValue);

  account = await import('@platform/state/accountLifecycle');
  account.accountLifecycle.activate('single-user', '');
  persistence = await import('./syncedPersistence');
  service = persistence.createSyncedWorkbenchPersistence(account.captureAccountScope());
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

  it('shares one backend import across StrictMode-style load replays', async () => {
    const state = createInitialWorkbenchState();
    storage.set(
      'invokeai:v7:webv2:workbench',
      JSON.stringify({ savedAt: '2026-07-19T00:00:00.000Z', state, version: 1 })
    );

    const [first, replay] = await Promise.all([service.loadWorkbench(), service.loadWorkbench()]);

    expect(replay).toBe(first);
    expect(api.listProjects).toHaveBeenCalledTimes(1);
    expect(api.createProject).toHaveBeenCalledTimes(state.projects.length);
    expect(api.getProject).not.toHaveBeenCalled();
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

  it('still starts a draft for a new-project request when the backend is unreachable', async () => {
    const state = createInitialWorkbenchState();
    const cachedProjectId = state.projects[0]?.id ?? '';

    storage.set(
      'invokeai:v7:webv2:workbench',
      JSON.stringify({ savedAt: '2026-07-19T00:00:00.000Z', state, version: 1 })
    );
    api.listProjects.mockRejectedValueOnce(new Error('offline'));

    const snapshot = await service.loadWorkbench({ createNew: true });

    // Returning the cache verbatim here used to reopen whichever project was
    // last active, so an offline "New project" landed the user in existing work
    // — and let the Launchpad's intent rearrange it.
    expect(snapshot?.state.projects).toHaveLength(state.projects.length + 1);
    expect(snapshot?.state.activeProjectId).not.toBe(cachedProjectId);
  });

  it('reopens the cached session when the backend is unreachable and no draft was requested', async () => {
    const state = createInitialWorkbenchState();

    storage.set(
      'invokeai:v7:webv2:workbench',
      JSON.stringify({ savedAt: '2026-07-19T00:00:00.000Z', state, version: 1 })
    );
    api.listProjects.mockRejectedValueOnce(new Error('offline'));

    const snapshot = await service.loadWorkbench();

    expect(snapshot?.state.projects).toHaveLength(state.projects.length);
    expect(snapshot?.state.activeProjectId).toBe(state.activeProjectId);
  });
});

describe('saveWorkbench', () => {
  it('cancels an account-A save queued behind the mutation microtask before it can touch account B', async () => {
    const ownerA = account.accountLifecycle.activate('user-a', ':user:a');
    const accountAService = persistence.createSyncedWorkbenchPersistence(ownerA);
    const stateA = createInitialWorkbenchState();

    stateA.projects[0]!.name = 'Account A queued edit';
    const queuedSave = accountAService.saveWorkbench(stateA);

    account.accountLifecycle.activate('user-b', ':user:b');

    await expect(queuedSave).rejects.toHaveProperty('name', 'AccountScopeExpiredError');
    expect(storage.has('invokeai:v7:webv2:workbench:user:a')).toBe(false);
    expect(storage.has('invokeai:v7:webv2:workbench:user:b')).toBe(false);
    expect(api.setClientStateValue).not.toHaveBeenCalled();
    expect(api.createProject).not.toHaveBeenCalled();
    expect(api.updateProject).not.toHaveBeenCalled();
  });

  it('stops an in-flight account-A save before the next backend request after account B activates', async () => {
    const ownerA = account.accountLifecycle.activate('user-a', ':user:a');
    const accountAService = persistence.createSyncedWorkbenchPersistence(ownerA);
    const stateA = createInitialWorkbenchState();
    const sessionWrite = deferred<void>();

    stateA.projects[0]!.name = 'Account A in-flight edit';
    api.setClientStateValue.mockImplementationOnce(() => sessionWrite.promise);

    const inFlightSave = accountAService.saveWorkbench(stateA);

    await vi.waitFor(() => expect(api.setClientStateValue).toHaveBeenCalledOnce());
    expect(storage.has('invokeai:v7:webv2:workbench:user:a')).toBe(true);

    account.accountLifecycle.activate('user-b', ':user:b');
    sessionWrite.resolve();

    await expect(inFlightSave).rejects.toHaveProperty('name', 'AccountScopeExpiredError');
    expect(storage.has('invokeai:v7:webv2:workbench:user:b')).toBe(false);
    // The already-started client-state request belonged to A. Its late
    // completion must not advance the chain to a project create under B.
    expect(api.createProject).not.toHaveBeenCalled();
    expect(api.updateProject).not.toHaveBeenCalled();
  });

  it('serializes concurrent saves against the latest acknowledged project revision', async () => {
    const project = seedServerProject('Original');
    const account = createInitialWorkbenchState().account;

    seedSessionBlob({ account, activeProjectId: project.id, openProjectIds: [project.id] });

    const loaded = await service.loadWorkbench();
    const original = loaded!.state.projects[0]!;
    const firstState = stateWithProjects([{ ...original, name: 'First edit' }]);
    const latestState = stateWithProjects([{ ...original, name: 'Latest edit' }]);
    const defaultUpdate = api.updateProject.getMockImplementation()!;
    const writes: {
      request: { name: string; data: Record<string, unknown>; expected_revision: number };
      resolve(value: Awaited<ReturnType<typeof defaultUpdate>>): void;
    }[] = [];

    api.updateProject.mockImplementation((_projectId, request) => {
      return new Promise((resolve) => {
        writes.push({ request, resolve });
      });
    });

    const firstSave = service.saveWorkbench(firstState);
    const latestSave = service.saveWorkbench(latestState);

    await vi.waitFor(() => expect(writes).toHaveLength(1));
    const firstWinner = await defaultUpdate(project.id, writes[0]!.request);

    writes[0]!.resolve(firstWinner);
    await vi.waitFor(() => expect(writes).toHaveLength(2));
    const latestWinner = await defaultUpdate(project.id, writes[1]!.request);

    writes[1]!.resolve(latestWinner);

    const results = await Promise.all([firstSave, latestSave]);

    expect(writes.map((write) => write.request.expected_revision)).toEqual([1, 2]);
    expect(results.flatMap((result) => result.conflicts)).toEqual([]);
    expect(api.__records.get(project.id)?.name).toBe('Latest edit');
  });

  it('serializes reconnect replay behind an in-flight save instead of creating and activating a recovery', async () => {
    const first = seedServerProject('First');
    const second = seedServerProject('Second');
    const account = createInitialWorkbenchState().account;

    seedSessionBlob({ account, activeProjectId: second.id, openProjectIds: [first.id, second.id] });

    const loaded = await service.loadWorkbench();
    const store = createWorkbenchStore(loaded!.state);
    const callbacks = new Map<number, () => void>();
    let nextTimerId = 0;
    const clock: PersistenceClock & { runAll(): void } = {
      clearTimeout: (id) => callbacks.delete(id as number),
      runAll: () => {
        const pending = [...callbacks.values()];
        callbacks.clear();
        for (const callback of pending) {
          callback();
        }
      },
      setTimeout: (callback) => {
        nextTimerId += 1;
        callbacks.set(nextTimerId, callback);
        return nextTimerId;
      },
    };
    const runtime = createWorkbenchPersistenceRuntime({
      aggregate: {
        ...store.internal.persistence,
        getPersistedRevision: store.getPersistedRevision,
        notifyProjectNotFound: vi.fn(),
        reportLoadError: vi.fn(),
        setHasHydrated: store.setHasHydrated,
        subscribe: store.subscribe,
      },
      clock,
      persistence: service,
    });

    runtime.start();
    await vi.waitFor(() => expect(store.getSnapshot().hasHydrated).toBe(true));
    store.commands.queue.setConnectionStatus({ status: 'connected' });
    store.commands.projects.rename(first.id, 'First local edit');
    store.commands.projects.rename(second.id, 'Second local edit A');

    const defaultUpdate = api.updateProject.getMockImplementation()!;
    let rejectedFirstProject = false;
    const secondProjectWrites: {
      reject(error: unknown): void;
      request: { name: string; data: Record<string, unknown>; expected_revision: number };
      resolve(value: Awaited<ReturnType<typeof defaultUpdate>>): void;
    }[] = [];

    api.updateProject.mockImplementation((projectId, request) => {
      if (projectId === first.id && !rejectedFirstProject) {
        rejectedFirstProject = true;
        return Promise.reject(new Error('transient HTTP failure'));
      }
      if (projectId === second.id) {
        return new Promise((resolve, reject) => {
          secondProjectWrites.push({ reject, request, resolve });
        });
      }
      return defaultUpdate(projectId, request);
    });

    clock.runAll();
    await vi.waitFor(() => expect(secondProjectWrites).toHaveLength(1));

    store.commands.projects.rename(second.id, 'Second local edit B');
    store.commands.queue.setConnectionStatus({ status: 'disconnected' });
    store.commands.queue.setConnectionStatus({ status: 'connected' });
    await new Promise<void>((resolve) => {
      setTimeout(resolve, 0);
    });

    const didOverlap = secondProjectWrites.length > 1;
    const firstWinner = await defaultUpdate(second.id, secondProjectWrites[0]!.request);

    secondProjectWrites[0]!.resolve(firstWinner);

    if (didOverlap) {
      secondProjectWrites[1]!.reject(Object.assign(new Error('conflict'), { __status: 409 }));
    } else {
      await vi.waitFor(() => expect(secondProjectWrites).toHaveLength(2));
      const secondWinner = await defaultUpdate(second.id, secondProjectWrites[1]!.request);

      secondProjectWrites[1]!.resolve(secondWinner);
    }

    await vi.waitFor(() => expect(runtime.getSnapshot().phase).toBe('idle'));

    const recoveryRecords = [...api.__records.values()].filter((record) => record.data.recoveryOf === second.id);

    expect(secondProjectWrites.map((write) => write.request.expected_revision)).toEqual([1, 2]);
    expect(recoveryRecords).toHaveLength(0);
    expect(store.getState().activeProjectId).toBe(second.id);
    runtime.dispose();
  });

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

/**
 * The document's `projectBoardId` is a cache of a relationship SQLite owns. Every path that learns
 * the real answer has to write it down, or the project points at a board that means nothing here.
 */
describe('authoritative project boards', () => {
  const galleryBoardIds = (project: Project): { projectBoardId?: unknown; selectedBoardId?: unknown } => {
    const instance = Object.values(project.widgetInstances).find((entry) => entry.typeId === 'gallery');

    return (instance?.state.values ?? {}) as { projectBoardId?: unknown; selectedBoardId?: unknown };
  };

  it('reports the board the server minted for a project it created', async () => {
    const draft = createDraftProject([]);

    const result = await service.saveWorkbench(stateWithProjects([draft]));

    expect(result.projectBoardAssignments).toEqual([{ boardId: `board-for-${draft.id}`, projectId: draft.id }]);
    // Drained, so the next save does not re-apply it.
    expect((await service.saveWorkbench(stateWithProjects([draft]))).projectBoardAssignments).toEqual([]);
  });

  it('overwrites a stale board id when hydrating a record', async () => {
    const draft = createDraftProject([]);
    const document = persistence.serializeProjectDocument(draft) as Record<string, unknown>;
    const instances = document.widgetInstances as Record<string, { state: { values: Record<string, unknown> } }>;
    const galleryId = Object.keys(instances).find(
      (key) => (instances[key] as unknown as { typeId: string }).typeId === 'gallery'
    )!;

    // A board id written by some other install, naming a board that does not exist here.
    instances[galleryId]!.state.values.projectBoardId = 'board-from-another-machine';
    instances[galleryId]!.state.values.selectedBoardId = 'deliberate-destination';
    api.__seed(document);

    const hydrated = await service.hydrateProjectFromServer(draft.id);

    expect(galleryBoardIds(hydrated!).projectBoardId).toBe(`board-for-${draft.id}`);
    // The chosen destination is the user's, not ours; resolving it is the gallery's job.
    expect(galleryBoardIds(hydrated!).selectedBoardId).toBe('deliberate-destination');
  });

  it('forks rather than resurrects a project deleted on another device', async () => {
    const project = seedServerProject('Deleted elsewhere');
    const opened = await service.hydrateProjectFromServer(project.id);
    const edited = { ...opened!, name: 'Edited locally' };

    // Deleted on the other device, after this one had already synced.
    api.__records.delete(project.id);

    const result = await service.saveWorkbench(stateWithProjects([edited]));

    // The deletion stands...
    expect(api.__records.has(project.id)).toBe(false);
    // ...and the work survives under a fresh id, with a board of its own.
    expect(result.deletedProjectForks).toHaveLength(1);
    const [fork] = result.deletedProjectForks;
    expect(fork!.projectId).toBe(project.id);
    expect(fork!.recoveredProject.id).not.toBe(project.id);
    expect(fork!.recoveredProject.name).toBe('Edited locally (recovered)');
    expect(api.__records.get(fork!.recoveredProject.id)).toBeDefined();
    expect(result.projectBoardAssignments).toEqual([
      { boardId: `board-for-${fork!.recoveredProject.id}`, projectId: fork!.recoveredProject.id },
    ]);
  });
});
